"""Fit positive temperatures on held-out labels; score disjoint test predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

from evaluate_calibration import evaluate_predictions, write_tables, plot_reliability
from prepare_chexpert_splits import sha256, CLASS_NAMES
from study_io import atomic_json, read_json


def load_export(folder):
    folder = Path(folder)
    result = {name: np.load(folder / (name + ".npy"), allow_pickle=False)
              for name in ("logits", "y_gt", "y_pred", "label_mask")}
    shape = result["logits"].shape
    if len(shape) != 2 or shape[1] != 5 or shape[0] < 1 or any(x.shape != shape for x in result.values()):
        raise ValueError("Invalid prediction shapes")
    if any(not np.isfinite(x).all() for x in result.values()):
        raise ValueError("Non-finite predictions")
    if not np.isin(result["label_mask"], [0, 1]).all() or not np.isin(result["y_gt"], [0, 1]).all():
        raise ValueError("Calibration exports require binary observed labels and masks")
    result["label_mask"] = result["label_mask"].astype(bool)
    if not np.allclose(expit(result["logits"]), result["y_pred"], atol=2e-7, rtol=2e-6):
        raise ValueError("Saved probability/log odds mismatch")
    result["samples"] = read_json(folder / "samples.json")
    if len(result["samples"]) != shape[0] or len({s["path"] for s in result["samples"]}) != shape[0]:
        raise ValueError("Missing or duplicated sample identifiers")
    result["evaluation"] = read_json(folder / "evaluation.json")
    result["folder"] = folder
    return result


def binary_nll(logits, targets):
    return np.logaddexp(0, logits) - targets * logits


def fit_temperatures(logits, labels, mask, per_class=False):
    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if logits.shape != labels.shape or mask.shape != labels.shape or not mask.any():
        raise ValueError("No valid calibration labels or incompatible arrays")
    temperatures, details = [], []
    for column in range(logits.shape[1]) if per_class else [None]:
        selected = mask if column is None else mask[:, column]
        z = logits[selected] if column is None else logits[:, column][selected]
        y = labels[selected] if column is None else labels[:, column][selected]
        identity = float(binary_nll(z, y).mean()) if len(y) else None
        if not len(y) or np.unique(y).size < 2:
            temperatures.append(1.0)
            details.append(dict(column=column, observed=len(y), fallback="no two-class labels", nll_before=identity))
            continue
        def cost(log_temperature):
            return float(binary_nll(z / np.exp(log_temperature), y).mean())
        fit = minimize_scalar(cost, bounds=(np.log(0.01), np.log(100.0)), method="bounded",
                              options={"xatol": 1e-7, "maxiter": 300})
        if not fit.success or not np.isfinite(fit.fun):
            raise ValueError("Temperature fitting did not converge")
        t = float(np.exp(fit.x)) if fit.fun <= identity else 1.0
        temperatures.append(t)
        details.append(dict(column=column, observed=len(y), nll_before=identity, nll_after=cost(np.log(t)),
                            at_bound=bool(t < .0101 or t > 99.9)))
    return np.array(temperatures), details


def flattened(report):
    rows = []
    for name, row in [(r["class"], r) for r in report["per_class"]] + [("macro", report["macro"])]:
        metrics = {key: row[key] for key in ("auroc", "brier", "nll")}
        for bins in report["bin_counts"]:
            b = row["calibration"][str(bins)]
            metrics["ece_" + str(bins)] = b["ece"] if name == "macro" else b["uniform"]["ece"]
            metrics["adaptive_ece_" + str(bins)] = b["adaptive_ece"] if name == "macro" else b["adaptive"]["ece"]
        rows.append(dict(metrics, disease=name))
    return rows


def postprocess(calibration_dir, test_dir, output, bins=(5, 10), plots=True, expert_fraction=0.5, split_seed=42):
    cal, test = load_export(calibration_dir), load_export(test_dir)
    if cal["evaluation"]["split"] != "calibration" or test["evaluation"]["split"] != "test":
        raise ValueError("Expected calibration and test splits")
    if cal["evaluation"]["checkpoint_sha256"] != test["evaluation"]["checkpoint_sha256"]:
        raise ValueError("Calibration and test must come from the same checkpoint")
    if {s["patient_id"] for s in cal["samples"]} & {s["patient_id"] for s in test["samples"]}:
        raise ValueError("Calibration/test patient leakage")
    if not test["label_mask"].all():
        raise ValueError("Expert test must have complete binary labels")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocols = {"report_holdout": (cal["logits"], cal["y_gt"], cal["label_mask"], np.arange(len(test["samples"])))}
    if expert_fraction:
        if not 0 < expert_fraction < 1:
            raise ValueError("Expert calibration fraction must be in (0,1)")
        patients = sorted({s["patient_id"] for s in test["samples"]})
        rng = np.random.RandomState(split_seed)
        rng.shuffle(patients)
        n = max(1, int(len(patients) * expert_fraction))
        if n >= len(patients):
            raise ValueError("Too few expert patients for disjoint calibration/test")
        fitting = set(patients[:n])
        fit_indices = np.array([i for i, s in enumerate(test["samples"]) if s["patient_id"] in fitting])
        test_indices = np.array([i for i, s in enumerate(test["samples"]) if s["patient_id"] not in fitting])
        protocols["expert_subset"] = (test["logits"][fit_indices], test["y_gt"][fit_indices],
                                       test["label_mask"][fit_indices], test_indices)
    summary = {"class_order": list(CLASS_NAMES), "protocols": {}, "inputs": {
        name: {f: sha256(record["folder"] / f) for f in ("logits.npy", "y_gt.npy", "label_mask.npy", "samples.json", "evaluation.json")}
        for name, record in (("calibration", cal), ("test", test))}}
    for protocol, (z, y, mask, indices) in protocols.items():
        folder = output / protocol
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "test_indices.npy", indices)
        methods, rows = {}, []
        for mode in ("raw", "ts_shared", "ts_per_class"):
            temperatures, fit = (np.ones(1), []) if mode == "raw" else fit_temperatures(z, y, mask, mode == "ts_per_class")
            probabilities = expit(test["logits"][indices].astype(np.float64) / temperatures)
            report = evaluate_predictions(test["y_gt"][indices], probabilities, bins)
            dest = folder / mode
            dest.mkdir(exist_ok=True)
            np.save(dest / "y_pred.npy", probabilities)
            write_tables(report, dest)
            if plots:
                plot_reliability(report, dest)
            methods[mode] = dict(temperatures=temperatures.tolist(), fitting=fit, metrics=report)
            rows.extend(dict(row, calibration=mode) for row in flattened(report))
        summary["protocols"][protocol] = dict(n_test_images=len(indices),
            n_test_patients=len({test["samples"][i]["patient_id"] for i in indices}),
            calibration_labels=int(mask.sum()), methods=methods, rows=rows)
    atomic_json(output / "summary.json", summary)
    print("POSTPROCESS COMPLETE: " + str(output / "summary.json"), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expert-fraction", type=float, default=.5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    postprocess(args.calibration_dir, args.test_dir, args.output_dir, plots=not args.no_plots,
                expert_fraction=args.expert_fraction, split_seed=args.split_seed)
