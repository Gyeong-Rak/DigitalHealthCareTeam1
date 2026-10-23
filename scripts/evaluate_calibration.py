#!/usr/bin/env python3
"""Evaluate saved CheXpert sigmoid probabilities without running the model."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


# EVA-X classification/utils/dataloader_med.py: CheXpert(train_cols=...).
CLASS_NAMES = (
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"
)


def validate_arrays(y_true, y_prob):
    """Reject uncertain/soft labels and logits instead of interpreting them as truth."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if y_true.ndim != 2 or y_true.shape != y_prob.shape:
        raise ValueError("y_gt and y_pred must have the same (N, 5) shape")
    if y_true.shape[0] == 0 or y_true.shape[1] != len(CLASS_NAMES):
        raise ValueError("Expected nonempty (N, 5) arrays in EVA-X CheXpert class order")
    for name, array in (("y_gt", y_true), ("y_pred", y_prob)):
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError("{} must contain finite numeric values".format(name))
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_gt must contain only binary 0/1 evaluation labels, not -1 or soft labels")
    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("y_pred must contain sigmoid probabilities in [0, 1], not logits")
    return y_true.astype(np.float64), y_prob.astype(np.float64)


def calibration_bins(labels, probabilities, n_bins, strategy):
    """Positive-class probability calibration; identical probabilities stay together.

    Intervals are (lower, upper], except the first interval also includes 0. Adaptive
    bins use quantile edges; duplicate edges are removed rather than splitting ties.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "adaptive":
        edges = np.quantile(probabilities, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    else:
        raise ValueError("Unknown binning strategy: " + strategy)
    assignments = np.searchsorted(edges[1:-1], probabilities, side="left")
    bins = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = assignments == index
        count = int(selected.sum())
        mean_probability = float(probabilities[selected].mean()) if count else None
        positive_fraction = float(labels[selected].mean()) if count else None
        gap = mean_probability - positive_fraction if count else None
        bins.append({
            "index": index, "lower": float(lower), "upper": float(upper),
            "count": count, "mean_probability": mean_probability,
            "positive_fraction": positive_fraction,
            "probability_minus_positive_fraction": gap,
            "ece_contribution": count / len(labels) * abs(gap) if count else 0.0,
        })
    return {
        "ece": float(sum(b["ece_contribution"] for b in bins)),
        "occupied_bins": sum(b["count"] > 0 for b in bins),
        "bins": bins,
    }


def evaluate_predictions(y_true, y_prob, bin_counts=(5, 10), eps=1e-7):
    y_true, y_prob = validate_arrays(y_true, y_prob)
    bin_counts = sorted(set(bin_counts))
    if not bin_counts or any(not isinstance(n, (int, np.integer)) or n < 1 for n in bin_counts):
        raise ValueError("bin counts must be positive integers")
    if not 0 < eps < 0.5:
        raise ValueError("eps must be between 0 and 0.5")
    per_class = []
    for column, name in enumerate(CLASS_NAMES):
        labels, probabilities = y_true[:, column], y_prob[:, column]
        clipped = np.clip(probabilities, eps, 1.0 - eps)
        auc = float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else None
        row = {
            "class": name, "n_samples": len(labels), "n_positive": int(labels.sum()),
            "prevalence": float(labels.mean()), "mean_probability": float(probabilities.mean()),
            "mean_probability_minus_prevalence": float((probabilities - labels).mean()),
            "auroc": auc, "brier": float(np.mean((probabilities - labels) ** 2)),
            "nll": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
            "calibration": {},
        }
        for n_bins in bin_counts:
            row["calibration"][str(n_bins)] = {
                strategy: calibration_bins(labels, probabilities, n_bins, strategy)
                for strategy in ("uniform", "adaptive")
            }
        per_class.append(row)
    auc_values = [row["auroc"] for row in per_class if row["auroc"] is not None]
    macro = {
        "auroc": float(np.mean(auc_values)) if auc_values else None,
        "auroc_valid_classes": len(auc_values),
        "brier": float(np.mean([row["brier"] for row in per_class])),
        "nll": float(np.mean([row["nll"] for row in per_class])),
        "calibration": {
            str(n): {
                "ece": float(np.mean([r["calibration"][str(n)]["uniform"]["ece"] for r in per_class])),
                "adaptive_ece": float(np.mean([r["calibration"][str(n)]["adaptive"]["ece"] for r in per_class])),
            } for n in bin_counts
        },
    }
    return {
        "schema_version": 1, "n_samples": y_true.shape[0], "class_order": list(CLASS_NAMES),
        "bin_counts": bin_counts, "nll_clip_eps": eps, "evaluation_unit": "image",
        "probability_type": "per-disease positive-class sigmoid probability",
        "adaptive_binning": "quantile edges, duplicate edges merged; ties are not split",
        "nll_source": "saved probabilities, float64; clipped only for NLL; natural logarithm",
        "per_class": per_class, "macro": macro,
    }


def write_tables(report, output_dir):
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    fields = ["class", "n_samples", "n_positive", "prevalence", "mean_probability",
              "mean_probability_minus_prevalence", "auroc", "brier", "nll", "n_bins", "ece", "adaptive_ece"]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["per_class"]:
            for n in report["bin_counts"]:
                values = {key: row[key] for key in fields if key in row}
                calibration = row["calibration"][str(n)]
                values.update(n_bins=n, ece=calibration["uniform"]["ece"],
                              adaptive_ece=calibration["adaptive"]["ece"])
                writer.writerow(values)
        for n in report["bin_counts"]:
            writer.writerow(dict({"class": "macro", "n_samples": report["n_samples"], "n_bins": n,
                                  "auroc": report["macro"]["auroc"], "brier": report["macro"]["brier"],
                                  "nll": report["macro"]["nll"]}, **report["macro"]["calibration"][str(n)]))
    fields = ["class", "strategy", "requested_bins", "index", "lower", "upper", "count",
              "mean_probability", "positive_fraction", "probability_minus_positive_fraction", "ece_contribution"]
    with (output_dir / "bin_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["per_class"]:
            for n in report["bin_counts"]:
                for strategy in ("uniform", "adaptive"):
                    for bin_row in row["calibration"][str(n)][strategy]["bins"]:
                        writer.writerow(dict(bin_row, **{"class": row["class"], "strategy": strategy, "requested_bins": n}))


def plot_reliability(report, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for n in report["bin_counts"]:
        for strategy in ("uniform", "adaptive"):
            figure, axes = plt.subplots(2, len(CLASS_NAMES), figsize=(20, 7),
                                        gridspec_kw={"height_ratios": [3, 1]})
            for column, row in enumerate(report["per_class"]):
                stats = row["calibration"][str(n)][strategy]
                occupied = [b for b in stats["bins"] if b["count"]]
                x = [b["mean_probability"] for b in occupied]
                y = [b["positive_fraction"] for b in occupied]
                ax, histogram = axes[0, column], axes[1, column]
                ax.plot([0, 1], [0, 1], "--", color="0.5", linewidth=1)
                ax.plot(x, y, color="tab:blue", linewidth=1)
                ax.scatter(x, y, s=[20 + 120 * b["count"] / report["n_samples"] for b in occupied], zorder=3)
                ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability",
                       ylabel="Observed positive fraction", title=row["class"])
                ax.text(0.04, 0.95, "ECE={:.4f}\nN={}; occupied bins={}".format(
                    stats["ece"], row["n_samples"], stats["occupied_bins"]),
                    transform=ax.transAxes, va="top", fontsize=9)
                ax.grid(alpha=0.2)
                histogram.bar([b["lower"] for b in stats["bins"]], [b["count"] for b in stats["bins"]],
                              width=[b["upper"] - b["lower"] for b in stats["bins"]],
                              align="edge", edgecolor="white", color="tab:blue", alpha=0.7)
                histogram.set(xlim=(0, 1), xlabel="Predicted probability", ylabel="Images / bin")
            figure.suptitle("CheXpert reliability: {} bins (requested {}), N={} images".format(
                strategy, n, report["n_samples"]))
            figure.tight_layout(rect=(0, 0, 1, 0.94))
            figure.savefig(str(output_dir / "reliability_{}_{}.png".format(strategy, n)), dpi=160)
            plt.close(figure)


def print_summary(report):
    def number(value):
        return "N/A" if value is None else "{:.6f}".format(value)

    header = "{:<20} {:>10} {:>10} {:>10}".format("Class", "AUROC", "Brier", "NLL")
    for n in report["bin_counts"]:
        header += " {:>10} {:>10}".format("ECE@" + str(n), "AECE@" + str(n))
    print(header)
    for row in report["per_class"] + [dict(report["macro"], **{"class": "macro"})]:
        line = "{:<20} {:>10} {:>10} {:>10}".format(row["class"], number(row["auroc"]), number(row["brier"]), number(row["nll"]))
        for n in report["bin_counts"]:
            stats = row["calibration"][str(n)]
            ece = stats["ece"] if row["class"] == "macro" else stats["uniform"]["ece"]
            aece = stats["adaptive_ece"] if row["class"] == "macro" else stats["adaptive"]["ece"]
            line += " {:>10} {:>10}".format(number(ece), number(aece))
        print(line)
    if report["macro"]["auroc_valid_classes"] != len(CLASS_NAMES):
        print("AUROC is N/A for single-label classes; macro AUROC averages {} valid classes.".format(
            report["macro"]["auroc_valid_classes"]))


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True, help="Directory containing y_gt.npy and y_pred.npy")
    parser.add_argument("--output-dir", type=Path, help="Default: <eval-dir>/calibration")
    parser.add_argument("--bins", type=int, nargs="+", default=[5, 10], help="Bin counts to report (default: 5 10)")
    parser.add_argument("--eps", type=float, default=1e-7, help="NLL probability clipping epsilon (default: 1e-7)")
    parser.add_argument("--no-plots", action="store_true", help="Write metrics/tables without PNGs")
    args = parser.parse_args(argv)
    eval_dir = args.eval_dir.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir else eval_dir / "calibration"
    try:
        paths = [eval_dir / "y_gt.npy", eval_dir / "y_pred.npy"]
        y_true, y_prob = [np.load(str(path), allow_pickle=False) for path in paths]
        mask_path = eval_dir / "label_mask.npy"
        if mask_path.exists():
            mask = np.load(str(mask_path), allow_pickle=False)
            if mask.shape != y_true.shape or not np.isin(mask, [0, 1]).all() or not mask.all():
                raise ValueError("This split has unobserved labels; use mask-aware metrics, not this complete-label CLI")
        report = evaluate_predictions(y_true, y_prob, args.bins, args.eps)
        report["inputs"] = {path.name: {"sha256": file_sha256(path), "dtype": str(array.dtype)}
                            for path, array in zip(paths, (y_true, y_prob))}
        output_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_plots:
            plot_reliability(report, output_dir)
        write_tables(report, output_dir)
    except (OSError, ValueError, EOFError) as error:
        parser.error(str(error))
    print_summary(report)
    print("Results: " + str(output_dir))


if __name__ == "__main__":
    main()
