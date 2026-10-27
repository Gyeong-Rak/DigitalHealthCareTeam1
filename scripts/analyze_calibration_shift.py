#!/usr/bin/env python3
"""Explain and verify the calibration study from saved predictions (CPU only, no training).

1. Label prevalence per split: report-labeled train/calibration vs expert-labeled official valid.
2. Post-hoc calibrators beyond per-class TS, fitted on the same data as each TS protocol:
   intercept-only shift, Platt (slope + intercept), label-free prior-shift correction.
3. Seed replication: mean/SD over seeds and patient-bootstrap CIs of the seed-averaged
   difference to the same-seed baseline, for every calibrator.

Inputs are the study folders written by run_calibration_study.py / run_seed_replication.py.
Outputs contain aggregated metrics only (no image paths, patient IDs or predictions).
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit

from calibration_postprocess import binary_nll, fit_temperatures, load_export
from evaluate_calibration import evaluate_predictions
from prepare_chexpert_splits import CLASS_NAMES, sha256
from study_io import atomic_json, read_json

CALIBRATORS = ("raw", "ts", "bias", "platt", "prior")
METRICS = ("auroc", "brier", "nll", "ece_10", "adaptive_ece_10")
PROTOCOLS = ("report_holdout", "expert_subset")
LOWER_IS_BETTER = {"auroc": False, "brier": True, "nll": True, "ece_10": True, "adaptive_ece_10": True}


# ---------------------------------------------------------------- calibrators

def fit_bias(z, y):
    """Per-class intercept on fixed-slope logits: sigmoid(z + b)."""
    identity = float(binary_nll(z, y).mean())
    fit = minimize_scalar(lambda b: float(binary_nll(z + b, y).mean()), bounds=(-15.0, 15.0), method="bounded",
                          options={"xatol": 1e-8, "maxiter": 500})
    return (1.0, float(fit.x)) if fit.success and fit.fun <= identity else (1.0, 0.0)


def fit_platt(z, y):
    """Per-class slope and intercept: sigmoid(a z + b), a > 0 kept by log-parametrisation."""
    identity = float(binary_nll(z, y).mean())
    def cost(params):
        a, b = np.exp(params[0]), params[1]
        u = a * z + b
        p = expit(u)
        grad_u = (p - y) / len(y)
        return float(binary_nll(u, y).mean()), np.array([float(grad_u @ z) * a, float(grad_u.sum())])
    fit = minimize(cost, np.zeros(2), jac=True, method="L-BFGS-B",
                   bounds=[(np.log(0.01), np.log(100.0)), (-15.0, 15.0)], options={"maxiter": 500})
    if not np.isfinite(fit.fun):
        raise ValueError("Platt fitting did not converge")
    return (float(np.exp(fit.x[0])), float(fit.x[1])) if fit.fun <= identity else (1.0, 0.0)


def observed_prevalence(labels, mask):
    labels, mask = np.asarray(labels, dtype=np.float64), np.asarray(mask, dtype=bool)
    return np.array([labels[mask[:, c], c].mean() if mask[:, c].any() else np.nan for c in range(labels.shape[1])])


def fit_calibrators(z, y, mask, source_prevalence):
    """Return {name: (slope[5], intercept[5])} applied as sigmoid(slope * logits + intercept)."""
    z, y, mask = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(mask, dtype=bool)
    k = z.shape[1]
    temperatures, _ = fit_temperatures(z, y, mask, per_class=True)
    params = {"raw": (np.ones(k), np.zeros(k)), "ts": (1.0 / temperatures, np.zeros(k))}
    for name, fitter in (("bias", fit_bias), ("platt", fit_platt)):
        slope, intercept = np.ones(k), np.zeros(k)
        for c in range(k):
            zc, yc = z[mask[:, c], c], y[mask[:, c], c]
            if len(yc) and np.unique(yc).size == 2:
                slope[c], intercept[c] = fitter(zc, yc)
        params[name] = (slope, intercept)
    # Label shift (Saerens et al. 2002): keep the classifier, move only the base rate
    # from the report-labeled source split to the fitting split. No fitted parameters.
    target = observed_prevalence(y, mask)
    shift = logit(np.clip(target, 1e-4, 1 - 1e-4)) - logit(np.clip(source_prevalence, 1e-4, 1 - 1e-4))
    params["prior"] = (np.ones(k), np.where(np.isfinite(shift), shift, 0.0))
    return params


def apply(params, logits):
    slope, intercept = params
    return expit(np.asarray(logits, dtype=np.float64) * slope + intercept)


# ---------------------------------------------------------------- loading

def protocol_split(test, expert_fraction, split_seed):
    """Identical to calibration_postprocess.postprocess (verified against saved test_indices)."""
    patients = sorted({s["patient_id"] for s in test["samples"]})
    rng = np.random.RandomState(split_seed)
    rng.shuffle(patients)
    fitting = set(patients[:max(1, int(len(patients) * expert_fraction))])
    fit = np.array([i for i, s in enumerate(test["samples"]) if s["patient_id"] in fitting])
    held = np.array([i for i, s in enumerate(test["samples"]) if s["patient_id"] not in fitting])
    return fit, held


def discover_runs(study_dirs):
    runs, seen = [], {}
    for study in study_dirs:
        study = Path(study).expanduser().resolve()
        spec = read_json(study / "study.json")
        config = spec["config"]
        for summary_path in sorted(study.glob("runs/*/seed*/processed/summary.json")):
            folder = summary_path.parent.parent
            case, seed = folder.parent.name, int(folder.name.removeprefix("seed"))
            summary = read_json(summary_path)
            for kind in ("calibration", "test"):
                for name, digest in summary["inputs"][kind].items():
                    if sha256(folder / kind / name) != digest:
                        raise ValueError("Predictions changed after postprocessing: " + str(folder / kind / name))
            digest = summary["inputs"]["test"]["logits.npy"]
            if (case, seed) in seen:
                if seen[(case, seed)] == digest:
                    continue  # Same run imported into another study folder.
                case = case + "@" + study.name
            seen[(case, seed)] = digest
            runs.append(dict(case=case, seed=seed, folder=folder, summary=summary,
                             expert_fraction=config["expert_calibration_fraction"], split_seed=config["split_seed"],
                             splits=study / "splits"))
    if not runs:
        raise ValueError("No postprocessed runs found")
    return runs


def train_prevalence(splits_dir):
    """Report labels as used for calibration exports: 1 positive, blank/0 negative, -1 excluded."""
    counts = {name: [0, 0, 0] for name in CLASS_NAMES}  # positive, negative, uncertain
    with (Path(splits_dir) / "train.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            for name in CLASS_NAMES:
                value = row[name].strip()
                counts[name][2 if value and float(value) == -1 else 0 if value and float(value) == 1 else 1] += 1
    return {name: dict(prevalence=p / (p + n), uncertain_rate=u / (p + n + u)) for name, (p, n, u) in counts.items()}


# ---------------------------------------------------------------- evaluation

def macro_metrics(labels, probabilities, bins=(5, 10)):
    report = evaluate_predictions(labels, probabilities, bins)
    macro = report["macro"]
    values = dict(auroc=macro["auroc"], brier=macro["brier"], nll=macro["nll"],
                  ece_10=macro["calibration"]["10"]["ece"], adaptive_ece_10=macro["calibration"]["10"]["adaptive_ece"])
    if "5" in macro["calibration"]:
        values["ece_5"] = macro["calibration"]["5"]["ece"]
    per_class = {row["class"]: dict(auroc=row["auroc"], ece_10=row["calibration"]["10"]["uniform"]["ece"],
                                    mean_minus_prevalence=row["mean_probability_minus_prevalence"])
                 for row in report["per_class"]}
    return values, per_class


def evaluate_run(run):
    cal, test = load_export(run["folder"] / "calibration"), load_export(run["folder"] / "test")
    source_prevalence = observed_prevalence(cal["y_gt"], cal["label_mask"])
    fit_idx, held_idx = protocol_split(test, run["expert_fraction"], run["split_seed"])
    data = {"report_holdout": ((cal["logits"], cal["y_gt"], cal["label_mask"]), np.arange(len(test["samples"]))),
            "expert_subset": ((test["logits"][fit_idx], test["y_gt"][fit_idx], test["label_mask"][fit_idx]), held_idx)}
    result, eval_sets = dict(case=run["case"], seed=run["seed"], protocols={}), {}
    for protocol, ((z, y, mask), indices) in data.items():
        saved = run["folder"] / "processed" / protocol / "test_indices.npy"
        if not np.array_equal(np.load(saved), indices):
            raise ValueError("Protocol split differs from postprocessing: " + str(saved))
        params = fit_calibrators(z, y, mask, source_prevalence)
        labels = test["y_gt"][indices].astype(np.float64)
        record = dict(fit_prevalence=observed_prevalence(y, mask).tolist(), calibrators={})
        eval_sets[protocol] = dict(labels=labels, patients=[test["samples"][i]["patient_id"] for i in indices],
                                   probabilities={})
        for name in CALIBRATORS:
            if protocol == "report_holdout" and name == "prior":
                continue  # Fitting data is the source split itself: the shift is zero by construction.
            probabilities = apply(params[name], test["logits"][indices])
            macro, per_class = macro_metrics(labels, probabilities)
            record["calibrators"][name] = dict(slope=params[name][0].tolist(), intercept=params[name][1].tolist(),
                                               macro=macro, per_class=per_class)
            eval_sets[protocol]["probabilities"][name] = probabilities
        # Reproduction check against the teammate's per-class TS numbers.
        saved_ts = next(r for r in run["summary"]["protocols"][protocol]["rows"]
                        if r["calibration"] == "ts_per_class" and r["disease"] == "macro")
        mine = record["calibrators"]["ts"]["macro"]
        for key in ("auroc", "brier", "nll", "ece_10"):
            if saved_ts[key] is not None and abs(saved_ts[key] - mine[key]) > 1e-6:
                raise ValueError("TS reproduction mismatch for {} {} {}: {} vs {}".format(
                    run["case"], protocol, key, saved_ts[key], mine[key]))
        record["ts_reproduced"] = True
        result["protocols"][protocol] = record
    return result, source_prevalence.tolist(), eval_sets


def bootstrap_draws(job):
    """Metrics for each patient-bootstrap replicate; identical replicate indices for every array."""
    labels, probabilities, groups, replicates, seed = job
    rng = np.random.RandomState(seed)
    draws = np.full((replicates, len(METRICS)), np.nan)
    for r in range(replicates):
        indices = np.concatenate([groups[i] for i in rng.randint(len(groups), size=len(groups))])
        y = labels[indices]
        if (y.min(axis=0) == y.max(axis=0)).any():
            continue  # A disease without both classes has no macro AUROC; keep the 5-disease estimand.
        values, _ = macro_metrics(y, probabilities[indices], bins=(10,))
        draws[r] = [values[m] for m in METRICS]
    return draws


def interval(values):
    values = values[np.isfinite(values)]
    return None if not len(values) else dict(low=float(np.quantile(values, .025)), high=float(np.quantile(values, .975)),
                                             valid_replicates=int(len(values)))


# ---------------------------------------------------------------- report

def fmt(value, digits=4):
    return "—" if value is None or (isinstance(value, float) and not np.isfinite(value)) else "{:.{}f}".format(value, digits)


def ci_text(point, ci, scale=1.0, digits=4):
    if ci is None:
        return fmt(point * scale, digits)
    return "{} [{}, {}]".format(fmt(point * scale, digits), fmt(ci["low"] * scale, digits), fmt(ci["high"] * scale, digits))


def build_markdown(analysis):
    lines = ["# 보정 결과 설명·검증 분석", "",
             "학습 없이 저장된 예측(logits)으로 계산했습니다. 모든 보정은 기존 TS와 같은 fitting 데이터·평가 데이터를 사용합니다.",
             "괄호는 환자 단위 paired bootstrap 95% CI({}회)입니다.".format(analysis["replicates"]), ""]
    lines += ["## 1. 질환별 유병률 (라벨 출처별)", "",
              "| 질환 | train (판독문) | 불확실 비율 | calibration (판독문) | 공식 valid (전문가) |", "|---|---:|---:|---:|---:|"]
    for c, name in enumerate(CLASS_NAMES):
        t = analysis["prevalence"]["train"][name]
        lines.append("| {} | {} | {} | {} | {} |".format(name, fmt(t["prevalence"], 3), fmt(t["uncertain_rate"], 3),
                     fmt(analysis["prevalence"]["calibration"][c], 3), fmt(analysis["prevalence"]["test"][c], 3)))
    lines += ["", "판독문 라벨은 -1(불확실)을 제외하고 미언급을 0으로 셉니다(calibration export와 동일).", ""]

    lines += ["## 2. 보정 방법 비교 (전체 조건·seed 평균, 같은 run의 TS 대비 차이)", "",
              "| 방법 | 설명 |", "|---|---|",
              "| ts | 기존 질환별 Temperature Scaling (기울기만) |",
              "| bias | 질환별 절편만 이동 (기울기 1 고정) |",
              "| platt | 질환별 기울기 + 절편 |",
              "| prior | 학습 없이 유병률 차이만 보정: logit + logit(π_fit) − logit(π_report) (expert_subset만 해당) |", ""]
    for protocol in PROTOCOLS:
        block = analysis["calibrator_vs_ts"].get(protocol)
        if not block:
            continue
        lines += ["### {} (run {}개)".format(protocol, block["n_runs"]), "",
                  "| 방법 | ΔAUROC | ΔBrier | ΔNLL | ΔECE@10 | ΔAdaptive ECE@10 |", "|---|---:|---:|---:|---:|---:|"]
        for name, values in block["methods"].items():
            lines.append("| {} | {} |".format(name, " | ".join(ci_text(values[m]["mean"], values[m]["ci"]) for m in METRICS)))
        lines.append("")

    lines += ["## 3. 보정 방법별 조건 성능 (seed 0)", ""]
    for protocol in PROTOCOLS:
        lines += ["### {}".format(protocol), "", "| 조건 | " + " | ".join("{} ECE@10".format(c) for c in ("ts", "bias", "platt", "prior")) +
                  " | mAUROC |", "|---|" + "---:|" * 5]
        for row in analysis["runs"]:
            if row["seed"] != 0 or protocol not in row["protocols"]:
                continue
            cal = row["protocols"][protocol]["calibrators"]
            lines.append("| {} | {} | {} |".format(row["case"], " | ".join(
                fmt(cal[c]["macro"]["ece_10"]) if c in cal else "—" for c in ("ts", "bias", "platt", "prior")),
                fmt(cal["ts"]["macro"]["auroc"])))
        lines.append("")

    lines += ["## 4. Baseline의 질환별 평균 예측 − 실제 유병률 (seed 0)", "",
              "양수면 위험을 과대, 음수면 과소 예측합니다. TS는 이 값을 직접 옮기지 못하고 절편 보정은 옮깁니다.", ""]
    base = next((r for r in analysis["runs"] if r["case"] == "baseline" and r["seed"] == 0), None)
    if base:
        for protocol in PROTOCOLS:
            cal = base["protocols"][protocol]["calibrators"]
            names = [c for c in CALIBRATORS if c in cal]
            lines += ["### {}".format(protocol), "", "| 질환 | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
            for name in CLASS_NAMES:
                lines.append("| {} | {} |".format(name, " | ".join(fmt(cal[c]["per_class"][name]["mean_minus_prevalence"], 3) for c in names)))
            lines.append("")

    if analysis["seed_replication"]:
        lines += ["## 5. Seed 반복: 같은 seed baseline 대비 차이 (seed 평균)", "",
                  "점추정은 seed별 차이의 평균, 괄호는 평가 환자 bootstrap CI(seed는 고정)입니다. SD는 seed 간 표준편차입니다.", ""]
        for protocol in PROTOCOLS:
            for calibrator in ("ts", "platt"):
                rows = [r for r in analysis["seed_replication"] if r["protocol"] == protocol and r["calibrator"] == calibrator]
                if not rows:
                    continue
                lines += ["### {} · {}".format(protocol, calibrator), "",
                          "| 조건 | seeds | ΔAUROC (%p) | ΔECE@10 | ΔBrier | ΔNLL | seed별 ΔAUROC (%p) | seed별 ΔECE@10 |",
                          "|---|---|---:|---:|---:|---:|---|---|"]
                for r in rows:
                    d = r["delta"]
                    lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        r["case"], ",".join(map(str, r["seeds"])),
                        ci_text(d["auroc"]["mean"], d["auroc"]["ci"], 100, 2), ci_text(d["ece_10"]["mean"], d["ece_10"]["ci"]),
                        ci_text(d["brier"]["mean"], d["brier"]["ci"]), ci_text(d["nll"]["mean"], d["nll"]["ci"]),
                        ", ".join(fmt(v * 100, 2) for v in d["auroc"]["per_seed"]),
                        ", ".join(fmt(v) for v in d["ece_10"]["per_seed"])))
                lines.append("")
    lines += ["## 해석 시 주의", "",
              "- expert_subset 평가는 약 100장·100명입니다. 작은 차이는 CI로 판단합니다.",
              "- prior 보정은 fitting 데이터의 유병률만 사용하며, 전문가 라벨 집합의 유병률이 평가 절반과 같다는 가정에 기댑니다.",
              "- 수치는 서버 원본 예측에서 다시 계산했고, 기존 질환별 TS 수치와 일치함을 확인했습니다(ts_reproduced)."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main

def analyze(study_dirs, output, replicates=1000, bootstrap_seed=2026, workers=4, focus=("a_mixup", "c_lora")):
    runs = discover_runs(study_dirs)
    evaluated, eval_sets, sources = [], {}, []
    for run in runs:
        result, source_prevalence, sets = evaluate_run(run)
        evaluated.append(result)
        eval_sets[(run["case"], run["seed"])] = sets
        sources.append(source_prevalence)
        print("evaluated {} seed{}".format(run["case"], run["seed"]), flush=True)
    any_run = runs[0]
    test = load_export(any_run["folder"] / "test")
    prevalence = dict(train=train_prevalence(any_run["splits"]), calibration=np.mean(sources, axis=0).tolist(),
                      test=test["y_gt"].mean(axis=0).tolist())

    # Bootstrap: one draw matrix per (run, protocol, calibrator), shared replicate indices per protocol.
    jobs, keys = [], []
    for key, sets in eval_sets.items():
        for protocol, data in sets.items():
            patients = sorted(set(data["patients"]))
            groups = [np.flatnonzero(np.asarray(data["patients"]) == p) for p in patients]
            seed = bootstrap_seed + PROTOCOLS.index(protocol)
            for name, probabilities in data["probabilities"].items():
                jobs.append((data["labels"], probabilities, groups, replicates, seed))
                keys.append((key, protocol, name))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        draws = dict(zip(keys, pool.map(bootstrap_draws, jobs, chunksize=4)))
    point = {(r["case"], r["seed"], protocol, name): np.array([values["macro"][m] for m in METRICS], dtype=float)
             for r in evaluated for protocol, record in r["protocols"].items()
             for name, values in record["calibrators"].items()}

    calibrator_vs_ts = {}
    for protocol in PROTOCOLS:
        run_keys = sorted({(k[0][0], k[0][1]) for k in draws if k[1] == protocol})
        methods = {}
        for name in CALIBRATORS:
            if name == "ts" or not all((rk, protocol, name) in draws for rk in run_keys):
                continue
            deltas = np.stack([draws[(rk, protocol, name)] - draws[(rk, protocol, "ts")] for rk in run_keys]).mean(axis=0)
            points = np.mean([point[(*rk, protocol, name)] - point[(*rk, protocol, "ts")] for rk in run_keys], axis=0)
            methods[name] = {m: dict(mean=float(points[j]), ci=interval(deltas[:, j])) for j, m in enumerate(METRICS)}
        calibrator_vs_ts[protocol] = dict(n_runs=len(run_keys), methods=methods)

    seed_replication = []
    seeds_by_case = {}
    for r in evaluated:
        seeds_by_case.setdefault(r["case"], set()).add(r["seed"])
    for case in [c for c in seeds_by_case if c != "baseline" and (c in focus or len(seeds_by_case[c]) > 1)]:
        seeds = sorted(seeds_by_case[case] & seeds_by_case.get("baseline", set()))
        if len(seeds) < 2:
            continue
        for protocol in PROTOCOLS:
            for name in ("ts", "platt", "bias", "prior", "raw"):
                if not all(((case, s), protocol, name) in draws for s in seeds):
                    continue
                per_seed = np.array([point[(case, s, protocol, name)] - point[("baseline", s, protocol, name)] for s in seeds])
                boot = np.stack([draws[((case, s), protocol, name)] - draws[(("baseline", s), protocol, name)] for s in seeds]).mean(axis=0)
                seed_replication.append(dict(case=case, protocol=protocol, calibrator=name, seeds=seeds, delta={
                    m: dict(mean=float(per_seed[:, j].mean()), sd=float(per_seed[:, j].std(ddof=1)),
                            per_seed=per_seed[:, j].tolist(), ci=interval(boot[:, j])) for j, m in enumerate(METRICS)}))

    analysis = dict(schema_version=1, replicates=replicates, bootstrap_seed=bootstrap_seed,
                    studies=[str(Path(s).expanduser().resolve()) for s in study_dirs], class_order=list(CLASS_NAMES),
                    prevalence=prevalence, runs=evaluated, calibrator_vs_ts=calibrator_vs_ts,
                    seed_replication=seed_replication)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "analysis.json", analysis)
    (output / "report.md").write_text(build_markdown(analysis), encoding="utf-8")
    print("ANALYSIS COMPLETE: " + str(output / "report.md"), flush=True)
    return analysis


def main():
    base = Path.home() / "workspace/outputs/chexpert"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-dir", type=Path, action="append",
                        help="Study output folder; repeat for seed replication/follow-up folders")
    parser.add_argument("--output-dir", type=Path, default=base / "calibration_shift_analysis")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    studies = args.study_dir or [base / "tiny_study_seed0"]
    analyze(studies, args.output_dir, args.replicates, args.bootstrap_seed, args.workers)


if __name__ == "__main__":
    main()
