#!/usr/bin/env python3
"""Compare paired Tiny fine-tuning runs on identical test predictions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from prepare_chexpert_splits import sha256


def flatten(row, bins, macro=False):
    values = {key: row[key] for key in ("auroc", "brier", "nll")}
    for n in bins:
        stats = row["calibration"][str(n)]
        values["ece_" + str(n)] = stats["ece"] if macro else stats["uniform"]["ece"]
        values["adaptive_ece_" + str(n)] = stats["adaptive_ece"] if macro else stats["adaptive"]["ece"]
    return values


def compare_runs(baseline_dir, candidate_dir, output_dir):
    folders = [Path(baseline_dir), Path(candidate_dir)]
    reports = [json.loads((folder / "calibration/metrics.json").read_text()) for folder in folders]
    evaluations = [json.loads((folder / "evaluation.json").read_text()) for folder in folders]
    left, right = reports
    if left["class_order"] != right["class_order"] or left["bin_counts"] != right["bin_counts"]:
        raise ValueError("Class order/bin settings differ")
    if left["n_samples"] != right["n_samples"] or left["inputs"]["y_gt.npy"]["sha256"] != right["inputs"]["y_gt.npy"]["sha256"]:
        raise ValueError("Comparisons must use identical ground-truth arrays")
    for folder, report in zip(folders, reports):
        for name in ("y_gt.npy", "y_pred.npy"):
            if sha256(folder / name) != report["inputs"][name]["sha256"]:
                raise ValueError("Prediction files changed after metrics were computed")
    if sha256(folders[0] / "samples.json") != sha256(folders[1] / "samples.json"):
        raise ValueError("Test image/patient ordering differs")
    configs = [record["training_config"] for record in evaluations]
    shared = [{key: value for key, value in config.items() if key not in ("loss", "smoothing")} for config in configs]
    if shared[0] != shared[1] or any(record["split"] != "test" for record in evaluations):
        raise ValueError("Training conditions differ beyond loss, or evaluation is not on test")
    if configs[0]["loss"] != "bce" or configs[1]["loss"] != "label_smoothing":
        raise ValueError("Expected BCE baseline and label-smoothing candidate")
    bins = left["bin_counts"]
    rows, deltas = [], []
    for column, name in enumerate(left["class_order"] + ["macro"]):
        values = []
        for method, report in zip(("bce", "label_smoothing"), reports):
            row = report["macro"] if name == "macro" else report["per_class"][column]
            metrics = flatten(row, bins, macro=name == "macro")
            values.append(metrics)
            rows.append(dict({"method": method, "class": name}, **metrics))
        delta = {key: values[1][key] - values[0][key] if values[0][key] is not None and values[1][key] is not None else None
                 for key in values[0]}
        deltas.append(dict({"class": name}, **delta))
    samples = json.loads((folders[0] / "samples.json").read_text())
    result = {"n_samples": left["n_samples"], "n_patients": len({s["patient_id"] for s in samples}),
              "class_order": left["class_order"], "bin_counts": bins, "rows": rows,
              "candidate_minus_baseline": deltas, "training_config_shared": shared[0],
              "selected_epochs": {"bce": evaluations[0]["checkpoint_epoch"],
                                  "label_smoothing": evaluations[1]["checkpoint_epoch"]},
              "test_used_for_checkpoint_selection": False,
              "uncertainty": "Point estimates only; no bootstrap confidence intervals or significance claims."}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, records in [("comparison.csv", rows), ("deltas.csv", deltas)]:
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    macro = [row for row in rows if row["class"] == "macro"]
    def value(number):
        return "N/A" if number is None else "{:.6f}".format(number)
    lines = ["# Tiny fine-tuning 비교", "", "BCE vs label smoothing (epsilon={})".format(configs[1]["smoothing"]),
             "", "동일한 test 이미지 {}장, 환자 {}명. 체크포인트는 별도 dev AUROC로 선택했습니다.".format(
                 result["n_samples"], result["n_patients"]), "",
             "| 지표 | BCE | Label smoothing | 차이 (후보 − baseline) | 좋은 방향 |",
             "|---|---:|---:|---:|---|"]
    for key in flatten(left["macro"], bins, True):
        lines.append("| {} | {} | {} | {} | {} |".format(
            key, value(macro[0][key]), value(macro[1][key]), value(deltas[-1][key]), "높음" if key == "auroc" else "낮음"))
    lines += ["", "## 질환별 결과", "", "| 질환 | 조건 | AUROC | Brier | NLL | ECE@10 | Adaptive ECE@10 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["class"] != "macro":
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                row["class"], row["method"], value(row["auroc"]), value(row["brier"]), value(row["nll"]),
                value(row.get("ece_10")), value(row.get("adaptive_ece_10"))))
    lines += ["", "## 해석 범위", "", "- 차이는 point estimate이며 통계적 유의성을 의미하지 않습니다.",
              "- AUROC와 calibration 지표를 함께 확인하세요. 개선은 사전에 보장하지 않습니다.",
              "- 학습 중 공식 test와 보정용 split은 사용하지 않았습니다. TS는 적용하지 않았습니다.",
              "- 공개 fine-tuned Tiny와 분할·학습 조건이 다르므로, 새 BCE가 loss 비교의 기준입니다.",
              "", "![질환별 비교](comparison.png)", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    plot_comparison(result, output)
    print("\n".join(lines[:16]), flush=True)
    print("Comparison report: " + str(output / "report.md"), flush=True)
    return result


def plot_comparison(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = result["class_order"]
    metrics = ["auroc", "brier", "nll", "ece_10", "adaptive_ece_10"]
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, metric in zip(axes.flat, metrics):
        for offset, method in [(-0.18, "bce"), (0.18, "label_smoothing")]:
            by_class = {row["class"]: row for row in result["rows"] if row["method"] == method}
            values = [by_class[name].get(metric) for name in names]
            ax.bar(np.arange(5) + offset, [v if v is not None else np.nan for v in values], width=.36, label=method)
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels([name.replace(" ", "\n") for name in names], rotation=20, ha="right")
        ax.set_title(metric + (" (higher is better)" if metric == "auroc" else " (lower is better)"))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    axes.flat[0].legend()
    axes.flat[-1].axis("off")
    figure.suptitle("Tiny: BCE vs label smoothing — same {} test images".format(result["n_samples"]))
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(str(output / "comparison.png"), dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compare_runs(args.baseline_dir, args.candidate_dir, args.output_dir)
