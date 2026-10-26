#!/usr/bin/env python3
"""Aggregate completed study exports, paired patient bootstrap, public-safe Markdown."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from calibration_postprocess import flattened, load_export
from evaluate_calibration import evaluate_predictions
from prepare_chexpert_splits import CLASS_NAMES, sha256
from study_io import atomic_json, read_json, now

GENERATED = "<!-- calibration-study: generated -->\n"
METRICS = ("auroc", "brier", "nll", "ece_5", "adaptive_ece_5", "ece_10", "adaptive_ece_10")


def number(value):
    return "—" if value is None else "{:.6f}".format(value)


def metric_table(rows):
    lines = ["| 평가 | 보정 | 질환 | AUROC | Brier | NLL | ECE@5 | AECE@5 | ECE@10 | AECE@10 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {} | {} | {} | {} |".format(row["protocol"], row["calibration"], row["disease"],
                     " | ".join(number(row[k]) for k in METRICS)))
    return "\n".join(lines)


def render_report(title, setup, metrics, interpretation, limitations, artifacts, template=None):
    sections = dict(title=title, setup=setup, metrics=metrics, interpretation=interpretation,
                    limitations=limitations, artifacts=artifacts)
    if template is None:
        text = "# {{title}}\n\n## 실험 내용\n\n{{setup}}\n\n## 결과\n\n{{metrics}}\n\n## 해석\n\n{{interpretation}}\n\n## 한계\n\n{{limitations}}\n\n## 산출물\n\n{{artifacts}}\n"
    else:
        text = Path(template).read_text(encoding="utf-8")
        keys = re.findall(r"\{\{([a-z_]+)\}\}", text)
        if set(keys) != set(sections):
            raise ValueError("Report template must contain the six documented placeholders")
    for key, value in sections.items():
        text = text.replace("{{" + key + "}}", value)
    return GENERATED + text


def write_managed(path, content):
    path = Path(path)
    if path.exists() and not path.read_text(encoding="utf-8").startswith(GENERATED):
        raise ValueError("Refusing to overwrite a hand-written report: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def collect(output):
    spec = read_json(output / "study.json")
    cases = spec["config"]["cases"] + [{"id": "combination", "axis": "combination"}]
    results = []
    for case in cases:
        marker = output / "completed" / (case["id"] + "_seed0_postprocess.json")
        if not marker.exists():
            continue
        for relative, digest in read_json(marker)["artifacts"].items():
            if sha256(output / relative) != digest:
                raise ValueError("Postprocess artifact changed: " + relative)
        folder = output / "runs" / case["id"] / "seed0"
        summary = read_json(folder / "processed/summary.json")
        for kind in ("calibration", "test"):
            for name, digest in summary["inputs"][kind].items():
                if sha256(folder / kind / name) != digest:
                    raise ValueError("Predictions changed after postprocessing")
        evaluation = read_json(folder / "test/evaluation.json")
        history = [json.loads(line) for line in (folder / "train/history.jsonl").read_text().splitlines()]
        rows = [dict(row, protocol=protocol, case=case["id"], axis=case["axis"], seed=0)
                for protocol, record in summary["protocols"].items() for row in record["rows"]]
        results.append(dict(case=case["id"], axis=case["axis"], seed=0, rows=rows,
                            checkpoint_epoch=evaluation["checkpoint_epoch"], config=evaluation["training_config"],
                            trained_epochs=history[-1]["epoch"], early_stopped=history[-1].get("early_stopped", False),
                            temperatures={p: {m: r["methods"][m]["temperatures"] for m in ("ts_shared", "ts_per_class")}
                                          for p, r in summary["protocols"].items()},
                            protocols={p: {k: r[k] for k in ("n_test_images", "n_test_patients", "calibration_labels")}
                                       for p, r in summary["protocols"].items()}))
    return spec, results


def score_array(labels, probabilities):
    report = evaluate_predictions(labels, probabilities)
    rows = flattened(report)
    result = np.array([[np.nan if r[k] is None else r[k] for k in METRICS] for r in rows])
    if report["macro"]["auroc_valid_classes"] != len(CLASS_NAMES):
        result[-1, 0] = np.nan  # Keep the five-disease estimand in bootstrap draws.
    return result


def intervals(draws):
    result = {}
    for column, disease in enumerate([*CLASS_NAMES, "macro"]):
        result[disease] = {}
        for j, metric in enumerate(METRICS):
            values = draws[:, column, j]
            values = values[np.isfinite(values)]
            result[disease][metric] = dict(low=float(np.quantile(values, .025)) if len(values) else None,
                high=float(np.quantile(values, .975)) if len(values) else None, valid_replicates=len(values))
    return result


def patient_bootstrap(labels, probabilities, patient_ids, baseline=None, raw=None, replicates=500, seed=2026):
    patients = sorted(set(patient_ids))
    if len(patients) < 2:
        raise ValueError("Patient bootstrap needs at least two patients")
    groups = [np.flatnonzero(np.asarray(patient_ids) == patient) for patient in patients]
    rng = np.random.RandomState(seed)
    absolute, paired, ts_delta = [], [], []
    for _ in range(replicates):
        indices = np.concatenate([groups[i] for i in rng.randint(len(patients), size=len(patients))])
        values = score_array(labels[indices], probabilities[indices])
        absolute.append(values)
        if baseline is not None:
            paired.append(values - score_array(labels[indices], baseline[indices]))
        if raw is not None:
            ts_delta.append(values - score_array(labels[indices], raw[indices]))
    return {"absolute": intervals(np.array(absolute)),
            "delta_vs_baseline_same_ts": intervals(np.array(paired)) if paired else None,
            "delta_vs_own_raw": intervals(np.array(ts_delta)) if ts_delta else None}


def bootstrap_results(output, spec, results):
    n = spec["config"]["bootstrap_replicates"]
    result = dict(replicates=n, interval="95% percentile", resampling_unit="patient, keeping all images of each draw",
                  conditional_on="fixed trained checkpoint and fitted temperatures; TS is not refit in bootstrap",
                  seeds=[0], seed_variation_estimated=False, results={})
    if n == 0:
        return result
    baseline_folder = output / "runs/baseline/seed0"
    for case in results:
        name = case["case"]
        folder = output / "runs" / name / "seed0"
        test = load_export(folder / "test")
        baseline_test = load_export(baseline_folder / "test") if any(c["case"] == "baseline" for c in results) else None
        if baseline_test is not None and (test["samples"] != baseline_test["samples"] or
                                         not np.array_equal(test["y_gt"], baseline_test["y_gt"])):
            raise ValueError("Paired bootstrap requires identical sample order and targets")
        result["results"][name] = {}
        for protocol in case["protocols"]:
            processed = folder / "processed" / protocol
            indices = np.load(processed / "test_indices.npy", allow_pickle=False)
            labels = test["y_gt"][indices]
            patients = [test["samples"][i]["patient_id"] for i in indices]
            result["results"][name][protocol] = {}
            raw = np.load(processed / "raw/y_pred.npy", allow_pickle=False)
            for mode in ("raw", "ts_shared", "ts_per_class"):
                probabilities = np.load(processed / mode / "y_pred.npy", allow_pickle=False)
                base = None
                if baseline_test is not None:
                    base_processed = baseline_folder / "processed" / protocol
                    if not np.array_equal(indices, np.load(base_processed / "test_indices.npy", allow_pickle=False)):
                        raise ValueError("Expert test partitions differ across conditions")
                    base = np.load(base_processed / mode / "y_pred.npy", allow_pickle=False)
                print("bootstrap {} {} {}: {} draws".format(name, protocol, mode, n), flush=True)
                result["results"][name][protocol][mode] = patient_bootstrap(labels, probabilities, patients,
                    baseline=base, raw=raw if mode != "raw" else None, replicates=n, seed=spec["config"]["bootstrap_seed"])
    return result


def build_report(output, bootstrap=True):
    output = Path(output)
    spec, cases = collect(output)
    if spec.get("report_template") and spec.get("report_template_sha256") and sha256(Path(spec["report_template"])) != spec["report_template_sha256"]:
        raise ValueError("Report template changed during the study; restore the frozen template")
    config = spec["config"]
    rows = [row for case in cases for row in case["rows"]]
    aggregate = dict(schema_version=1, generated_at=now(), completed_conditions=len(cases), expected_conditions=len(config["cases"]) + 1,
                     seed_count=1, seeds=[0], rows=rows,
                     selection_mode=spec.get("selection_mode", "dev_only"),
                     uncertainty="Single training seed. Bootstrap reflects test-patient sampling only.")
    atomic_json(output / "aggregate.json", aggregate)
    ci = bootstrap_results(output, spec, cases) if bootstrap else None
    if ci is not None:
        atomic_json(output / "bootstrap.json", ci)
    template = spec.get("report_template")
    docs = Path(spec["docs_dir"]) if spec.get("docs_dir") else None
    setup_common = ("EVA-X Tiny MIM에서 시작. 환자별 train/dev/calibration/test 분리. "
                    "seed 0 하나이며 모델 선택은 dev mAUROC(동률 시 dev NLL). "
                    "TS는 양의 온도로 sigmoid log odds를 나눕니다. "
                    "출력 순서는 Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion입니다.\n\n"
                    "`report_holdout`: 판독문 라벨의 별도 calibration split에서 TS fitting, 공식 정면 test 전체 평가.\n\n"
                    "`expert_subset`: 공식 valid 환자를 사전 고정 비율로 나눠 한쪽에서 TS fitting, 나머지만 평가하는 별도 보조 프로토콜. "
                    "두 프로토콜은 평가 표본 수가 달라 수치를 직접 비교하지 않습니다.")
    limitations = ("- seed 반복은 하지 않았습니다. seed 간 재현성이나 통계적 유의성을 단정하지 않습니다.\n"
                   "- Bootstrap은 환자 단위 95% percentile CI이며 학습·TS fitting 불확실성을 포함하지 않습니다.\n"
                   "- test 결과로 체크포인트, 조합, 온도 또는 hyperparameter를 선택하지 않았습니다.\n"
                   "- dev·report calibration은 자동 추출 라벨이고 명시적 -1은 평가/TS에서 제외, 미언급은 0입니다.\n"
                   "- 공개 체크포인트와 분할·학습 일정이 달라 공식 성능 재현 실험이 아닙니다.\n"
                   "- ECE·AECE는 질환별 양성 확률 기준이며 bin 5/10을 모두 봅니다.")
    if spec.get("selection_mode") == "post_hoc_exploratory":
        setup_common += ("\n\n**추가 탐색 조합: U-Ignore + LoRA + Mixup.** 기존 조건은 원래 학습 결과를 재사용하며, "
                         "추가 조합만 새로 학습했습니다. 조합은 중간 test 결과를 확인한 뒤 사용자가 지정했습니다.")
        limitations = limitations.replace(
            "- test 결과로 체크포인트, 조합, 온도 또는 hyperparameter를 선택하지 않았습니다.",
            "- 추가 조합은 중간 test 결과를 검토한 뒤 지정한 사후 탐색입니다. 같은 test의 점수는 독립적인 최종 검증으로 볼 수 없습니다. "
            "학습 checkpoint는 dev, 온도 fitting은 별도 calibration 환자만 사용합니다.")
    for case in cases:
        cfg = case["config"]
        details = {k: cfg.get(k) for k in ("method", "epochs", "batch_size_per_gpu", "world_size", "lr", "layer_decay", "weight_decay", "clip_grad", "early_stopping", "trainable_parameters")}
        case_setup = setup_common + "\n\n실제 학습 **{} epoch**, 조기 종료 **{}**, 선택 checkpoint **epoch {}**, 축 **{}**.\n\n```json\n{}\n```".format(
            case["trained_epochs"], case["early_stopped"], case["checkpoint_epoch"], case["axis"], json.dumps(details, indent=2, ensure_ascii=False))
        for p, values in case["protocols"].items():
            case_setup += "\n\n{}: 평가 {}장 / {}명, TS fitting 관측 라벨 {}개.".format(p, values["n_test_images"], values["n_test_patients"], values["calibration_labels"])
            case_setup += "\n\n공통 T: `{}`. 질환별 T(위 출력 순서): `{}`.".format(
                case["temperatures"][p]["ts_shared"], case["temperatures"][p]["ts_per_class"])
        interpretation = ("AUROC와 Brier/NLL/ECE/AECE를 함께 확인합니다. 개선을 자동으로 단정하지 않습니다. "
                          "공통·질환별 TS는 질환 내 순위를 보존하지만 확률 반올림에 의한 미세 차이는 확인이 필요합니다.")
        baseline = next((c for c in cases if c["case"] == "baseline"), None)
        if baseline is not None:
            interpretation += "\n\n동일 TS baseline 대비 primary macro 차이(후보 − baseline):\n\n"
            interpretation += "| 보정 | ΔmAUROC | ΔBrier | ΔNLL | ΔECE@5 | ΔAECE@5 | ΔECE@10 | ΔAECE@10 |\n|---|---:|---:|---:|---:|---:|---:|---:|"
            for mode in ("raw", "ts_shared", "ts_per_class"):
                def primary(c):
                    return next(r for r in c["rows"] if r["protocol"] == "report_holdout" and r["disease"] == "macro" and r["calibration"] == mode)
                own, base = primary(case), primary(baseline)
                delta = [own[k] - base[k] if own[k] is not None and base[k] is not None else None for k in METRICS]
                interpretation += "\n| {} | {} |".format(mode, " | ".join(number(v) for v in delta))
        if ci is None:
            interpretation += "\n\n환자 bootstrap CI는 전체 실행 종료 시 추가됩니다."
        elif config["bootstrap_replicates"]:
            interpretation += "\n\n| 보정 | 지표 | 동일 TS baseline 대비 차이의 95% CI | 유효 반복 |\n|---|---|---:|---:|"
            for mode in ("raw", "ts_shared", "ts_per_class"):
                item = ci["results"][case["case"]]["report_holdout"][mode]["delta_vs_baseline_same_ts"]
                if item:
                    for metric in METRICS:
                        interval = item["macro"][metric]
                        interpretation += "\n| {} | {} | [{}, {}] | {} |".format(mode, metric,
                            number(interval["low"]), number(interval["high"]), interval["valid_replicates"])
        text = render_report("Tiny calibration — " + case["case"], case_setup, metric_table(case["rows"]),
            interpretation, limitations,
            "서버 실험 폴더의 해당 조건 아래 `train/`, `dev/`, `calibration/`, `test/`, `processed/`에 저장됩니다. "
            "원시 데이터·환자 ID·예측 배열·가중치는 공유 문서에 포함하지 않습니다.", template)
        write_managed(output / "reports" / (case["case"] + "_seed0.md"), text)
        if docs:
            write_managed(docs / (case["case"] + "_seed0.md"), text)
    macro = [r for r in rows if r["disease"] == "macro"]
    table = ["| 조건 | 평가 | 보정 | mAUROC | Brier | NLL | ECE@5 | AECE@5 | ECE@10 | AECE@10 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in macro:
        table.append("| {} | {} | {} | {} |".format(row["case"], row["protocol"], row["calibration"], " | ".join(number(row[k]) for k in METRICS)))
    links = ["- [{}](reports/{}_seed0.md)".format(c["case"], c["case"]) for c in cases]
    description = "완료한 조건 **{}/{}**, 학습 seed **0 하나**. 조건이 완료될 때마다 갱신됩니다.\n\n".format(len(cases), len(config["cases"]) + 1) + setup_common
    state = read_json(output / "status.json") if (output / "status.json").exists() else {"tasks": {}}
    description += "\n\n| 조건 | 검사 | 학습 | dev | calibration | test | 지표/보고서 |\n|---|---|---|---|---|---|---|"
    for case in config["cases"] + [{"id": "combination"}]:
        name = case["id"]
        phases = [state["tasks"].get("check_" + name, {}).get("status", "pending")]
        phases += [state["tasks"].get(name + "_seed0_" + phase, {}).get("status", "pending")
                   for phase in ("train", "dev", "calibration", "test", "postprocess")]
        description += "\n| {} | {} |".format(name, " | ".join(phases))
    combination = output / "combination_selection.json"
    if combination.exists():
        label = "사용자 지정 추가 탐색 조합" if spec.get("selection_mode") == "post_hoc_exploratory" else "조합 선택(dev만 사용)"
        description += "\n\n{}: `{}`.".format(label, json.dumps(read_json(combination)["selected"], ensure_ascii=False))
    report = render_report("Tiny calibration 연구 진행 및 결과", description, "\n".join(table),
        "각 조건의 상세 표와 TS fitting 정보는 개별 보고서를 확인하세요. " + ("Bootstrap 계산 완료." if bootstrap else "현재 중간 결과이며 bootstrap은 전체 종료 후 계산합니다."),
        limitations, "\n".join(links) if links else "완료된 조건이 아직 없습니다.", template)
    write_managed(output / "report.md", report)
    if docs:
        docs_report = report.replace("](reports/", "](")
        write_managed(docs / "README.md", docs_report)
    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    build_report(args.output_dir, bootstrap=not args.partial)
