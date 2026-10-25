#!/usr/bin/env python3
"""Read-only terminal status, heartbeat, latest dev/test results and error tails."""
import argparse
import datetime
import json
from pathlib import Path
import time

from study_io import read_json


def display(output):
    state = read_json(output / "status.json")
    tasks = state["tasks"]
    counts = {name: sum(t["status"] == name for t in tasks.values())
              for name in ("complete", "running", "pending", "failed", "blocked", "interrupted")}
    timestamp = datetime.datetime.fromisoformat(state["heartbeat"])
    age = (datetime.datetime.now(datetime.timezone.utc) - timestamp).total_seconds()
    lines = ["Study: " + output.name, "Status: {} | complete {}/{} | failed {} | blocked {}".format(
        state["status"], counts["complete"], len(tasks), counts["failed"], counts["blocked"]),
        "Heartbeat: {:.0f}s ago{}".format(age, " — STALE, check process/log" if state["status"] == "running" and age > 90 else ""),
        "Current: " + str(state.get("current"))]
    current = state.get("current")
    if current and current.endswith("_train"):
        case = current.removesuffix("_seed0_train")
        progress = output / "runs" / case / "seed0/train/progress.json"
        if progress.exists():
            p = read_json(progress)
            lines.append("Epoch {}/{} | phase {} | step {}/{} | loss {} | epoch ETA {}s | peak GPU {} MB".format(
                p.get("epoch"), p.get("epochs"), p.get("phase"), p.get("step", "—"), p.get("steps", "—"),
                p.get("train_loss", p.get("train_loss_rank0", "—")), round(p.get("epoch_eta_seconds", 0)), round(p.get("gpu_peak_memory_mb", 0))))
            if p.get("dev"):
                lines.append("Latest dev: " + str(p["dev"]))
            else:
                history = progress.parent / "history.jsonl"
                for line in reversed(history.read_text().splitlines() if history.exists() else []):
                    try:
                        last = json.loads(line)
                    except ValueError:
                        continue
                    lines.append("Latest completed epoch {} dev: {}".format(last["epoch"], last["dev"]))
                    break
    if current and current in tasks:
        lines.append("Log: " + str(output / tasks[current].get("log", "pipeline.log")))
        if tasks[current].get("activity"):
            lines.append("Activity: " + tasks[current]["activity"])
    durations = [t["elapsed_seconds"] for name, t in tasks.items()
                 if name.endswith("_train") and t["status"] == "complete" and t.get("elapsed_seconds")]
    remaining = sum(name.endswith("_train") and t["status"] in ("pending", "running") for name, t in tasks.items())
    if durations and remaining:
        average = sum(durations) / len(durations)
        lines.append("Training-only rough estimate for {} remaining runs: {:.1f}h (based on completed average; excludes evaluation/TS/bootstrap)".format(
            remaining, average * remaining / 3600))
    lines += ["", "Condition stages: check / train / dev / calibration / test / metrics"]
    for name in tasks:
        if name.startswith("check_"):
            case = name.removeprefix("check_")
            phases = [tasks[name]["status"]] + [tasks.get(case + "_seed0_" + phase, {}).get("status", "pending")
                for phase in ("train", "dev", "calibration", "test", "postprocess")]
            lines.append("{:<18} {}".format(case, " / ".join(phases)))
    aggregate = output / "aggregate.json"
    if aggregate.exists():
        report = read_json(aggregate)
        lines += ["", "Completed condition results (primary full test):", "condition                 TS              mAUROC    Brier     NLL       ECE@10"]
        for row in report["rows"]:
            if row["disease"] == "macro" and row["protocol"] == "report_holdout":
                def f(x):
                    return "N/A" if x is None else "{:.5f}".format(x)
                lines.append("{:<25} {:<15} {}   {}   {}   {}".format(row["case"], row["calibration"],
                    f(row["auroc"]), f(row["brier"]), f(row["nll"]), f(row["ece_10"])))
    errors = [(key, value) for key, value in tasks.items() if value["status"] in ("failed", "blocked", "interrupted")]
    if errors:
        lines += ["", "Errors:"]
        for key, value in errors:
            lines.append("{}: {} — {}".format(key, value["status"], value.get("error")))
    if state.get("error"):
        lines.append("Driver error: " + state["error"])
    lines += ["", "Recent important events (UTC):"] + state.get("events", [])[-12:]
    lines += ["", "Report: " + str(output / "report.md"), "Detailed errors: " + str(output / "status.json")]
    return "\n".join(lines), state["status"]


def write_snapshot(output):
    text, _ = display(output)
    path = output / "STATUS.md"
    temporary = path.with_suffix(".tmp")
    temporary.write_text("# 실험 현재 상황\n\n약 10초마다 갱신합니다. 시간은 UTC입니다. "
                         "파일 상단 heartbeat가 오래되면 프로세스와 로그를 확인하세요.\n\n```text\n" + text + "\n```\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--watch", type=float, default=0, help="Refresh seconds, e.g. --watch 10")
    args = parser.parse_args()
    if args.watch < 0:
        parser.error("watch must be nonnegative")
    try:
        while True:
            text, status = display(args.output_dir.expanduser())
            print(("\033[2J\033[H" if args.watch else "") + text, flush=True)
            if not args.watch or status in ("complete", "failed", "interrupted"):
                break
            time.sleep(max(args.watch, 1))
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError) as error:
        parser.exit(1, "Cannot read study status: {}\n".format(error))


if __name__ == "__main__":
    main()
