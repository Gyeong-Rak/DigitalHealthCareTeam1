#!/usr/bin/env python3
"""One command: split, check, train two fixed losses, test, and summarize."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from prepare_chexpert_splits import sha256

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts/run_tiny_calibration.sh"


def experiment_spec(environ, output):
    home = Path.home() / "workspace"
    defaults = {"DATA_DIR": str(home / "datasets/chexpert"), "WEIGHTS_DIR": str(home / "pretrained"),
                "SPLIT_SEED": "42", "SEED": "0", "EPOCHS": "5", "NGPU": "4", "BATCH_SIZE": "128",
                "NUM_WORKERS": "8", "ACCUMULATION": "1", "LR": "0.001", "LAYER_DECAY": "0.55",
                "WEIGHT_DECAY": "0.05", "SMOOTHING": "0.05"}
    settings = {key: environ.get(key, value) for key, value in defaults.items()}
    settings["SPLITS_DIR"] = environ.get("SPLITS_DIR", str(home / "outputs/chexpert" / ("splits_seed" + settings["SPLIT_SEED"])))
    return {"schema_version": 1, "model": "tiny", "methods": ["bce", "label_smoothing"],
            "settings": settings, "output_dir": str(output.resolve()),
            "code_sha256": {name: sha256(REPO / "scripts" / name) for name in (
                "train_tiny_calibration.py", "prepare_chexpert_splits.py", "run_tiny_calibration.sh",
                "run_tiny_comparison.py", "compare_calibration_runs.py", "evaluate_calibration.py")},
            "selection": "dev AUROC; tie: dev observed-label NLL", "test": "official frontal valid; evaluate once per trained condition",
            "temperature_scaling": False}


def plan(spec):
    steps = [{"stage": "prepare", "command": ["bash", str(LAUNCHER), "prepare"]}]
    for loss in spec["methods"]:
        steps.append({"stage": "check_" + loss, "command": ["bash", str(LAUNCHER), "check", loss]})
    for loss in spec["methods"]:
        steps += [{"stage": "train_" + loss, "command": ["bash", str(LAUNCHER), "train", loss]},
                  {"stage": "evaluate_" + loss, "command": ["bash", str(LAUNCHER), "evaluate", loss]}]
    output = Path(spec["output_dir"])
    steps.append({"stage": "summarize", "command": [sys.executable, str(REPO / "scripts/compare_calibration_runs.py"),
        "--baseline-dir", str(output / "bce/test"), "--candidate-dir", str(output / "label_smoothing/test"),
        "--output-dir", str(output / "comparison")]})
    return steps


def run_step(step, env, output):
    print("\n=== {} ===".format(step["stage"]), flush=True)
    with (output / "pipeline.log").open("a") as log:
        log.write("\n=== {} ===\n".format(step["stage"]))
        log.flush()
        process = subprocess.Popen(step["command"], cwd=str(REPO), env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   universal_newlines=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, step["command"])


def run_comparison(output, dry_run=False):
    spec = experiment_spec(os.environ, output)
    steps = plan(spec)
    if dry_run:
        print(json.dumps({"experiment": spec, "steps": steps}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    spec_path = output / "experiment.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text()) != spec:
            raise ValueError("Experiment settings changed; use the original settings or a new --output-dir")
    elif any(output.iterdir()):
        raise ValueError("Output directory has unrelated files; choose a fresh directory")
    else:
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    env = dict(os.environ, **spec["settings"])
    # Per-run paths and resume state belong to the driver, not an inherited shell setting.
    for key in ("RUN_DIR", "RESUME", "CHECKPOINT", "RESULT_DIR", "EVAL_SPLIT"):
        env.pop(key, None)
    for step in steps:
        stage = step["stage"]
        stage_env = dict(env)
        if stage.startswith("train_") or stage.startswith("evaluate_"):
            loss = stage.split("_", 1)[1]
            case = output / loss
            stage_env.update(RUN_DIR=str(case), CHECKPOINT=str(case / "best.pth"),
                             RESULT_DIR=str(case / "test"), EVAL_SPLIT="test")
            marker = case / ("train.complete.json" if stage.startswith("train_") else "test.complete.json")
            artifact = case / "best.pth" if stage.startswith("train_") else case / "test/calibration/metrics.json"
            if marker.exists():
                record = json.loads(marker.read_text())
                if not artifact.is_file() or sha256(artifact) != record["artifact_sha256"]:
                    raise ValueError("Completed artifact changed: " + str(artifact))
                print("Already complete: " + stage, flush=True)
                continue
            if stage.startswith("train_") and (case / "last.pth").is_file():
                stage_env["RESUME"] = str(case / "last.pth")
            if stage.startswith("evaluate_") and (case / "test").exists():
                # Preserve incomplete exports when retrying instead of mixing arrays from different attempts.
                suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                (case / "test").rename(case / ("test_incomplete_" + suffix))
            run_step(step, stage_env, output)
            if not artifact.is_file() or artifact.stat().st_size == 0:
                raise ValueError("Stage exited without its expected artifact: " + str(artifact))
            marker.write_text(json.dumps({"stage": stage, "artifact_sha256": sha256(artifact)}) + "\n")
        else:
            run_step(step, stage_env, output)
    (output / "complete.json").write_text(json.dumps({"report": "comparison/report.md"}) + "\n")
    print("\nALL COMPLETE: " + str(output / "comparison/report.md"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(os.environ.get("OUT_ROOT", str(Path.home() / "workspace/outputs/chexpert")))
    parser.add_argument("--output-dir", type=Path, default=default_root / (
        "tiny_comparison_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),
        help="Existing experiment directory resumes completed stages and epoch-boundary checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Print the complete plan without reading data or launching training")
    args = parser.parse_args()
    try:
        run_comparison(args.output_dir.expanduser().resolve(), args.dry_run)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, "Pipeline stopped: {}\nResume with the same --output-dir and environment settings.\n".format(error))
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Resume with the same --output-dir and environment settings.\n")


if __name__ == "__main__":
    main()
