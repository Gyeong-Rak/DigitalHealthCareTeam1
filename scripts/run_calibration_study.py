#!/usr/bin/env python3
"""Sequential PPT study: preflight, all axes, dev-only combination, TS and reports."""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time

from prepare_chexpert_splits import sha256
from study_io import atomic_json, read_json, now

REPO = Path(__file__).resolve().parents[1]
CODE = ("run_calibration_study.py", "train_tiny_calibration.py", "calibration_methods.py",
        "prepare_chexpert_splits.py", "evaluate_calibration.py", "calibration_postprocess.py",
        "study_report.py", "study_io.py", "study_status.py")


def validate_config(config):
    allowed = {"schema_version", "description", "seeds", "training", "split_seed", "expert_calibration_fraction",
               "bootstrap_replicates", "bootstrap_seed", "combination_auroc_tolerance", "retries", "plots", "cases"}
    if set(config) != allowed or config["schema_version"] != 1:
        raise ValueError("Unknown or missing study settings")
    if config["seeds"] != [0]:
        raise ValueError("This time-limited study uses seed 0 only")
    training = config["training"]
    if set(training) != {"epochs", "min_epochs", "patience", "min_delta", "batch_size", "gpus", "workers",
                         "accumulation", "lr", "layer_decay", "weight_decay", "clip_grad"}:
        raise ValueError("Unknown or missing training settings")
    for key in ("epochs", "min_epochs", "batch_size", "gpus", "accumulation"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError("Training dimensions must be positive integers")
    for key in ("workers", "patience"):
        if type(training[key]) is not int or training[key] < 0:
            raise ValueError("Worker/patience counts must be nonnegative integers")
    if training["min_epochs"] > training["epochs"]:
        raise ValueError("min_epochs exceeds max epochs")
    for key in ("min_delta", "lr", "layer_decay", "weight_decay", "clip_grad"):
        if not math.isfinite(training[key]) or training[key] < 0:
            raise ValueError("Training scalars must be finite and nonnegative")
    if not training["lr"] or not 0 < training["layer_decay"] <= 1:
        raise ValueError("Invalid LR/layer decay")
    for key in ("retries", "bootstrap_replicates", "split_seed", "bootstrap_seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError("Invalid count/seed: " + key)
    if not 0 <= config["expert_calibration_fraction"] < 1 or not math.isfinite(config["combination_auroc_tolerance"]) or config["combination_auroc_tolerance"] < 0:
        raise ValueError("Invalid expert fraction or AUROC tolerance")
    if type(config["plots"]) is not bool or not config["cases"]:
        raise ValueError("plots must be boolean and cases must not be empty")
    names = set()
    for case in config["cases"]:
        if set(case) != {"id", "axis", "method"} or not re.fullmatch(r"[a-z][a-z0-9_]*", case["id"]):
            raise ValueError("Invalid case definition")
        if case["id"] in names or case["id"] == "combination" or case["axis"] not in ("baseline", "A", "B", "C"):
            raise ValueError("Duplicate/reserved case ID or unknown axis")
        names.add(case["id"])
    if config["cases"][0]["id"] != "baseline" or config["cases"][0]["method"] != {"loss": "bce", "scope": "full", "uncertain": "original"}:
        raise ValueError("First case must be the full/original/BCE baseline")
    return config


def choose_combination(cases, dev_results, tolerance):
    """Use only dev AUROC/NLL. Baseline is a fallback candidate on every axis."""
    baseline = dev_results["baseline"]
    selected, method = {}, dict(cases[0]["method"])
    for axis in ("C", "B", "A"):
        candidates = [case for case in cases if case["axis"] in (axis, "baseline")]
        eligible = [case for case in candidates if dev_results[case["id"]]["auroc"] >= baseline["auroc"] - tolerance]
        winner = min(eligible, key=lambda case: (dev_results[case["id"]]["nll"], -dev_results[case["id"]]["auroc"], case["id"]))
        selected[axis] = winner["id"]
        method.update(winner["method"] if winner["id"] != "baseline" else {})
    return {"method": method, "selected": selected, "criterion": "dev NLL minimum subject to dev mAUROC >= baseline - tolerance",
            "auroc_tolerance": tolerance, "dev_metrics": dev_results, "test_used": False}


class Study:
    def __init__(self, output, spec):
        self.output, self.spec = Path(output), spec
        self.config = spec["config"]
        state_path = self.output / "status.json"
        self.state = read_json(state_path) if state_path.exists() else {"created_at": now(), "tasks": {}}
        self.state.update(status="running", pid=os.getpid(), current=None, error=None)
        self.state["tasks"].setdefault("prepare", {"status": "pending"})
        for case in self.config["cases"] + [{"id": "combination"}]:
            self.state["tasks"].setdefault("check_" + case["id"], {"status": "pending"})
            for stage in ("train", "dev", "calibration", "test", "postprocess"):
                self.state["tasks"].setdefault(case["id"] + "_seed0_" + stage, {"status": "pending"})
        self.state["tasks"].setdefault("summarize", {"status": "pending"})
        self.env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"),
                        PYTHONPATH=str(REPO / "compat") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        self.env.pop("RESUME", None)

    def save(self):
        self.state["heartbeat"] = now()
        atomic_json(self.output / "status.json", self.state)
        from study_status import write_snapshot
        write_snapshot(self.output)

    def event(self, text):
        self.state["events"] = (self.state.get("events", []) + [now() + " " + text])[-40:]

    def marker(self, task):
        return self.output / "completed" / (task + ".json")

    def valid(self, task):
        if not self.marker(task).is_file():
            return False
        marker = read_json(self.marker(task))
        for relative, digest in marker["artifacts"].items():
            path = self.output / relative
            if not path.is_file() or sha256(path) != digest:
                raise ValueError("Completed artifact changed: " + str(path))
        self.state["tasks"].setdefault(task, {})["status"] = "complete"
        return True

    def stream(self, task, command):
        log_path = self.output / "logs" / (task + ".log")
        log_path.parent.mkdir(exist_ok=True)
        print("\n=== {} ===".format(task), flush=True)
        events = queue.Queue()
        process = subprocess.Popen(command, cwd=str(REPO), env=self.env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        record = self.state["tasks"][task]
        record.update(child_pid=process.pid, log="logs/" + task + ".log", command=command)
        self.save()
        def read_lines():
            try:
                for line in process.stdout:
                    events.put(line)
            finally:
                events.put(None)
        threading.Thread(target=read_lines, daemon=True).start()
        heartbeat = time.monotonic()
        try:
            with log_path.open("a", encoding="utf-8") as log, (self.output / "pipeline.log").open("a", encoding="utf-8") as pipeline:
                pipeline.write("\n=== {} {} ===\n".format(now(), task))
                while True:
                    try:
                        line = events.get(timeout=1)
                    except queue.Empty:
                        line = ""
                    if line is None:
                        break
                    if line:
                        print(line, end="", flush=True)
                        log.write(line)
                        pipeline.write(line)
                        log.flush()
                        pipeline.flush()
                        record["last_output_at"] = now()
                        record["tail"] = (record.get("tail", []) + [line.rstrip()[-2000:]])[-12:]
                        if "Error:" in line or "Exception:" in line or "CUDA out of memory" in line:
                            record["error_summary"] = (record.get("error_summary", []) + [line.strip()[-1500:]])[-4:]
                        if line.startswith(("bootstrap ", "CHECK OK", "EARLY STOP", "POSTPROCESS COMPLETE", "TRAIN COMPLETE")):
                            record["activity"] = line.strip()[:300]
                    if time.monotonic() - heartbeat >= 10:
                        self.save()
                        heartbeat = time.monotonic()
                code = process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        finally:
            process.stdout.close()
        return code

    def task(self, task, command_factory, artifacts):
        for name, digest in self.spec.get("code_sha256", {}).items():
            if sha256(REPO / "scripts" / name) != digest:
                raise ValueError("Study source changed during execution: " + name)
        if self.spec.get("architecture_sha256") and sha256(REPO / "EVA-X/classification/models/models_eva.py") != self.spec["architecture_sha256"]:
            raise ValueError("EVA-X architecture changed during execution")
        if self.valid(task):
            print("SKIP complete: " + task, flush=True)
            self.save()
            return True
        if task != "summarize" and self.marker("summarize").exists():
            self.marker("summarize").unlink()  # New results require a fresh aggregate/CI.
        self.state["current"] = task
        record = self.state["tasks"].setdefault(task, {})
        for attempt in range(self.config["retries"] + 1):
            record.update(status="running", started_at=now(), attempt=attempt + 1, error=None, error_summary=[])
            self.event("START {} (attempt {})".format(task, attempt + 1))
            self.save()
            started = time.monotonic()
            try:
                code = self.stream(task, command_factory())
                if code:
                    raise RuntimeError("Child process exited with code {}{}".format(code,
                        "; " + " | ".join(record["error_summary"]) if record["error_summary"] else ""))
                paths = artifacts() if callable(artifacts) else artifacts
                if not paths or any(not path.is_file() or not path.stat().st_size for path in paths):
                    raise ValueError("Missing/empty output after successful exit")
                atomic_json(self.marker(task), {"artifacts": {str(p.relative_to(self.output)): sha256(p) for p in paths}})
                record.update(status="complete", finished_at=now(), elapsed_seconds=time.monotonic() - started)
                self.event("DONE {} ({:.0f}s)".format(task, record["elapsed_seconds"]))
                self.save()
                return True
            except (OSError, ValueError, RuntimeError) as error:
                record.update(status="failed", finished_at=now(), error=str(error))
                self.event("ERROR {}: {}".format(task, error))
                self.save()
                print("FAILED {}: {}".format(task, error), flush=True)
        return False

    def trainer_command(self, case, stage, seed=0, split=None):
        method_path = self.output / "methods" / (case["id"] + ".json")
        if method_path.exists():
            if read_json(method_path) != case["method"]:
                raise ValueError("Frozen method changed: " + case["id"])
        else:
            atomic_json(method_path, case["method"])
        training = self.config["training"]
        folder = self.output / "runs" / case["id"] / ("seed" + str(seed))
        command = [sys.executable]
        if stage == "train":
            command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=" + str(training["gpus"])]
        command += [str(REPO / "scripts/train_tiny_calibration.py"), stage, "--data-root", self.spec["data_root"],
                    "--splits-dir", str(self.output / "splits"), "--experiment-config", str(method_path), "--seed", str(seed)]
        for key in ("epochs", "batch_size", "workers", "accumulation", "lr", "layer_decay", "weight_decay", "clip_grad",
                    "patience", "min_epochs", "min_delta"):
            command += ["--" + key.replace("_", "-"), str(training[key])]
        if stage in ("train", "check"):
            command += ["--pretrained", self.spec["pretrained"]]
        if stage == "train":
            train_dir = folder / "train"
            if (train_dir / "last.pth").exists():
                command += ["--resume", str(train_dir / "last.pth")]
            elif train_dir.exists() and any(train_dir.iterdir()):
                train_dir.rename(train_dir.with_name("train_incomplete_" + str(time.time_ns())))
            command += ["--output-dir", str(train_dir)]
        if stage == "evaluate":
            dest = folder / split
            if dest.exists():
                dest.rename(dest.with_name(split + "_incomplete_" + str(time.time_ns())))
            command += ["--checkpoint", str(folder / "train/best.pth"), "--eval-split", split, "--output-dir", str(dest)]
        return command

    def run_case(self, case):
        seed = 0
        prefix = case["id"] + "_seed0_"
        folder = self.output / "runs" / case["id"] / "seed0"
        train_dir = folder / "train"
        def blocked(reason):
            for stage in ("train", "dev", "calibration", "test", "postprocess"):
                record = self.state["tasks"][prefix + stage]
                if record["status"] in ("pending", "blocked"):
                    record.update(status="blocked", error=reason)
            self.save()
            return False
        check = "check_" + case["id"]
        # A successful preflight has its own small receipt, never a fake checkpoint.
        receipt = self.output / "checks" / (case["id"] + ".json")
        if not self.valid(check):
            def check_command():
                return self.trainer_command(case, "check") + ["--check-receipt", str(receipt)]
            if not self.task(check, check_command, [receipt]):
                return blocked("preflight failed")
        if not self.task(prefix + "train", lambda: self.trainer_command(case, "train", seed),
                         [train_dir / p for p in ("best.pth", "last.pth", "config.json", "history.jsonl")]):
            return blocked("training failed")
        for split in ("dev", "calibration", "test"):
            dest = folder / split
            if not self.task(prefix + split, lambda split=split: self.trainer_command(case, "evaluate", seed, split),
                             [dest / p for p in ("logits.npy", "y_gt.npy", "y_pred.npy", "label_mask.npy", "indices.npy", "samples.json", "evaluation.json")]):
                return blocked(split + " export failed")
        processed = folder / "processed"
        command = [sys.executable, str(REPO / "scripts/calibration_postprocess.py"),
            "--calibration-dir", str(folder / "calibration"), "--test-dir", str(folder / "test"),
            "--output-dir", str(processed), "--expert-fraction", str(self.config["expert_calibration_fraction"]),
            "--split-seed", str(self.config["split_seed"])]
        if not self.config["plots"]:
            command.append("--no-plots")
        success = self.task(prefix + "postprocess", lambda: command,
                            lambda: sorted(p for p in processed.rglob("*") if p.is_file()))
        if success:
            from study_report import build_report
            if not self.marker("summarize").exists():
                build_report(self.output, bootstrap=False)
            self.event("RESULT {}: docs and calibration report updated".format(case["id"]))
            self.save()
        return success

    def run(self):
        self.save()
        prepare = [sys.executable, str(REPO / "scripts/prepare_chexpert_splits.py"), "--data-root", self.spec["data_root"],
                   "--output-dir", str(self.output / "splits"), "--seed", str(self.config["split_seed"]), "--check-images"]
        def prepare_command():
            folder = self.output / "splits"
            if folder.exists() and not (folder / "manifest.json").exists():
                folder.rename(self.output / ("splits_incomplete_" + str(time.time_ns())))
            return prepare
        if not self.task("prepare", prepare_command, [self.output / "splits" / p for p in
            ("manifest.json", "train.csv", "dev.csv", "calibration.csv", "test.csv")]):
            raise RuntimeError("Dataset preparation failed; no training was started")
        for case in self.config["cases"]:
            self.run_case(case)  # Failed cases do not erase results or cancel independent cases.
        dev, inputs = {}, {}
        for case in self.config["cases"]:
            name = case["id"]
            path = self.output / "runs" / name / "seed0/dev/evaluation.json"
            if self.valid(name + "_seed0_dev"):
                dev[name] = read_json(path)["metrics"]
                inputs[name] = sha256(path)
        if len(dev) == len(self.config["cases"]):
            selection = dict(choose_combination(self.config["cases"], dev, self.config["combination_auroc_tolerance"]), inputs=inputs)
            path = self.output / "combination_selection.json"
            if path.exists() and read_json(path) != selection:
                raise ValueError("Frozen dev-only combination selection changed")
            atomic_json(path, selection)
            self.run_case({"id": "combination", "axis": "combination", "method": selection["method"]})
        else:
            for name, record in self.state["tasks"].items():
                if name == "check_combination" or name.startswith("combination_seed0_"):
                    record.update(status="blocked", error="Not all independent dev results are available")
        report = [sys.executable, str(REPO / "scripts/study_report.py"), "--output-dir", str(self.output)]
        self.task("summarize", lambda: report, [self.output / "report.md", self.output / "aggregate.json", self.output / "bootstrap.json"])
        errors = {key: value for key, value in self.state["tasks"].items() if value["status"] != "complete"}
        self.state.update(current=None, status="failed" if errors else "complete", finished_at=now())
        self.save()
        print("\n{}: {}".format("FAILED/PARTIAL" if errors else "ALL COMPLETE", self.output / "report.md"), flush=True)
        return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--docs-dir", type=Path, help="Public-safe Markdown destination; default docs/experiments/generated/<study>")
    parser.add_argument("--report-template", type=Path, help="Team Markdown template with six placeholders")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base = Path.home() / "workspace"
    output = (args.output_dir or Path(os.environ.get("OUT_ROOT", str(base / "outputs/chexpert"))) /
              ("tiny_study_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))).expanduser().resolve()
    old = read_json(output / "study.json") if (output / "study.json").exists() else None
    try:
        config = validate_config(read_json(args.config or REPO / "configs/tiny_study_day.json") if args.config or not old else old["config"])
        pretrained = (args.pretrained or (Path(old["pretrained"]) if old else
                      Path(os.environ.get("WEIGHTS_DIR", str(base / "pretrained"))) / "eva_x_tiny_patch16_merged520k_mim.pt")).expanduser().resolve()
        spec = dict(config=config, data_root=str((args.data_root or Path(old["data_root"] if old else os.environ.get("DATA_DIR", str(base / "datasets/chexpert")))).expanduser().resolve()),
            pretrained=str(pretrained),
            code_sha256={p: sha256(REPO / "scripts" / p) for p in CODE}, output_dir=str(output))
        spec["architecture_sha256"] = sha256(REPO / "EVA-X/classification/models/models_eva.py")
        spec["docs_dir"] = str((args.docs_dir or Path(old["docs_dir"] if old else
            REPO / "docs/experiments/generated" / output.name)).expanduser().resolve())
        template = args.report_template or ((Path(old["report_template"]) if old.get("report_template") else None)
                                            if old else REPO / "docs/experiments/_template.md")
        spec["report_template"] = str(template.expanduser().resolve()) if template else None
        spec["report_template_sha256"] = sha256(template.expanduser()) if template else None
        if not args.dry_run:
            spec["pretrained_sha256"] = sha256(pretrained)
        elif old:
            spec["pretrained_sha256"] = old["pretrained_sha256"]
        if old and old != spec:
            raise ValueError("Study code/settings changed; resume with original code or use a new output directory")
        if args.dry_run:
            print(json.dumps(dict(spec, training_runs=len(config["cases"]) + 1,
                stages=["prepare", "check/train/dev/calibration/test/postprocess per case", "dev-only combination", "bootstrap/report"]), indent=2))
            return 0
        if template:
            from study_report import render_report
            render_report("check", "check", "check", "check", "check", "check", template)
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another study process owns this output directory")
            if not old and any(p.name != ".lock" for p in output.iterdir()):
                raise ValueError("Choose a fresh output directory")
            atomic_json(output / "study.json", spec)
            study = Study(output, spec)
            try:
                return study.run()
            except BaseException as error:
                study.state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
                active = study.state.get("current")
                if active and study.state["tasks"][active]["status"] == "running":
                    study.state["tasks"][active].update(status=study.state["status"], error=str(error))
                study.save()
                raise
    except KeyboardInterrupt:
        print("Interrupted. Resume with --output-dir " + str(output), file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print("Study stopped: {}\nResume: python scripts/run_calibration_study.py --output-dir {}".format(error, output), file=sys.stderr)
        return 1


if __name__ == "__main__":
    def stop_requested(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGHUP, stop_requested)
    sys.exit(main())
