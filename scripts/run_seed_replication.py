#!/usr/bin/env python3
"""Retrain selected conditions of a finished study with new seeds; only the seed changes.

Splits, method definitions, training settings, pretrained weights and numerical code are taken
from the source study and hash-checked, so seed-k runs are directly comparable to seed 0.
Results are written in the same layout (runs/<case>/seed<k>/...) for analyze_calibration_shift.py.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import sys

from prepare_chexpert_splits import sha256
from run_calibration_followup import NUMERIC_CODE, copy_inventory
from run_calibration_study import Study, CODE, REPO, validate_config
from study_io import atomic_json, read_json, now

STAGES = ("train", "dev", "calibration", "test", "postprocess")
DEFAULT_CASES = ("baseline", "a_mixup", "c_lora")


def task_name(case, seed, stage):
    return "{}_seed{}_{}".format(case, seed, stage)


def replication_spec(source, output, original, cases, seeds):
    for name in NUMERIC_CODE:
        if sha256(REPO / "scripts" / name) != original["code_sha256"][name]:
            raise ValueError("Numerical code differs from the source study: " + name)
    if sha256(REPO / "EVA-X/classification/models/models_eva.py") != original["architecture_sha256"]:
        raise ValueError("Architecture differs from the source study")
    if sha256(Path(original["pretrained"])) != original["pretrained_sha256"]:
        raise ValueError("Pretrained weights differ from the source study: " + original["pretrained"])
    if not seeds or any(type(s) is not int or s < 1 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct positive integers (seed 0 is the source study)")
    by_id = {case["id"]: case for case in original["config"]["cases"]}
    unknown = [c for c in cases if c not in by_id]
    if unknown or len(set(cases)) != len(cases):
        raise ValueError("Unknown or duplicated case IDs: " + ", ".join(unknown))
    for name in cases:
        method = source / "methods" / (name + ".json")
        if method.exists() and read_json(method) != by_id[name]["method"]:
            raise ValueError("Source method file differs from its configuration: " + name)
    config = copy.deepcopy(validate_config(original["config"]))
    config["description"] = "Seed replication of " + ", ".join(cases)
    config["cases"] = [by_id[name] for name in cases]
    return dict(config=config, seeds=list(seeds), output_dir=str(output), data_root=original["data_root"],
                pretrained=original["pretrained"], pretrained_sha256=original["pretrained_sha256"],
                architecture_sha256=original["architecture_sha256"], source_study=str(source),
                source_manifest_sha256=sha256(source / "study.json"),
                code_sha256={name: sha256(REPO / "scripts" / name) for name in (*CODE, "run_seed_replication.py")})


def split_inventory(source):
    manifest = read_json(source / "splits/manifest.json")
    inventory = {"splits/manifest.json": sha256(source / "splits/manifest.json")}
    inventory.update({"splits/" + name: digest for name, digest in manifest["files"].items()})
    return inventory


class SeedStudy(Study):
    def __init__(self, output, spec):
        # Study.__init__ registers seed-0 tasks for every original case; register only this plan.
        self.output, self.spec, self.config = Path(output), spec, spec["config"]
        state_path = self.output / "status.json"
        self.state = read_json(state_path) if state_path.exists() else {"created_at": now(), "tasks": {}}
        self.state.update(status="running", pid=os.getpid(), current=None, error=None)
        for seed in spec["seeds"]:
            for case in self.config["cases"]:
                self.state["tasks"].setdefault("check_" + case["id"], {"status": "pending"})
                for stage in STAGES:
                    self.state["tasks"].setdefault(task_name(case["id"], seed, stage), {"status": "pending"})
        self.env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"),
                        PYTHONPATH=str(REPO / "compat") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        self.env.pop("RESUME", None)

    def save(self):
        self.state["heartbeat"] = now()
        atomic_json(self.output / "status.json", self.state)
        lines = ["Status: {} | current: {} | heartbeat (UTC): {}".format(
            self.state["status"], self.state.get("current"), self.state["heartbeat"]), ""]
        current = self.state.get("current") or ""
        if current.endswith("_train"):
            case, seed = current.removesuffix("_train").rsplit("_seed", 1)
            progress = self.output / "runs" / case / ("seed" + seed) / "train/progress.json"
            if progress.exists():
                p = read_json(progress)
                lines += ["Epoch {}/{} | phase {} | step {}/{} | epoch ETA {}s".format(p.get("epoch"), p.get("epochs"),
                          p.get("phase"), p.get("step", "—"), p.get("steps", "—"), round(p.get("epoch_eta_seconds", 0))), ""]
        lines.append("{:<12} {:<6} {}".format("case", "seed", " / ".join(STAGES)))
        for seed in self.spec["seeds"]:
            for case in self.config["cases"]:
                lines.append("{:<12} {:<6} {}".format(case["id"], seed, " / ".join(
                    self.state["tasks"][task_name(case["id"], seed, s)]["status"] for s in STAGES)))
        errors = [(k, v) for k, v in self.state["tasks"].items() if v["status"] in ("failed", "blocked", "interrupted")]
        if errors:
            lines += ["", "Errors:"] + ["{}: {}".format(k, v.get("error")) for k, v in errors]
        lines += ["", "Recent events (UTC):"] + self.state.get("events", [])[-10:]
        temporary = self.output / "STATUS.tmp"
        temporary.write_text("# Seed 반복 진행 상황\n\n```text\n" + "\n".join(lines) + "\n```\n", encoding="utf-8")
        temporary.replace(self.output / "STATUS.md")

    def run_case(self, case, seed):
        folder = self.output / "runs" / case["id"] / ("seed" + str(seed))
        train_dir = folder / "train"
        def blocked(reason):
            for stage in STAGES:
                record = self.state["tasks"][task_name(case["id"], seed, stage)]
                if record["status"] in ("pending", "blocked"):
                    record.update(status="blocked", error=reason)
            self.save()
            return False
        receipt = self.output / "checks" / (case["id"] + ".json")
        if not self.valid("check_" + case["id"]):
            command = self.trainer_command(case, "check", seed) + ["--check-receipt", str(receipt)]
            if not self.task("check_" + case["id"], lambda: command, [receipt]):
                return blocked("preflight failed")
        if not self.task(task_name(case["id"], seed, "train"), lambda: self.trainer_command(case, "train", seed),
                         [train_dir / p for p in ("best.pth", "last.pth", "config.json", "history.jsonl")]):
            return blocked("training failed")
        for split in ("dev", "calibration", "test"):
            dest = folder / split
            if not self.task(task_name(case["id"], seed, split),
                             lambda split=split: self.trainer_command(case, "evaluate", seed, split),
                             [dest / p for p in ("logits.npy", "y_gt.npy", "y_pred.npy", "label_mask.npy", "indices.npy",
                                                 "samples.json", "evaluation.json")]):
                return blocked(split + " export failed")
        processed = folder / "processed"
        command = [sys.executable, str(REPO / "scripts/calibration_postprocess.py"),
                   "--calibration-dir", str(folder / "calibration"), "--test-dir", str(folder / "test"),
                   "--output-dir", str(processed), "--expert-fraction", str(self.config["expert_calibration_fraction"]),
                   "--split-seed", str(self.config["split_seed"])] + ([] if self.config["plots"] else ["--no-plots"])
        return self.task(task_name(case["id"], seed, "postprocess"), lambda: command,
                         lambda: sorted(p for p in processed.rglob("*") if p.is_file()))

    def run(self):
        self.save()
        # Seed-major order: a finished seed already gives a complete paired comparison.
        results = [self.run_case(case, seed) for seed in self.spec["seeds"] for case in self.config["cases"]]
        failed = [k for k, v in self.state["tasks"].items() if v["status"] != "complete"]
        self.state.update(current=None, status="failed" if failed else "complete", finished_at=now())
        self.save()
        print("{}: {}".format("ALL COMPLETE" if all(results) and not failed else "FAILED/PARTIAL", self.output / "STATUS.md"), flush=True)
        return 0 if all(results) and not failed else 1


def main():
    base = Path.home() / "workspace/outputs/chexpert"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=base / "tiny_study_seed0")
    parser.add_argument("--output-dir", type=Path, default=base / "tiny_seed_replication")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source, output = args.source_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    try:
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Source and replication output must be separate directories")
        original = read_json(source / "study.json")
        spec = replication_spec(source, output, original, args.cases, args.seeds)
        if args.dry_run:
            print(json.dumps(dict(cases=args.cases, seeds=args.seeds, training_runs=len(args.cases) * len(args.seeds),
                                  training=spec["config"]["training"], pretrained=spec["pretrained"],
                                  data_root=spec["data_root"], output_dir=str(output)), indent=2))
            return 0
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another replication process owns this output directory")
            manifest = output / "study.json"
            if manifest.exists():
                if read_json(manifest) != spec:
                    raise ValueError("Replication code/settings changed; restore them or use a new output directory")
            elif any(p.name != ".lock" for p in output.iterdir()):
                raise ValueError("Choose a fresh output directory")
            else:
                atomic_json(manifest, spec)
            copy_inventory(source, output, split_inventory(source), lambda message: print(message, flush=True))
            study = SeedStudy(output, spec)
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
        print("Interrupted. Resume with the same command.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print("Replication stopped: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    def stop_requested(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGHUP, stop_requested)
    sys.exit(main())
