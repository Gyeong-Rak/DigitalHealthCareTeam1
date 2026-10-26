#!/usr/bin/env python3
"""Reuse completed original conditions and train only LoRA + Mixup + U-Ignore."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys

from prepare_chexpert_splits import sha256
from run_calibration_study import Study, CODE, REPO, validate_config
from study_io import atomic_json, read_json, now

METHOD = dict(scope="lora", loss="mixup", uncertain="ignore", lora_rank=8,
              lora_alpha=16.0, mixup_alpha=.2)
NUMERIC_CODE = ("train_tiny_calibration.py", "calibration_methods.py", "prepare_chexpert_splits.py",
                "evaluate_calibration.py", "calibration_postprocess.py")


def safe_relative(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe artifact path in source manifest")
    resolved = (root / path).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError("Artifact resolves outside the source study")
    return path


def import_inventory(source, original, progress=None):
    """Only small result artifacts; never copy model/optimizer weights or images."""
    for name in NUMERIC_CODE:
        if sha256(REPO / "scripts" / name) != original["code_sha256"][name]:
            raise ValueError("Numerical code differs from original conditions: " + name)
    if sha256(REPO / "EVA-X/classification/models/models_eva.py") != original["architecture_sha256"]:
        raise ValueError("Architecture differs from original conditions")
    inventory = {}
    def include(relative, digest=None):
        path = safe_relative(source, relative)
        if path.suffix in (".pt", ".pth", ".ckpt"):
            raise ValueError("Checkpoint copying is not part of result import")
        actual = sha256(source / path)
        if digest is not None and digest != actual:
            raise ValueError("Source artifact changed: " + str(path))
        inventory[str(path)] = actual
    manifest = read_json(source / "splits/manifest.json")
    include("splits/manifest.json")
    for name, digest in manifest["files"].items():
        include("splits/" + name, digest)
    for case_index, case in enumerate(original["config"]["cases"]):
        name = case["id"]
        marker = "completed/{}_seed0_postprocess.json".format(name)
        if not (source / marker).is_file():
            raise ValueError("Original condition is not complete: " + name)
        include(marker)
        for relative, digest in read_json(source / marker)["artifacts"].items():
            include(relative, digest)
        run = source / "runs" / name / "seed0"
        trained = read_json(source / "completed" / (name + "_seed0_train.json"))["artifacts"]
        for relative in ("train/config.json", "train/history.jsonl"):
            relative = str(Path("runs") / name / "seed0" / relative)
            include(relative, trained[relative])
        # Dev metrics preserve selection provenance; calibration/test exports
        # preserve paired bootstrap and TS provenance. No inference is repeated.
        for split in ("dev", "calibration", "test"):
            evaluation = read_json(run / split / "evaluation.json")
            expected = trained[str(Path("runs") / name / "seed0/train/best.pth")]
            if evaluation["checkpoint_sha256"] != expected or evaluation["training_config"]["split_manifest_sha256"] != inventory["splits/manifest.json"]:
                raise ValueError("Source evaluation checkpoint/split provenance differs: " + name)
            for file in (run / split).iterdir():
                if file.is_file() and file.suffix in (".json", ".npy"):
                    include(str(file.relative_to(source)))
        summary = read_json(run / "processed/summary.json")
        for kind in ("calibration", "test"):
            for file, digest in summary["inputs"][kind].items():
                include(str(Path("runs") / name / "seed0" / kind / file), digest)
        if progress:
            progress("Verified original condition {}/{}: {}".format(case_index + 1, len(original["config"]["cases"]), name))
    return inventory


def copy_inventory(source, output, inventory, progress=None):
    for index, (relative, digest) in enumerate(inventory.items()):
        src, dst = source / relative, output / relative
        if dst.exists():
            if sha256(dst) != digest:
                raise ValueError("Imported artifact changed: " + relative)
            if progress and (index % 20 == 0 or index + 1 == len(inventory)):
                progress("Verified imported files {}/{}".format(index + 1, len(inventory)))
            continue
        if sha256(src) != digest:
            raise ValueError("Source changed while importing: " + relative)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".import.tmp")
        shutil.copyfile(src, tmp)
        if sha256(tmp) != digest:
            raise ValueError("Result copy failed integrity check: " + relative)
        tmp.replace(dst)
        if progress and (index % 20 == 0 or index + 1 == len(inventory)):
            progress("Copied result files {}/{}; no model weights or images".format(index + 1, len(inventory)))


def followup_spec(source, output, original):
    config = copy.deepcopy(validate_config(original["config"]))
    config["description"] = "Post-hoc exploratory LoRA + Mixup + U-Ignore; original conditions imported"
    template = REPO / "docs/experiments/_template.md"
    return dict(config=config, output_dir=str(output), data_root=original["data_root"],
                pretrained=original["pretrained"], pretrained_sha256=original["pretrained_sha256"],
                architecture_sha256=original["architecture_sha256"], source_study=str(source),
                source_manifest_sha256=sha256(source / "study.json"), selection_mode="post_hoc_exploratory",
                combination_method=METHOD, code_sha256={name: sha256(REPO / "scripts" / name)
                    for name in (*CODE, "run_calibration_followup.py")},
                docs_dir=str(REPO / "docs/experiments/generated" / output.name),
                report_template=str(template), report_template_sha256=sha256(template))


def run_followup(study):
    # Imported stages are provenance-marked as complete without claiming that
    # their original model checkpoints have been copied into this new run.
    for name, record in study.state["tasks"].items():
        if name == "prepare" or any(name == "check_" + c["id"] or name.startswith(c["id"] + "_seed0_")
                                    for c in study.config["cases"]):
            record.update(status="complete", imported=True)
    study.state.update(selection_mode="post_hoc_exploratory", imported_conditions=len(study.config["cases"]))
    study.event("Original conditions imported without retraining; starting requested exploratory combination")
    study.save()
    from study_report import build_report
    if not study.marker("summarize").exists():
        build_report(study.output, bootstrap=False)
    case = dict(id="combination", axis="combination", method=METHOD)
    success = study.run_case(case)
    if not success:
        build_report(study.output, bootstrap=False)
        study.state.update(status="failed", current=None, error="Follow-up combination failed; original results are preserved")
        study.save()
        return 1
    command = [sys.executable, str(REPO / "scripts/study_report.py"), "--output-dir", str(study.output)]
    success = study.task("summarize", lambda: command,
        [study.output / file for file in ("report.md", "aggregate.json", "bootstrap.json")])
    study.state.update(status="complete" if success else "failed", current=None, finished_at=now())
    study.save()
    print("{}: {}".format("ALL COMPLETE" if success else "FAILED", study.output / "report.md"), flush=True)
    return 0 if success else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path.home() / "workspace/outputs/chexpert"
    parser.add_argument("--source-dir", type=Path, default=base / "tiny_study_seed0")
    parser.add_argument("--output-dir", type=Path, default=base / "tiny_study_lora_mixup_ignore")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source, output = args.source_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    try:
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Source and follow-up output must be separate directories")
        original = read_json(source / "study.json")
        spec = followup_spec(source, output, original)
        if args.dry_run:
            print(json.dumps(dict(method=METHOD, imported_conditions=len(original["config"]["cases"]),
                                  additional_training_runs=1, selection_mode=spec["selection_mode"]), indent=2))
            return 0
        # Hold the source lock throughout import/training: an accidental resume
        # cannot launch the old driver on the same GPUs during this follow-up.
        with (source / ".lock").open("a") as source_lock:
            try:
                fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Source study is still running. Stop that driver before switching.")
            output.mkdir(parents=True, exist_ok=True)
            with (output / ".lock").open("a") as output_lock:
                try:
                    fcntl.flock(output_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError("Follow-up study is already running")
                manifest = output / "study.json"
                if manifest.exists():
                    if read_json(manifest) != spec:
                        raise ValueError("Follow-up code/settings changed; restore the original code to resume")
                elif any(p.name != ".lock" for p in output.iterdir()):
                    raise ValueError("Choose a fresh follow-up output directory")
                else:
                    atomic_json(manifest, spec)
                study = Study(output, spec)
                try:
                    study.state["current"] = "import_results"
                    study.state["tasks"]["import_results"] = dict(status="running", started_at=now())
                    study.event("IMPORT: validating completed original conditions")
                    study.save()
                    def progress(message):
                        study.state["tasks"]["import_results"]["activity"] = message
                        print(message, flush=True)
                        study.save()
                    inventory = import_inventory(source, original, progress)
                    imports = output / "imported_results.json"
                    if imports.exists() and read_json(imports) != inventory:
                        raise ValueError("Source results changed after the first import")
                    atomic_json(imports, inventory)
                    copy_inventory(source, output, inventory, progress)
                    selection = dict(selected={"C": "c_lora", "B": "b_ignore", "A": "a_mixup"}, method=METHOD,
                        selection_mode="post_hoc_exploratory", criterion="User-specified after reviewing interim test results",
                        test_used=True, automatic_dev_choice=read_json(source / "combination_selection.json")
                        if (source / "combination_selection.json").exists() else None)
                    path = output / "combination_selection.json"
                    if path.exists() and read_json(path) != selection:
                        raise ValueError("Requested combination or selection provenance changed")
                    atomic_json(path, selection)
                    study.state["tasks"]["import_results"].update(status="complete", finished_at=now())
                    study.state["current"] = None
                    return run_followup(study)
                except BaseException as error:
                    study.state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
                    active = study.state.get("current")
                    if active and study.state["tasks"][active]["status"] == "running":
                        study.state["tasks"][active].update(status=study.state["status"], error=str(error))
                    study.save()
                    raise
    except KeyboardInterrupt:
        print("Interrupted. Resume with the same follow-up command.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print("Follow-up stopped: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    def stop_requested(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGHUP, stop_requested)
    sys.exit(main())
