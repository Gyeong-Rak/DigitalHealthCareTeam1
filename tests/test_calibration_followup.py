"""Import without retraining, explicit exploratory provenance, targeted stop."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from test_calibration_study import export_fixture, torch
from test_tiny_training import write_fixture, splits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_calibration_followup as followup
from run_calibration_study import REPO, CODE
from study_io import atomic_json, read_json
from prepare_chexpert_splits import sha256
from stop_calibration_study import stop_study, process_matches

if torch is not None:
    from calibration_methods import method_config, configure_scope, training_mode, mix_batch, objective
    from train_tiny_calibration import build_model, seed_everything
    from calibration_postprocess import postprocess
    from study_report import build_report


def make_source(root):
    root.mkdir()
    (root / ".lock").touch()
    data = root / "data"
    data.mkdir()
    write_fixture(data, create_images=True)
    splits.prepare_splits(data, root / "splits")
    config = read_json(REPO / "configs/tiny_study_day.json")
    config["cases"] = [c for c in config["cases"] if c["id"] in ("baseline", "c_lora", "b_ignore", "a_mixup")]
    config["plots"], config["bootstrap_replicates"] = False, 2
    weights = root / "mim.pt"
    weights.write_bytes(b"mock weights; numerical training is not run in orchestration tests")
    original = dict(config=config, pretrained=str(weights), pretrained_sha256=sha256(weights), data_root=str(data),
        architecture_sha256=sha256(REPO / "EVA-X/classification/models/models_eva.py"),
        code_sha256={name: sha256(REPO / "scripts" / name) for name in CODE})
    atomic_json(root / "study.json", original)
    for case in config["cases"]:
        run = root / "runs" / case["id"] / "seed0"
        train = run / "train"
        train.mkdir(parents=True)
        (train / "best.pth").write_bytes(case["id"].encode())
        (train / "history.jsonl").write_text('{"epoch":2}\n')
        cfg = dict(method=method_config(case["method"]), split_manifest_sha256=sha256(root / "splits/manifest.json"))
        atomic_json(train / "config.json", cfg)
        for split, offset in (("dev", 0), ("calibration", 100), ("test", 200)):
            export_fixture(run / split, split, checkpoint=sha256(train / "best.pth"), offset=offset)
            record = read_json(run / split / "evaluation.json")
            record["training_config"].update(cfg)
            atomic_json(run / split / "evaluation.json", record)
        processed = run / "processed"
        postprocess(run / "calibration", run / "test", processed, plots=False)
        atomic_json(root / "completed" / (case["id"] + "_seed0_postprocess.json"),
                    {"artifacts": {str(p.relative_to(root)): sha256(p) for p in processed.rglob("*") if p.is_file()}})
        atomic_json(root / "completed" / (case["id"] + "_seed0_train.json"),
                    {"artifacts": {str(p.relative_to(root)): sha256(p) for p in train.iterdir()}})
    atomic_json(root / "combination_selection.json", dict(selected=dict(C="baseline", B="b_ignore", A="baseline"),
                                                       method=dict(loss="bce", scope="full", uncertain="ignore"), test_used=False))
    return original


@unittest.skipIf(torch is None, "Requires numerical training environment")
class FollowupTests(unittest.TestCase):
    def test_import_integrity_no_checkpoint_copy_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "output"
            original = make_source(source)
            inventory = followup.import_inventory(source, original)
            self.assertFalse(any(Path(p).suffix in (".pth", ".pt") for p in inventory))
            self.assertFalse(any("/data/" in p for p in inventory))
            followup.copy_inventory(source, output, inventory)
            followup.copy_inventory(source, output, inventory)
            self.assertTrue(all(sha256(source / p) == sha256(output / p) == digest for p, digest in inventory.items()))
            target = "runs/baseline/seed0/train/history.jsonl"
            (output / target).write_text("changed")
            with self.assertRaises(ValueError):
                followup.copy_inventory(source, output, inventory)
            (source / "runs/baseline/seed0/test/logits.npy").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                followup.import_inventory(source, original)

    def test_only_requested_combination_runs_and_resume_skips_completed_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "output"
            make_source(source)
            # Keep generated public fixture Markdown in temp, not the repository.
            original_spec = followup.followup_spec
            def spec(*args):
                result = original_spec(*args)
                result["docs_dir"] = str(Path(tmp) / "docs")
                return result
            calls = []
            def stream(task, command):
                calls.append(task)
                def value(flag):
                    return command[command.index(flag) + 1]
                self.assertTrue(task.startswith(("check_combination", "combination_seed0_")) or task == "summarize")
                if task == "check_combination":
                    self.assertEqual(read_json(value("--experiment-config")), followup.METHOD)
                    atomic_json(value("--check-receipt"), dict(checked=True))
                elif task == "combination_seed0_train":
                    train = Path(value("--output-dir"))
                    train.mkdir(parents=True)
                    for name in ("best.pth", "last.pth"):
                        (train / name).write_bytes(b"mock checkpoint")
                    atomic_json(train / "config.json", dict(method=method_config(read_json(value("--experiment-config")))))
                    (train / "history.jsonl").write_text('{"epoch":2}\n')
                elif task.endswith(("_dev", "_calibration", "_test")):
                    split, dest = value("--eval-split"), Path(value("--output-dir"))
                    export_fixture(dest, split, checkpoint=sha256(dest.parent / "train/best.pth"),
                                   offset=dict(dev=0, calibration=100, test=200)[split])
                    record = read_json(dest / "evaluation.json")
                    record["training_config"].update(read_json(dest.parent / "train/config.json"))
                    atomic_json(dest / "evaluation.json", record)
                elif task.endswith("_postprocess"):
                    postprocess(Path(value("--calibration-dir")), Path(value("--test-dir")), Path(value("--output-dir")), plots=False)
                elif task == "summarize":
                    build_report(output)
                return 0
            argv = ["followup", "--source-dir", str(source), "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), patch.object(followup, "followup_spec", side_effect=spec), \
                    patch.object(followup.Study, "stream", side_effect=stream):
                self.assertEqual(followup.main(), 0)
                n = len(calls)
                self.assertEqual(followup.main(), 0)
                self.assertEqual(len(calls), n)
            self.assertEqual(sum(c == "combination_seed0_train" for c in calls), 1)
            self.assertEqual(read_json(output / "aggregate.json")["completed_conditions"], 5)
            selection = read_json(output / "combination_selection.json")
            self.assertEqual(selection["method"]["uncertain"], "ignore")
            self.assertTrue(selection["test_used"])
            self.assertEqual(selection["automatic_dev_choice"]["selected"]["C"], "baseline")
            report = (Path(tmp) / "docs/combination_seed0.md").read_text()
            self.assertIn("사후 탐색", report)
            self.assertNotIn("test 결과로 체크포인트, 조합, 온도 또는 hyperparameter를 선택하지 않았습니다", report)
            self.assertEqual(read_json(output / "status.json")["status"], "complete")

    def test_joint_lora_mixup_ignore_real_architecture_backward(self):
        torch.set_num_threads(1)
        seed_everything(0)
        model = build_model(torch.device("cpu"))
        cfg = method_config(followup.METHOD)
        configure_scope(model, cfg)
        training_mode(model, cfg)
        images, targets, mask = torch.randn(2, 3, 224, 224), torch.rand(2, 5), torch.ones(2, 5)
        mask[0, :3] = 0
        images, targets, mass = mix_batch(images, targets, mask, cfg["mixup_alpha"])
        loss = objective(model(images), targets, mass, cfg)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)


class TargetedStopTests(unittest.TestCase):
    def test_free_lock_does_not_signal_a_stale_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".lock").touch()
            atomic_json(root / "status.json", dict(pid=os.getpid(), status="complete"))
            with patch("stop_calibration_study.os.kill") as kill:
                stop_study(root)
                kill.assert_not_called()

    def test_only_matching_combination_driver_is_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "run_calibration_study.py"
            script.write_text('''import fcntl,json,os,signal,sys,time
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--output-dir')+1])
with (root/'.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 (root/'status.json').write_text(json.dumps(dict(pid=os.getpid(),current='combination_seed0_train',tasks={})))
 signal.signal(signal.SIGTERM,lambda *_:sys.exit(0))
 while True: time.sleep(1)
''')
            process = subprocess.Popen([sys.executable, str(script), "--output-dir", str(root)])
            try:
                for _ in range(100):
                    if (root / "status.json").exists():
                        break
                    time.sleep(.02)
                self.assertTrue(process_matches(process.pid, root))
                self.assertFalse(process_matches(process.pid, root / "other"))
                stop_study(root)
                self.assertEqual(process.wait(timeout=3), 0)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=3)

    def test_unrelated_pid_is_never_signaled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / ".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                atomic_json(root / "status.json", dict(pid=os.getpid(), current="combination_seed0_train"))
                with patch("stop_calibration_study.os.kill") as kill:
                    with self.assertRaises(ValueError):
                        stop_study(root)
                    kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
