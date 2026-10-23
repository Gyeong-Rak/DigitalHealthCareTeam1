"""Synthetic tests for the full experiment plan and paired report generation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_calibration_runs as compare
import evaluate_calibration as calibration
import run_tiny_comparison as pipeline


def write_evaluation(folder, loss, change_labels=False):
    folder.mkdir(parents=True)
    labels = np.tile(np.array([0, 0, 1, 1], dtype=np.float32)[:, None], (1, 5))
    if change_labels:
        labels[0, 0] = 1
    probabilities = np.tile(np.array([.1, .4, .6, .9], dtype=np.float32)[:, None], (1, 5))
    if loss == "label_smoothing":
        probabilities = .95 * probabilities + .025
    np.save(folder / "y_gt.npy", labels)
    np.save(folder / "y_pred.npy", probabilities)
    report = calibration.evaluate_predictions(labels, probabilities)
    report["inputs"] = {name: {"sha256": compare.sha256(folder / name)} for name in ("y_gt.npy", "y_pred.npy")}
    (folder / "calibration").mkdir()
    calibration.write_tables(report, folder / "calibration")
    config = {"model": "tiny", "loss": loss, "smoothing": .05 if loss == "label_smoothing" else 0.,
              "seed": 0, "pretrained_sha256": "same", "split_manifest_sha256": "same"}
    (folder / "evaluation.json").write_text(json.dumps({"split": "test", "checkpoint_epoch": 2, "training_config": config}))
    (folder / "samples.json").write_text(json.dumps([
        {"path": "synthetic/{}".format(index), "patient_id": "patient{}".format(index // 2)} for index in range(4)]))


class ComparisonTests(unittest.TestCase):
    def test_paired_summary_files_and_arithmetic(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_evaluation(root / "baseline", "bce")
            write_evaluation(root / "candidate", "label_smoothing")
            report = compare.compare_runs(root / "baseline", root / "candidate", root / "comparison")
            self.assertEqual(report["n_patients"], 2)
            self.assertEqual(report["candidate_minus_baseline"][-1]["auroc"], 0)
            macro = [row for row in report["rows"] if row["class"] == "macro"]
            self.assertAlmostEqual(report["candidate_minus_baseline"][-1]["brier"], macro[1]["brier"] - macro[0]["brier"])
            for name in ["report.md", "comparison.csv", "deltas.csv", "comparison.json", "comparison.png"]:
                self.assertTrue((root / "comparison" / name).is_file())
            with Image.open(root / "comparison/comparison.png") as image:
                image.verify()

    def test_changed_predictions_labels_and_training_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_evaluation(root / "a", "bce")
            write_evaluation(root / "b", "label_smoothing", change_labels=True)
            with self.assertRaises(ValueError):
                compare.compare_runs(root / "a", root / "b", root / "out")
            write_evaluation(root / "c", "label_smoothing")
            meta = root / "c/evaluation.json"
            data = json.loads(meta.read_text())
            data["training_config"]["seed"] = 3
            meta.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                compare.compare_runs(root / "a", root / "c", root / "out")
            write_evaluation(root / "d", "label_smoothing")
            np.save(root / "d/y_pred.npy", np.ones((4, 5)))
            with self.assertRaises(ValueError):
                compare.compare_runs(root / "a", root / "d", root / "out")

    def test_driver_full_plan_resume_and_stop_on_failure_without_real_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "experiment"
            calls = []

            def fake_step(step, env, folder):
                stage = step["stage"]
                calls.append(stage)
                if stage.startswith("train_"):
                    case = Path(env["RUN_DIR"])
                    case.mkdir(parents=True, exist_ok=True)
                    (case / "best.pth").write_bytes(b"synthetic model artifact")
                elif stage.startswith("evaluate_"):
                    write_evaluation(Path(env["RESULT_DIR"]), stage.split("_", 1)[1])
                elif stage == "summarize":
                    compare.compare_runs(folder / "bce/test", folder / "label_smoothing/test", folder / "comparison")

            with patch.object(pipeline, "run_step", side_effect=fake_step):
                pipeline.run_comparison(output)
                self.assertEqual(calls, ["prepare", "check_bce", "check_label_smoothing", "train_bce", "evaluate_bce",
                                         "train_label_smoothing", "evaluate_label_smoothing", "summarize"])
                self.assertTrue((output / "complete.json").is_file())
                calls.clear()
                pipeline.run_comparison(output)
                self.assertEqual(calls, ["prepare", "check_bce", "check_label_smoothing", "summarize"])
                with patch.dict(os.environ, {"EPOCHS": "99"}):
                    with self.assertRaises(ValueError):
                        pipeline.run_comparison(output)
            def fail(step, env, folder):
                raise subprocess.CalledProcessError(17, step["command"])
            with patch.object(pipeline, "run_step", side_effect=fail):
                with self.assertRaises(subprocess.CalledProcessError):
                    pipeline.run_comparison(root / "failed")
            self.assertFalse((root / "failed/complete.json").exists())

    def test_dry_run_never_launches_training_or_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dry-run"
            with patch.object(pipeline, "run_step") as execute:
                pipeline.run_comparison(root, dry_run=True)
                execute.assert_not_called()
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
