"""Run with: python -m unittest discover -s tests -v"""

import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_calibration.py"
SPEC = importlib.util.spec_from_file_location("evaluate_calibration", str(SCRIPT))
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


def five_columns(values):
    return np.tile(np.asarray(values)[:, None], (1, 5))


class CalibrationMetricsTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        labels = five_columns([0, 0, 1, 1])
        probabilities = five_columns([0.1, 0.4, 0.6, 0.9])
        report = calibration.evaluate_predictions(labels, probabilities, [2])
        for row in report["per_class"]:
            self.assertEqual(row["auroc"], 1.0)
            self.assertAlmostEqual(row["brier"], 0.085)
            self.assertAlmostEqual(row["nll"], -(math.log(0.9) + math.log(0.6)) / 2)
            self.assertAlmostEqual(row["calibration"]["2"]["uniform"]["ece"], 0.25)
            self.assertAlmostEqual(row["calibration"]["2"]["adaptive"]["ece"], 0.25)
        self.assertEqual(report["macro"]["auroc"], 1.0)
        self.assertEqual(report["macro"]["auroc_valid_classes"], 5)

    def test_curves_and_scores_match_sklearn(self):
        rng = np.random.RandomState(17)
        # Even N keeps sample values off the interior quantile boundaries;
        # historical sklearn releases use different boundary inclusion rules.
        labels = rng.randint(0, 2, (140, 5))
        probabilities = rng.uniform(0.001, 0.999, (140, 5))
        report = calibration.evaluate_predictions(labels, probabilities, [5, 10])
        for column, row in enumerate(report["per_class"]):
            self.assertAlmostEqual(row["brier"], brier_score_loss(labels[:, column], probabilities[:, column]))
            self.assertAlmostEqual(row["nll"], log_loss(labels[:, column], probabilities[:, column], labels=[0, 1]))
            for n in [5, 10]:
                for ours, reference in [("uniform", "uniform"), ("adaptive", "quantile")]:
                    observed, predicted = calibration_curve(labels[:, column], probabilities[:, column],
                                                             n_bins=n, strategy=reference)
                    bins = [b for b in row["calibration"][str(n)][ours]["bins"] if b["count"]]
                    np.testing.assert_allclose([b["positive_fraction"] for b in bins], observed)
                    np.testing.assert_allclose([b["mean_probability"] for b in bins], predicted)

    def test_boundaries_empty_bins_and_nll_saturation(self):
        labels = five_columns([1, 0, 1, 0, 1])
        probabilities = five_columns([0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float16)
        report = calibration.evaluate_predictions(labels, probabilities, [4], eps=1e-7)
        stats = report["per_class"][0]["calibration"]["4"]["uniform"]
        self.assertEqual([b["count"] for b in stats["bins"]], [2, 1, 1, 1])
        self.assertTrue(math.isfinite(report["macro"]["nll"]))
        endpoints = calibration.evaluate_predictions(five_columns([0, 1]), five_columns([0, 1]), [10])
        bins = endpoints["per_class"][0]["calibration"]["10"]["uniform"]["bins"]
        self.assertEqual(sum(b["count"] == 0 for b in bins), 8)
        self.assertEqual(endpoints["macro"]["calibration"]["10"]["ece"], 0.0)
        self.assertTrue(all(b["positive_fraction"] is None for b in bins if b["count"] == 0))
        json.dumps(endpoints, allow_nan=False)

    def test_adaptive_bins_do_not_split_ties_or_depend_on_row_order(self):
        labels = np.array([0, 0, 1, 1, 1, 0])
        probabilities = np.array([0.2, 0.2, 0.2, 0.8, 0.8, 0.8])
        first = calibration.calibration_bins(labels, probabilities, 10, "adaptive")
        reverse = calibration.calibration_bins(labels[::-1], probabilities[::-1], 10, "adaptive")
        self.assertEqual(first, reverse)
        self.assertEqual(first["occupied_bins"], 2)
        for probability in [0.0, 0.5, 1.0]:
            stats = calibration.calibration_bins(labels, np.full(6, probability), 10, "adaptive")
            self.assertEqual(stats["occupied_bins"], 1)
            self.assertAlmostEqual(stats["ece"], abs(probability - labels.mean()))

    def test_single_class_auroc_is_null_and_not_zero(self):
        labels = five_columns([0, 0, 0, 0])
        labels[:, 0] = [0, 0, 1, 1]
        report = calibration.evaluate_predictions(labels, five_columns([0.1, 0.4, 0.6, 0.9]))
        self.assertIsNone(report["per_class"][1]["auroc"])
        self.assertEqual(report["macro"]["auroc"], 1.0)
        self.assertEqual(report["macro"]["auroc_valid_classes"], 1)
        labels[:, 0] = 0
        self.assertIsNone(calibration.evaluate_predictions(labels, np.zeros_like(labels))["macro"]["auroc"])

    def test_invalid_inputs_are_rejected(self):
        labels, probabilities = five_columns([0, 1]), five_columns([0.2, 0.8])
        cases = [
            (labels[:1], probabilities),
            (labels[:, :4], probabilities[:, :4]),
            (labels[:0], probabilities[:0]),
            (labels.ravel(), probabilities.ravel()),
            (labels - 1, probabilities),
            (labels.astype(float) + 0.1, probabilities),
            (labels, probabilities + 1.0),
            (labels, np.full((2, 5), np.nan)),
            (labels, np.full((2, 5), np.inf)),
            (labels, probabilities.astype(complex)),
        ]
        for y_true, y_prob in cases:
            with self.subTest(shape=y_true.shape, dtype=y_prob.dtype):
                with self.assertRaises(ValueError):
                    calibration.evaluate_predictions(y_true, y_prob)
        for bins in [[], [0], [-1], [1.5]]:
            with self.assertRaises(ValueError):
                calibration.evaluate_predictions(labels, probabilities, bins)
        for eps in [0, 0.5, float("nan")]:
            with self.assertRaises(ValueError):
                calibration.evaluate_predictions(labels, probabilities, eps=eps)


class CalibrationCLITests(unittest.TestCase):
    def test_real_npy_to_json_csv_and_png(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rng = np.random.RandomState(20261023)
            labels = rng.randint(0, 2, (202, 5)).astype(np.float32)
            probabilities = rng.uniform(0, 1, (202, 5)).astype(np.float16)
            np.save(str(folder / "y_gt.npy"), labels)
            np.save(str(folder / "y_pred.npy"), probabilities)
            original_labels = (folder / "y_gt.npy").read_bytes()
            original_probabilities = (folder / "y_pred.npy").read_bytes()
            result = subprocess.run([sys.executable, str(SCRIPT), "--eval-dir", tmp],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("macro", result.stdout)
            output = folder / "calibration"
            report = json.loads((output / "metrics.json").read_text())
            self.assertEqual(report["n_samples"], 202)
            self.assertEqual(report["class_order"], list(calibration.CLASS_NAMES))
            self.assertEqual(report["inputs"]["y_pred.npy"]["dtype"], "float16")
            with (output / "metrics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertEqual(len([r for r in rows if r["class"] == "macro"]), 2)
            with (output / "bin_stats.csv").open() as handle:
                bin_rows = list(csv.DictReader(handle))
            for name in calibration.CLASS_NAMES:
                for strategy in ["uniform", "adaptive"]:
                    for n in [5, 10]:
                        count = sum(int(r["count"]) for r in bin_rows
                                    if r["class"] == name and r["strategy"] == strategy and int(r["requested_bins"]) == n)
                        self.assertEqual(count, 202)
            images = list(output.glob("*.png"))
            self.assertEqual(len(images), 4)
            for path in images:
                with Image.open(str(path)) as image:
                    image.verify()
            self.assertEqual((folder / "y_gt.npy").read_bytes(), original_labels)
            self.assertEqual((folder / "y_pred.npy").read_bytes(), original_probabilities)

    def test_cli_returns_failure_for_missing_or_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for save_bad_input in [False, True]:
                if save_bad_input:
                    np.save(str(folder / "y_gt.npy"), five_columns([0, 1]))
                    np.save(str(folder / "y_pred.npy"), five_columns([float("nan"), 0.5]))
                result = subprocess.run([sys.executable, str(SCRIPT), "--eval-dir", tmp, "--no-plots"],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("error:", result.stderr)
                self.assertFalse((folder / "calibration").exists())

    def test_no_plots_custom_output_and_undefined_auroc(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            output = folder / "custom-report"
            np.save(str(folder / "y_gt.npy"), np.zeros((3, 5)))
            np.save(str(folder / "y_pred.npy"), np.full((3, 5), 0.5))
            result = subprocess.run([sys.executable, str(SCRIPT), "--eval-dir", tmp,
                                     "--output-dir", str(output), "--bins", "5", "--no-plots"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("N/A", result.stdout)
            report = json.loads((output / "metrics.json").read_text())
            self.assertIsNone(report["macro"]["auroc"])
            self.assertEqual(report["macro"]["auroc_valid_classes"], 0)
            self.assertEqual(list(output.glob("*.png")), [])


class EvaluationLauncherTests(unittest.TestCase):
    def test_empty_checkpoint_is_rejected_and_worker_failure_is_propagated(self):
        launcher = SCRIPT.parent / "run_chexpert.sh"
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            checkpoint = folder / "eva_x_tiny_patch16_mrerged520k_mim_chexpert_ft.pth"
            checkpoint.write_bytes(b"")
            env = dict(os.environ, WEIGHTS_DIR=tmp, OUT_ROOT=str(folder / "outputs"))
            result = subprocess.run(["bash", str(launcher), "eval", "tiny"], env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("0바이트", result.stdout)
            self.assertFalse((folder / "outputs").exists())
            checkpoint.write_bytes(b"stub checkpoint, never loaded by the mock worker")
            bin_dir = folder / "bin"
            bin_dir.mkdir()
            worker = bin_dir / "torchrun"
            worker.write_text("#!/usr/bin/env bash\necho 'intentional mock worker failure'\nexit 23\n")
            worker.chmod(0o755)
            env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
            result = subprocess.run(["bash", str(launcher), "eval", "tiny"], env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            self.assertEqual(result.returncode, 23, result.stderr)
            self.assertIn("intentional mock worker failure", result.stdout)
            logs = list((folder / "outputs").glob("chexpert/*/run.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn("intentional mock worker failure", logs[0].read_text())


if __name__ == "__main__":
    unittest.main()
