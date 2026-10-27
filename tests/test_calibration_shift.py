"""Post-hoc calibrator, prior-shift, seed aggregation and seed-replication setup checks (CPU, synthetic)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_calibration_shift as shift
from calibration_postprocess import binary_nll, fit_temperatures, postprocess
from prepare_chexpert_splits import CLASS_NAMES, sha256
from study_io import atomic_json

TEST_PREVALENCE = np.array([.35, .33, .16, .21, .32])  # Expert valid, shifted from the report labels below.
REPORT_PREVALENCE = np.array([.31, .12, .07, .32, .40])


def write_export(folder, split, logits, labels, mask, patients, checkpoint):
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "logits.npy", logits.astype(np.float32))
    np.save(folder / "y_pred.npy", expit(logits.astype(np.float32)))
    np.save(folder / "y_gt.npy", labels.astype(np.float32))
    np.save(folder / "label_mask.npy", mask.astype(np.float32))
    np.save(folder / "indices.npy", np.arange(len(labels)))
    atomic_json(folder / "samples.json", [dict(path="{}/patient{:05d}/view{}.jpg".format(split, p, i), patient_id="patient{:05d}".format(p))
                                          for i, p in enumerate(patients)])
    atomic_json(folder / "evaluation.json", dict(split=split, checkpoint_sha256=checkpoint))


def make_study(root, runs, n_cal=600, n_test=200):
    """runs: {(case, seed): logit noise SD} on a shared split, as in the real study.

    Model scores are calibrated to the report-labeled prevalence; the expert test set has the same
    scores but a different base rate (pure label shift), plus run-specific noise.
    """
    root.mkdir(parents=True)
    atomic_json(root / "study.json", dict(config=dict(expert_calibration_fraction=.5, split_seed=42)))
    (root / "splits").mkdir()
    with (root / "splits/train.csv").open("w") as handle:
        handle.write("Path,Frontal/Lateral," + ",".join(CLASS_NAMES) + "\n")
        for i, row in enumerate([["1", "", "-1", "1.0", "0"], ["", "1", "", "", "1"], ["0", "", "", "-1", ""]]):
            handle.write("train/patient{:05d}/study1/view1_frontal.jpg,Frontal,{}\n".format(i, ",".join(row)))
    data_rng = np.random.RandomState(0)
    label_shift = logit(TEST_PREVALENCE) - logit(REPORT_PREVALENCE)
    cal_scores = logit(REPORT_PREVALENCE) + 1.5 * data_rng.randn(n_cal, 5)
    cal_labels = (data_rng.rand(n_cal, 5) < expit(cal_scores)).astype(np.float64)
    cal_mask = data_rng.rand(n_cal, 5) > .1
    test_scores = logit(REPORT_PREVALENCE) + 1.5 * data_rng.randn(n_test, 5)
    test_labels = (data_rng.rand(n_test, 5) < expit(test_scores + label_shift)).astype(np.float64)
    test_patients = np.repeat(np.arange(n_test // 2), 2)[:n_test] + 50000
    for (case, seed), noise in runs.items():
        rng = np.random.RandomState(100 * seed + len(case))
        folder = root / "runs" / case / ("seed" + str(seed))
        checkpoint = "{}-{}".format(case, seed)
        write_export(folder / "calibration", "calibration", cal_scores + noise * rng.randn(n_cal, 5), cal_labels,
                     cal_mask, np.arange(n_cal), checkpoint)
        write_export(folder / "test", "test", test_scores + noise * rng.randn(n_test, 5), test_labels,
                     np.ones_like(test_labels), test_patients, checkpoint)
        postprocess(folder / "calibration", folder / "test", folder / "processed", plots=False)
    return root


class CalibratorTests(unittest.TestCase):
    def test_bias_recovers_intercept_shift_that_ts_cannot(self):
        rng = np.random.RandomState(1)
        z = rng.randn(20000) * 2
        y = (rng.rand(20000) < expit(z + 1.0)).astype(float)
        slope, intercept = shift.fit_bias(z, y)
        self.assertEqual(slope, 1.0)
        self.assertAlmostEqual(intercept, 1.0, delta=.06)
        t, _ = fit_temperatures(z[:, None], y[:, None], np.ones((len(y), 1), bool))
        nll_ts = binary_nll(z / t[0], y).mean()
        self.assertLess(binary_nll(z + intercept, y).mean(), nll_ts - .01)

    def test_platt_recovers_slope_and_intercept(self):
        rng = np.random.RandomState(2)
        z = rng.randn(30000) * 2
        y = (rng.rand(30000) < expit(.5 * z - .7)).astype(float)
        slope, intercept = shift.fit_platt(z, y)
        self.assertAlmostEqual(slope, .5, delta=.03)
        self.assertAlmostEqual(intercept, -.7, delta=.05)

    def test_fitted_calibrators_never_worse_than_identity_on_fit_data(self):
        rng = np.random.RandomState(3)
        z = rng.randn(400, 5)
        y = (rng.rand(400, 5) < .3).astype(float)
        mask = rng.rand(400, 5) > .2
        params = shift.fit_calibrators(z, y, mask, np.full(5, .3))
        for name in ("ts", "bias", "platt"):
            slope, intercept = params[name]
            for c in range(5):
                m = mask[:, c]
                self.assertLessEqual(binary_nll(z[m, c] * slope[c] + intercept[c], y[m, c]).mean(),
                                     binary_nll(z[m, c], y[m, c]).mean() + 1e-12)

    def test_prior_shift_is_label_free_log_odds_difference(self):
        y = np.array([[1, 0], [0, 0], [0, 1], [0, 0]], dtype=float)
        params = shift.fit_calibrators(np.zeros((4, 2)), y, np.ones((4, 2), bool), np.array([.5, .5]))
        np.testing.assert_allclose(params["prior"][1], logit(np.array([.25, .25])))
        np.testing.assert_allclose(params["prior"][0], 1.0)

    def test_constant_class_falls_back_to_identity(self):
        z = np.random.RandomState(4).randn(50, 5)
        y = np.zeros((50, 5))
        y[:25, 1:] = 1
        params = shift.fit_calibrators(z, y, np.ones((50, 5), bool), np.full(5, .3))
        for name in ("bias", "platt"):
            self.assertEqual((params[name][0][0], params[name][1][0]), (1.0, 0.0))


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_end_to_end_reproduces_ts_and_aggregates_seeds(self):
        runs = {("baseline", 0): .6, ("a_mixup", 0): .3, ("c_head", 0): 1.2,
                ("baseline", 1): .6, ("a_mixup", 1): .3}
        study = make_study(self.root / "study", runs)
        analysis = shift.analyze([study], self.root / "out", replicates=30, workers=1)
        self.assertEqual(len(analysis["runs"]), 5)
        self.assertTrue(all(p["ts_reproduced"] for r in analysis["runs"] for p in r["protocols"].values()))
        # Prior correction only exists where the fitting labels differ from the report-labeled source.
        for r in analysis["runs"]:
            self.assertNotIn("prior", r["protocols"]["report_holdout"]["calibrators"])
            self.assertIn("prior", r["protocols"]["expert_subset"]["calibrators"])
        # Under a pure prevalence shift, an intercept fitted on expert labels beats slope-only TS.
        expert = analysis["calibrator_vs_ts"]["expert_subset"]["methods"]
        self.assertLess(expert["bias"]["nll"]["mean"], 0)
        self.assertLess(expert["platt"]["nll"]["mean"], 0)
        self.assertAlmostEqual(expert["bias"]["auroc"]["mean"], 0, places=12)  # Monotone maps keep ranking.
        rows = [r for r in analysis["seed_replication"] if r["case"] == "a_mixup" and r["calibrator"] == "ts"]
        self.assertEqual({r["protocol"] for r in rows}, set(shift.PROTOCOLS))
        self.assertTrue(all(r["seeds"] == [0, 1] and r["delta"]["auroc"]["mean"] > 0 for r in rows))
        self.assertFalse(any(r["case"] == "c_head" for r in analysis["seed_replication"]))  # Single seed only.
        prevalence = analysis["prevalence"]["train"]
        self.assertAlmostEqual(prevalence["Atelectasis"]["prevalence"], 1 / 3)  # "1" vs blank and "0".
        self.assertAlmostEqual(prevalence["Consolidation"]["uncertain_rate"], 1 / 3)
        text = (self.root / "out/analysis.json").read_text() + (self.root / "out/report.md").read_text()
        self.assertNotIn("patient5", text)  # Aggregates only: no patient identifiers or image paths.
        self.assertIn("## 5. Seed 반복", text)

    def test_imported_duplicate_run_is_counted_once(self):
        runs = {("baseline", 0): .6, ("a_mixup", 0): .3}
        first = make_study(self.root / "a", runs)
        second = self.root / "b"
        second.mkdir()
        import shutil
        shutil.copytree(first / "runs", second / "runs")
        shutil.copytree(first / "splits", second / "splits")
        atomic_json(second / "study.json", json.loads((first / "study.json").read_text()))
        self.assertEqual(len(shift.discover_runs([first, second])), 2)

    def test_changed_predictions_are_rejected(self):
        study = make_study(self.root / "study", {("baseline", 0): .6})
        path = study / "runs/baseline/seed0/test/logits.npy"
        np.save(path, np.load(path) + 1)
        with self.assertRaises(ValueError):
            shift.discover_runs([study])


class SeedReplicationSpecTests(unittest.TestCase):
    def setUp(self):
        import run_seed_replication as replication
        from run_calibration_study import CODE, REPO
        self.replication, self.repo = replication, REPO
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        weights = root / "tiny.pt"
        weights.write_bytes(b"weights")
        cases = [dict(id="baseline", axis="baseline", method=dict(loss="bce", scope="full", uncertain="original")),
                 dict(id="a_mixup", axis="A", method=dict(loss="mixup", mixup_alpha=.2))]
        config = json.loads((REPO / "configs/tiny_study_day.json").read_text())
        config["cases"] = cases
        self.source = root / "source"
        self.source.mkdir()
        self.original = dict(config=config, data_root=str(root / "data"), pretrained=str(weights),
                             pretrained_sha256=sha256(weights), code_sha256={n: sha256(REPO / "scripts" / n) for n in CODE},
                             architecture_sha256=sha256(REPO / "EVA-X/classification/models/models_eva.py"))
        atomic_json(self.source / "study.json", self.original)

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self, cases=("baseline", "a_mixup"), seeds=(1, 2)):
        return self.replication.replication_spec(self.source, self.source.parent / "out", self.original, list(cases), list(seeds))

    def test_spec_keeps_everything_but_the_seed(self):
        spec = self.spec(cases=("a_mixup",))
        self.assertEqual(spec["config"]["training"], self.original["config"]["training"])
        self.assertEqual([c["id"] for c in spec["config"]["cases"]], ["a_mixup"])
        self.assertEqual(spec["pretrained_sha256"], self.original["pretrained_sha256"])
        self.assertIn("run_seed_replication.py", spec["code_sha256"])
        study = self.replication.SeedStudy(self.source.parent / "out", spec)
        self.assertEqual(sorted(k for k in study.state["tasks"] if k.endswith("_train")),
                         ["a_mixup_seed1_train", "a_mixup_seed2_train"])

    def test_rejects_seed_zero_unknown_cases_and_changed_inputs(self):
        with self.assertRaises(ValueError):
            self.spec(seeds=(0, 1))
        with self.assertRaises(ValueError):
            self.spec(cases=("baseline", "nonexistent"))
        Path(self.original["pretrained"]).write_bytes(b"other weights")
        with self.assertRaises(ValueError):
            self.spec()

    def test_rejects_numerical_code_drift(self):
        self.original["code_sha256"]["calibration_methods.py"] = "0" * 64
        with self.assertRaises(ValueError):
            self.spec()


if __name__ == "__main__":
    unittest.main()
