"""Numerical, leakage, real-architecture adapter, reporting and recovery checks."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from study_io import atomic_json, read_json
from prepare_chexpert_splits import sha256
from run_calibration_study import Study, choose_combination, validate_config, REPO
from test_tiny_training import write_fixture, splits, torch, training

if torch is not None:
    import numpy as np
    from scipy.special import expit
    import calibration_methods as methods
    from calibration_postprocess import fit_temperatures, postprocess, binary_nll
    from study_report import build_report, patient_bootstrap, render_report, write_managed


def export_fixture(folder, split, checkpoint="same", patients=12, offset=0):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(11)
    labels = rng.randint(2, size=(patients * 2, 5)).astype(np.float32)
    # Guarantee each small fixture has both labels in each disease.
    labels[0], labels[1] = 0, 1
    logits = ((labels * 2 - 1) * .8 + rng.randn(*labels.shape) * 1.4).astype(np.float32)
    for name, array in dict(logits=logits, y_gt=labels, y_pred=expit(logits),
                            label_mask=np.ones_like(labels, dtype=bool), indices=np.arange(len(labels))).items():
        np.save(folder / (name + ".npy"), array)
    atomic_json(folder / "samples.json", [dict(path="image_{}_{}".format(split, i), patient_id="patient" + str(offset + i // 2))
                                          for i in range(len(labels))])
    atomic_json(folder / "evaluation.json", dict(split=split, checkpoint_sha256=checkpoint, checkpoint_epoch=2,
        training_config=dict(method=methods.method_config(), epochs=3, seed=0, batch_size_per_gpu=8, world_size=1,
                             lr=.001, layer_decay=.55, weight_decay=.05, clip_grad=0.0)))


@unittest.skipIf(torch is None, "Requires PyTorch/scipy environment")
class StudyMethodTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_losses_masking_focal_and_spa_link_stationarity(self):
        z = torch.tensor([[-3., -.7, .1, 2., 6.]], requires_grad=True)
        y = torch.tensor([[0., 1., .75, .25, 1.]])
        mask = torch.tensor([[1., 0., 1., 1., 0.]])
        expected = (torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none") * mask).sum() / mask.sum()
        self.assertTrue(torch.equal(methods.objective(z, y, mask, methods.method_config()), expected))
        focal = methods.method_config(dict(loss="focal", focal_gamma=0))
        self.assertTrue(torch.allclose(methods.objective(z, y, mask, focal), expected))
        for loss in methods.LOSSES:
            z.grad = None
            value = methods.objective(z, y, mask, methods.method_config(dict(loss=loss)))
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(z.grad).all())
            self.assertTrue(torch.equal(z.grad[mask == 0], torch.zeros_like(z.grad[mask == 0])))
        zero = methods.objective(z, y, torch.zeros_like(mask), methods.method_config(dict(loss="mdca")))
        self.assertEqual(float(zero), 0)
        for params in [dict(loss="spa_lpr", lpr_weight=0), dict(loss="spa_lpr", spa_k=1., spa_zeta=1., spa_b=0., lpr_weight=0)]:
            cfg = methods.method_config(params)
            target = methods.probability_logits(z, cfg).sigmoid().detach()
            gradient = torch.autograd.grad(methods.objective(z, target, torch.ones_like(z), cfg), z)[0]
            self.assertLess(float(gradient.abs().max()), 2e-7)
        self.assertTrue(torch.allclose(methods.probability_logits(z, methods.method_config(params)), z))
        linked = methods.probability_logits(torch.linspace(-50, 50, 1000), methods.method_config(dict(loss="spa_lpr")))
        self.assertTrue(torch.all(linked[1:] > linked[:-1]))

    def test_lpr_pair_formula_and_mixup_ignore_mass(self):
        cfg = methods.method_config(dict(loss="spa_lpr", lpr_weight=1.))
        z = torch.tensor([[.5, -.4, 1., 2., -1.]])
        y = torch.tensor([[1., 0., 1., 0., 0.]])
        mask = torch.ones_like(y)
        no_pair = methods.objective(z, y, mask, dict(cfg, lpr_weight=0.))
        odds = methods.probability_logits(z, cfg)
        expected = sum(torch.nn.functional.softplus(-(odds[0, i] - odds[0, j]))
                       for i in (0, 2) for j in (1, 3, 4)) * .1
        self.assertTrue(torch.allclose(methods.objective(z, y, mask, cfg) - no_pair, expected))
        x, y, mask = torch.randn(6, 3), torch.rand(6, 5), torch.zeros(6, 5)
        mixed_x, mixed_y, mass = methods.mix_batch(x, y, mask, .2)
        self.assertTrue(torch.isfinite(mixed_x).all() and torch.isfinite(mixed_y).all())
        self.assertEqual(float(mass.sum()), 0.)

    def test_uncertain_policies_only_change_uncertain_training_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, create_images=True)
            fields, rows = splits.read_rows(root / "train.csv")
            rows[0]["Atelectasis"] = "-1"
            import csv
            with (root / "train.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
            for policy in methods.UNCERTAINTIES:
                cfg = methods.method_config(dict(uncertain=policy))
                data = training.CheXpertImages(root, root / "train.csv", True, 0, cfg)
                _, target, mask, _ = data[0]
                self.assertEqual(bool(mask[0]), policy != "ignore")
                if policy == "soft":
                    self.assertTrue(.55 <= target[0] <= .85)
                    self.assertTrue(torch.equal(target, data[0][1]))
                if policy in ("ones", "zeros"):
                    self.assertEqual(float(target[0]), float(policy == "ones"))
                self.assertTrue(torch.equal(target[1:], torch.zeros(4)))

    def test_real_tiny_scopes_backward_and_adapter_initial_equivalence(self):
        # Actual upstream architecture, random weights only; two synthetic images,
        # no optimizer step, dataset access, checkpoint download or real training.
        device = torch.device("cpu")
        for scope in methods.SCOPES:
            training.seed_everything(0)
            model = training.build_model(device)
            model.eval()
            images = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                before = model(images)
            cfg = methods.method_config(dict(scope=scope))
            methods.configure_scope(model, cfg)
            model.eval()
            with torch.no_grad():
                after = model(images)
            self.assertTrue(torch.equal(before, after))
            methods.training_mode(model, cfg)
            loss = methods.objective(model(images), torch.rand(2, 5), torch.ones(2, 5), cfg)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, (scope, name))
                    self.assertTrue(torch.isfinite(parameter.grad).all(), (scope, name))
                else:
                    self.assertIsNone(parameter.grad, (scope, name))
            if scope == "head":
                self.assertTrue(all(n.startswith("head.") for n, p in model.named_parameters() if p.requires_grad))
            if scope == "lora":
                self.assertTrue(any("lora_b" in n and p.grad.abs().sum() > 0 for n, p in model.named_parameters()))

    def test_early_stop_checkpoint_resume_and_scope_evaluation(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([])
                self.head = torch.nn.Linear(3, 5)
            def no_weight_decay(self):
                return set()
            def forward(self, x):
                return self.head(x.mean((2, 3)))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, create_images=True)
            splits.prepare_splits(root, root / "splits")
            (root / "fixture.pt").write_bytes(b"mocked pretrained")
            atomic_json(root / "method.json", dict(scope="head", loss="bce"))
            args = argparse.Namespace(stage="train", loss="bce", smoothing=.05, data_root=root,
                splits_dir=root / "splits", pretrained=root / "fixture.pt", output_dir=root / "run", checkpoint=None,
                eval_split="test", resume=None, epochs=5, batch_size=16, workers=0, accumulation=1,
                lr=.001, weight_decay=.05, layer_decay=.55, seed=0, cpu=True,
                experiment_config=root / "method.json", clip_grad=0., patience=1, min_epochs=2, min_delta=.0001)
            def dev(*_):
                return dict(auroc=.8, nll=.4, auroc_per_class=[.8] * 5, observed_labels=[2] * 5), {}
            with patch.object(training, "build_model", side_effect=lambda device: ToyModel().to(device)), \
                 patch.object(training, "load_pretrained"), patch.object(training, "evaluate", side_effect=dev):
                training.run(args)
                saved = torch.load(root / "run/last.pth")
                self.assertEqual(saved["epoch"], 1)
                self.assertEqual(len((root / "run/history.jsonl").read_text().splitlines()), 2)
                args.resume = root / "run/last.pth"
                training.run(args)
                self.assertEqual(torch.load(root / "run/last.pth")["epoch"], 1)
            # Evaluate with the checkpoint's own scope, never default to full.
            args.stage, args.checkpoint, args.output_dir = "evaluate", root / "run/best.pth", root / "test"
            with patch.object(training, "build_model", side_effect=lambda device: ToyModel().to(device)):
                training.run(args)
            self.assertEqual(read_json(root / "test/evaluation.json")["training_config"]["method"]["scope"], "head")


@unittest.skipIf(torch is None, "Requires numerical environment")
class PostprocessReportTests(unittest.TestCase):
    def test_temperatures_positive_nll_and_order_preservation(self):
        rng = np.random.RandomState(9)
        logits = rng.randn(400, 5) * 4
        labels = (rng.rand(400, 5) < expit(logits / 3)).astype(float)
        mask = rng.rand(400, 5) > .1
        for per_class in (False, True):
            temperatures, fit = fit_temperatures(logits, labels, mask, per_class)
            self.assertTrue(np.all(temperatures > 0))
            self.assertLessEqual(binary_nll(logits / temperatures, labels)[mask].mean(), binary_nll(logits, labels)[mask].mean())
            self.assertTrue(np.array_equal(np.argsort(logits, axis=0), np.argsort(logits / temperatures, axis=0)))
        temperatures, fit = fit_temperatures(logits, np.zeros_like(labels), mask, True)
        self.assertTrue(np.array_equal(temperatures, np.ones(5)))
        with self.assertRaises(ValueError):
            fit_temperatures(logits, labels, np.zeros_like(mask))

    def test_disjoint_protocols_public_reports_and_paired_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = read_json(REPO / "configs/tiny_study_day.json")
            config["cases"] = config["cases"][:1]
            config["bootstrap_replicates"] = 4
            spec = dict(config=config, docs_dir=str(root / "docs"), report_template=None)
            atomic_json(root / "study.json", spec)
            folder = root / "runs/baseline/seed0"
            export_fixture(folder / "calibration", "calibration", offset=100)
            export_fixture(folder / "test", "test", offset=200)
            (folder / "train").mkdir()
            (folder / "train/history.jsonl").write_text('{"epoch":2}\n')
            processed = folder / "processed"
            report = postprocess(folder / "calibration", folder / "test", processed, plots=False)
            self.assertEqual(set(report["protocols"]), {"report_holdout", "expert_subset"})
            primary = report["protocols"]["report_holdout"]
            self.assertEqual(primary["n_test_images"], 24)
            self.assertEqual(report["protocols"]["expert_subset"]["n_test_patients"], 6)
            paths = [p for p in processed.rglob("*") if p.is_file()]
            atomic_json(root / "completed/baseline_seed0_postprocess.json", {"artifacts": {str(p.relative_to(root)): sha256(p) for p in paths}})
            build_report(root, bootstrap=True)
            self.assertTrue((root / "docs/baseline_seed0.md").exists())
            text = (root / "docs/baseline_seed0.md").read_text()
            self.assertNotIn("patient200", text)
            self.assertNotIn(str(root), text)
            self.assertIn("ts_shared", text)
            self.assertIn("ECE@10", text)
            ci = read_json(root / "bootstrap.json")["results"]["baseline"]["report_holdout"]["raw"]["delta_vs_baseline_same_ts"]
            self.assertEqual(ci["macro"]["brier"]["low"], 0.)
            self.assertEqual(ci["macro"]["brier"]["high"], 0.)
            # Leakage must be rejected even though array shapes and labels are valid.
            export_fixture(folder / "calibration", "calibration", offset=200)
            with self.assertRaises(ValueError):
                postprocess(folder / "calibration", folder / "test", root / "leaked", plots=False)

    def test_templates_and_manual_files_are_protected(self):
        template = REPO / "docs/experiments/_template.md"
        text = render_report("Title", "Setup", "Metrics", "Interpret", "Limits", "Files", template)
        self.assertNotIn("{{", text)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manual.md"
            path.write_text("Team manual report")
            with self.assertRaises(ValueError):
                write_managed(path, text)
            self.assertEqual(path.read_text(), "Team manual report")


class StudyRecoveryTests(unittest.TestCase):
    def test_cli_resume_reuses_full_checkpoint_path_and_dry_run_is_read_only(self):
        import contextlib
        import io
        import run_calibration_study as driver
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "mim.pt"
            weights.write_bytes(b"fixture; Study.run mocked, no weight loading or training")
            output = root / "study"
            with patch.object(sys, "argv", ["study", "--output-dir", str(output), "--pretrained", str(weights), "--docs-dir", str(root / "docs")]), \
                 patch.object(driver.Study, "run", return_value=0):
                self.assertEqual(driver.main(), 0)
            self.assertEqual(read_json(output / "study.json")["pretrained"], str(weights))
            with patch.object(sys, "argv", ["study", "--output-dir", str(output)]), patch.object(driver.Study, "run", return_value=0):
                self.assertEqual(driver.main(), 0)
            untouched = root / "dry"
            with patch.object(sys, "argv", ["study", "--output-dir", str(untouched), "--dry-run"]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(driver.main(), 0)
            self.assertFalse(untouched.exists())

    def test_config_single_seed_and_dev_only_selection(self):
        cfg = read_json(REPO / "configs/tiny_study_day.json")
        validate_config(cfg)
        cfg["seeds"] = [0, 1]
        with self.assertRaises(ValueError):
            validate_config(cfg)
        cfg["seeds"] = [0]
        metrics = {c["id"]: dict(auroc=.88, nll=.4) for c in cfg["cases"]}
        metrics["c_head"] = dict(auroc=.80, nll=.1)  # Good NLL cannot override AUROC gate.
        metrics["b_ignore"] = dict(auroc=.88, nll=.3)
        metrics["a_smoothing"] = dict(auroc=.881, nll=.35)
        selected = choose_combination(cfg["cases"], metrics, .005)
        self.assertNotEqual(selected["selected"]["C"], "c_head")
        self.assertEqual(selected["selected"]["B"], "b_ignore")
        self.assertEqual(selected["method"]["uncertain"], "ignore")
        self.assertEqual(selected["method"]["loss"], "label_smoothing")
        self.assertFalse(selected["test_used"])

    def test_real_subprocess_retry_hash_guard_and_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = read_json(REPO / "configs/tiny_study_day.json")
            study = Study(root, dict(config=cfg))
            artifact = root / "artifact.json"
            calls = []
            def command():
                calls.append(1)
                if len(calls) == 1:
                    return [sys.executable, "-c", "raise SystemExit(7)"]
                return [sys.executable, "-c", "from pathlib import Path; Path({!r}).write_text('ok')".format(str(artifact))]
            self.assertTrue(study.task("example", command, [artifact]))
            self.assertEqual(len(calls), 2)
            self.assertTrue(study.task("example", command, [artifact]))
            self.assertEqual(len(calls), 2)
            artifact.write_text("tampered")
            with self.assertRaises(ValueError):
                study.valid("example")
            state = read_json(root / "status.json")
            self.assertEqual(state["tasks"]["example"]["attempt"], 2)

    @unittest.skipIf(torch is None, "Requires numerical postprocessing")
    def test_full_pipeline_reports_combination_retry_resume_and_status(self):
        from study_status import display
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = read_json(REPO / "configs/tiny_study_day.json")
            cfg["cases"] = [cfg["cases"][0], cfg["cases"][1], cfg["cases"][10]]
            cfg["bootstrap_replicates"], cfg["plots"] = 2, False
            data = root / "data"
            data.mkdir()
            write_fixture(data, create_images=True)
            spec = dict(config=cfg, data_root=str(data), pretrained=str(root / "mock.pt"), docs_dir=str(root / "docs"), report_template=None)
            output = root / "output"
            output.mkdir()
            atomic_json(output / "study.json", spec)
            study = Study(output, spec)
            attempts = {}
            def stream(task, command):
                attempts[task] = attempts.get(task, 0) + 1
                def value(flag):
                    return command[command.index(flag) + 1]
                if task == "prepare":
                    splits.prepare_splits(data, output / "splits", check_images=True)
                elif task.startswith("check_"):
                    atomic_json(value("--check-receipt"), dict(checked=True))
                elif task.endswith("_train"):
                    if task == "c_head_seed0_train" and attempts[task] == 1:
                        return 9
                    dest = Path(value("--output-dir"))
                    dest.mkdir(parents=True, exist_ok=True)
                    (dest / "best.pth").write_bytes(task.encode())
                    (dest / "last.pth").write_bytes(task.encode())
                    atomic_json(dest / "config.json", dict(method=methods.method_config(read_json(value("--experiment-config")))))
                    (dest / "history.jsonl").write_text('{"epoch":1}\n')
                elif task.endswith(("_dev", "_calibration", "_test")):
                    split = value("--eval-split")
                    dest = Path(value("--output-dir"))
                    export_fixture(dest, split, offset=dict(dev=0, calibration=100, test=200)[split])
                    evaluation = read_json(dest / "evaluation.json")
                    evaluation["checkpoint_epoch"] = 1
                    evaluation["training_config"].update(read_json(dest.parent / "train/config.json"))
                    evaluation["checkpoint_sha256"] = sha256(dest.parent / "train/best.pth")
                    evaluation["metrics"] = dict(auroc=.88, nll=.4)
                    atomic_json(dest / "evaluation.json", evaluation)
                elif task.endswith("_postprocess"):
                    postprocess(Path(value("--calibration-dir")), Path(value("--test-dir")), Path(value("--output-dir")), plots=False)
                elif task == "summarize":
                    build_report(output, bootstrap=True)
                else:
                    self.fail(task)
                return 0
            with patch.object(study, "stream", side_effect=stream):
                self.assertEqual(study.run(), 0)
            self.assertEqual(attempts["c_head_seed0_train"], 2)
            self.assertTrue((output / "combination_selection.json").exists())
            self.assertFalse(read_json(output / "combination_selection.json")["test_used"])
            self.assertTrue((root / "docs/combination_seed0.md").exists())
            self.assertNotIn("](reports/", (root / "docs/README.md").read_text())
            self.assertIn("complete", (output / "STATUS.md").read_text())
            text, status = display(output)
            self.assertEqual(status, "complete")
            # Resume must not invoke training, inference or reporting a second time.
            resumed = Study(output, spec)
            with patch.object(resumed, "stream", side_effect=AssertionError("completed stage executed twice")):
                self.assertEqual(resumed.run(), 0)
            self.assertEqual(read_json(output / "aggregate.json")["completed_conditions"], 4)
            # A permanently failed middle condition must not cancel the later
            # independent MDCA run or falsely mark the combination complete.
            output = root / "partial_output"
            output.mkdir()
            spec = dict(spec, docs_dir=str(root / "partial_docs"))
            atomic_json(output / "study.json", spec)
            partial = Study(output, spec)
            def fail_head(task, command):
                return 9 if task == "c_head_seed0_train" else stream(task, command)
            with patch.object(partial, "stream", side_effect=fail_head):
                self.assertEqual(partial.run(), 1)
            self.assertTrue(partial.valid("a_mdca_seed0_postprocess"))
            state = read_json(output / "status.json")
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["tasks"]["combination_seed0_train"]["status"], "blocked")
            self.assertIn("Child process exited with code 9", (output / "STATUS.md").read_text())
            baseline_calls = attempts["baseline_seed0_train"]
            recovered = Study(output, spec)
            with patch.object(recovered, "stream", side_effect=stream):
                self.assertEqual(recovered.run(), 0)
            self.assertEqual(attempts["baseline_seed0_train"], baseline_calls)
            self.assertEqual(read_json(output / "aggregate.json")["completed_conditions"], 4)


if __name__ == "__main__":
    unittest.main()
