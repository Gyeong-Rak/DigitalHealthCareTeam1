"""Patient splits and tiny CPU checks; never train on the real dataset here."""
import copy
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_chexpert_splits as splits


def write_fixture(root, create_images=False):
    fields = ["Path", "Frontal/Lateral", *splits.CLASS_NAMES]
    for split, count, offset in [("train", 20, 0), ("valid", 4, 100)]:
        rows = []
        for patient in range(offset, offset + count):
            for view in [0, 1]:
                path = "{}/patient{:05d}/study{}/view_frontal.jpg".format(split, patient, view)
                row = {"Path": "CheXpert-v1.0-small/" + path, "Frontal/Lateral": "Frontal"}
                row.update({name: str(view) for name in splits.CLASS_NAMES})
                rows.append(row)
                if create_images:
                    from PIL import Image
                    target = root / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (32, 40), color=(patient % 255, view * 120, 60)).save(target)
        with (root / ("train.csv" if split == "train" else "valid.csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


class PatientSplitTests(unittest.TestCase):
    def test_reproducible_disjoint_split_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root)
            first = splits.prepare_splits(root, root / "splits", seed=42)
            self.assertEqual(first, splits.prepare_splits(root, root / "splits", seed=42))
            again = splits.prepare_splits(root, root / "again", seed=42)
            self.assertEqual(first, again)
            patient_sets = []
            for name in ["train", "dev", "calibration", "test"]:
                _, rows = splits.read_rows(root / "splits" / (name + ".csv"))
                patient_sets.append({splits.patient_id(r["Path"]) for r in rows})
            for i, patients in enumerate(patient_sets):
                for other in patient_sets[i + 1:]:
                    self.assertFalse(patients & other)
            self.assertEqual(sum(first["splits"][n]["images"] for n in ["train", "dev", "calibration"]), 40)
            with self.assertRaises(ValueError):
                splits.prepare_splits(root, root / "splits", seed=43)
            (root / "splits/train.csv").write_text("tampered")
            with self.assertRaises(ValueError):
                splits.prepare_splits(root, root / "splits", seed=42)

    def test_official_patient_overlap_and_missing_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root)
            with self.assertRaises(ValueError):
                splits.prepare_splits(root, root / "splits", check_images=True)
            (root / "valid.csv").write_text((root / "train.csv").read_text())
            with self.assertRaises(ValueError):
                splits.prepare_splits(root, root / "splits")


try:
    import torch
    import train_tiny_calibration as training
except ImportError:
    torch, training = None, None


@unittest.skipIf(torch is None, "Training checks require the evax/PyTorch environment")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_binary_smoothing_loss_and_gradients(self):
        logits = torch.tensor([[0.3, -1.0, 2.0, -3.0, 0.0]], requires_grad=True)
        targets = torch.tensor([[0., 1., 0., .75, .25]])
        base = training.training_loss(logits, targets)
        self.assertTrue(torch.equal(base, torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)))
        candidate = training.training_loss(logits, targets, "label_smoothing", .05)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets * .95 + .025)
        self.assertTrue(torch.equal(candidate, expected))
        candidate.backward()
        self.assertTrue(torch.allclose(logits.grad, (logits.detach().sigmoid() - (targets * .95 + .025)) / 5))
        with self.assertRaises(ValueError):
            training.training_loss(logits, targets, "label_smoothing", 1)

    def test_dataset_targets_are_reproducible_and_evaluation_masks_unknown_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, create_images=True)
            fields, rows = splits.read_rows(root / "train.csv")
            rows[0]["Atelectasis"], rows[0]["Cardiomegaly"] = "-1", ""
            with (root / "train.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            a = training.CheXpertImages(root, root / "train.csv", training=True, seed=0)
            b = training.CheXpertImages(root, root / "train.csv", training=True, seed=0)
            self.assertTrue(torch.equal(a[0][1], b[0][1]))
            self.assertTrue(.7 <= a[0][1][0] <= .8)
            self.assertEqual(float(a[0][1][1]), 0)
            evaluation = training.CheXpertImages(root, root / "train.csv")
            self.assertFalse(evaluation[0][2][0])
            self.assertTrue(evaluation[0][2][1])  # Missing report label = 0, same as training.
            self.assertEqual(len(a), 40)  # No upstream fake-epoch truncation/random half-dataset sampling.

    def test_accumulation_handles_partial_final_batches(self):
        from torch.utils.data import DataLoader, TensorDataset
        training.seed_everything(4)
        images, targets = torch.randn(7, 3), torch.rand(7, 5)
        dataset = TensorDataset(images, targets, torch.ones(7, 5), torch.arange(7))
        first = torch.nn.Linear(3, 5)
        second = copy.deepcopy(first)
        for model, batch, accumulation in [(first, 7, 1), (second, 2, 4)]:
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            training.train_epoch(model, DataLoader(dataset, batch_size=batch), optimizer,
                                  torch.cuda.amp.GradScaler(enabled=False), torch.device("cpu"), "bce", .05, accumulation)
        for a, b in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.allclose(a, b, atol=1e-7))

    def test_synthetic_training_checkpoint_resume_and_test_export(self):
        import argparse

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([])
                self.head = torch.nn.Linear(3, 5)

            def no_weight_decay(self):
                return set()

            def forward(self, images):
                return self.head(images.mean(dim=(2, 3)))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, create_images=True)
            splits.prepare_splits(root, root / "splits", check_images=True)
            pretrained = root / "fixture.pt"
            pretrained.write_bytes(b"fixture: load_pretrained mocked in this synthetic test")
            args = argparse.Namespace(stage="train", loss="bce", smoothing=.05, data_root=root,
                splits_dir=root / "splits", pretrained=pretrained, output_dir=root / "run", checkpoint=None,
                eval_split="test", resume=None, epochs=2, batch_size=8, workers=0, accumulation=3,
                lr=.001, weight_decay=.05, layer_decay=.55, seed=0, cpu=True)
            save = training.save_checkpoint

            def snapshot(path, model, optimizer, scaler, epoch, config, best, states):
                save(path, model, optimizer, scaler, epoch, config, best, states)
                if path.name == "last.pth" and epoch == 0:
                    save(path.parent / "epoch0.pth", model, optimizer, scaler, epoch, config, best, states)

            with patch.object(training, "build_model", side_effect=lambda device: ToyModel().to(device)), \
                    patch.object(training, "load_pretrained"), patch.object(training, "save_checkpoint", side_effect=snapshot):
                training.run(args)
                uninterrupted = torch.load(root / "run/last.pth", map_location="cpu")
                self.assertEqual(uninterrupted["epoch"], 1)
                self.assertFalse((root / "run/y_pred.npy").exists())
                args.resume = root / "run/epoch0.pth"
                training.run(args)
                resumed = torch.load(root / "run/last.pth", map_location="cpu")
                for key in uninterrupted["model"]:
                    self.assertTrue(torch.equal(uninterrupted["model"][key], resumed["model"][key]))
                args.stage, args.checkpoint, args.output_dir = "evaluate", root / "run/best.pth", root / "evaluation"
                training.run(args)
                import numpy as np
                self.assertEqual(np.load(root / "evaluation/y_pred.npy").shape, (8, 5))
                self.assertTrue(np.load(root / "evaluation/label_mask.npy").all())
                samples = json.loads((root / "evaluation/samples.json").read_text())
                self.assertEqual(len(samples), 8)
                self.assertEqual(len({s["patient_id"] for s in samples}), 4)


if __name__ == "__main__":
    unittest.main()
