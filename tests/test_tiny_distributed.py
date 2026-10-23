"""Optional 2-process CPU DDP check: RUN_DISTRIBUTED_TESTS=1 python -m unittest discover -s tests -v"""
import argparse
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from test_tiny_training import write_fixture, splits, torch, training


def distributed_worker(rank, root_name, port):
    root = Path(root_name)
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([])
            self.head = torch.nn.Linear(3, 5)

        def no_weight_decay(self):
            return set()

        def forward(self, images):
            return self.head(images.mean(dim=(2, 3)))

    args = argparse.Namespace(stage="train", loss="label_smoothing", smoothing=.05, data_root=root,
        splits_dir=root / "splits", pretrained=root / "fixture.pt", output_dir=root / "run", checkpoint=None,
        eval_split="test", resume=None, epochs=1, batch_size=8, workers=0, accumulation=2,
        lr=.001, weight_decay=.05, layer_decay=.55, seed=0, cpu=True)
    with patch.object(training, "build_model", side_effect=lambda device: ToyModel().to(device)), \
            patch.object(training, "load_pretrained"):
        training.run(args)


@unittest.skipUnless(torch is not None and os.environ.get("RUN_DISTRIBUTED_TESTS") == "1",
                     "Set RUN_DISTRIBUTED_TESTS=1 for the local-loopback CPU DDP smoke test")
class DistributedTrainingTests(unittest.TestCase):
    def test_two_ranks_save_one_checkpoint_and_rank_rng_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_fixture(root, create_images=True)
            splits.prepare_splits(root, root / "splits", check_images=True)
            (root / "fixture.pt").write_bytes(b"synthetic fixture; model loading mocked")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            torch.multiprocessing.spawn(distributed_worker, args=(tmp, port), nprocs=2, join=True)
            checkpoint = torch.load(root / "run/last.pth", map_location="cpu")
            self.assertEqual(checkpoint["config"]["world_size"], 2)
            self.assertEqual(len(checkpoint["rng_states"]), 2)
            records = (root / "run/history.jsonl").read_text().splitlines()
            self.assertEqual(len(records), 1)
            self.assertEqual(json.loads(records[0])["epoch"], 1)
            self.assertFalse((root / "run/test").exists())


if __name__ == "__main__":
    unittest.main()
