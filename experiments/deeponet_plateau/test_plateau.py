import importlib.util
from pathlib import Path
import unittest
import argparse
import tempfile
import h5py
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("plateau", Path(__file__).with_name("train.py"))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class PlateauTests(unittest.TestCase):
    def test_two_disjoint_windows(self):
        s = {"new_epochs": 49, "anchor": 1., "best": .995, "streak": 0}
        self.assertIsNone(p.window_update(s))
        s["new_epochs"] = 50
        self.assertAlmostEqual(p.window_update(s), .005)
        self.assertEqual(s["streak"], 1)
        s.update(new_epochs=100, best=.991)
        p.window_update(s)
        self.assertEqual(s["streak"], 2)

    def test_substantial_improvement_resets_plateau(self):
        s = {"new_epochs": 100, "anchor": 1., "best": .9, "streak": 1}
        p.window_update(s)
        self.assertEqual(s["streak"], 0)
        self.assertEqual(s["anchor"], .9)

    def test_segment_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "tiny.h5"
            with h5py.File(data, "w") as f:
                f["x"] = np.full((8, 4), .2, dtype=np.float32)
                f["u"] = np.full((8, 2, 64, 128), .05, dtype=np.float32)
                f["mask"] = np.zeros((8, 64, 128), dtype=bool)
                f["y_over_h"] = np.tile(np.linspace(-4, 4, 128, dtype=np.float32), (8, 1))
                f["z_over_h"] = np.tile(np.linspace(-2, 2, 64, dtype=np.float32), (8, 1))
                for i, key in enumerate(p.plain.SPLITS):
                    f["split/" + key] = np.array([i * 2, i * 2 + 1])
            torch.manual_seed(1)
            model = p.plain.DeepONet()
            opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-6)
            sum(t.square().sum() for t in model.parameters()).backward()
            opt.step()
            cfg = dict(width=128, rank=64, batch_size=16, pixels=1024, lr=.001, seed=6180)
            initial = root / "initial.pt"
            torch.save(dict(config=cfg, model=model.state_dict(), optimizer=opt.state_dict(),
                            epoch=95, validation_masked_mse=1.), initial)
            def args(out, seconds=3600, resume=False):
                return argparse.Namespace(data=data, initial=initial, output=root / out, device="cpu",
                    smoke=True, smoke_epochs=3, segment_seconds=seconds, resume=resume)
            p.train(args("full"))
            p.train(args("split", seconds=0))
            self.assertFalse((root / "split/metrics.json").exists())
            # Simulate an uncommitted history append before a process crash.
            with (root / "split/history.jsonl").open("a") as f:
                f.write('{"epoch": 99999}\n')
            p.train(args("split", resume=True))
            a = torch.load(root / "full/last.pt", weights_only=False)
            b = torch.load(root / "split/last.pt", weights_only=False)
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            self.assertEqual(b["state"]["new_epochs"], 3)
            self.assertEqual(len((root / "split/history.jsonl").read_text().splitlines()), 3)
            self.assertTrue((root / "split/test_per_design.json").exists())


if __name__ == "__main__":
    unittest.main()
