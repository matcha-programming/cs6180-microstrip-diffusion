"""Schedule math, adaptive state, controlled selection, and CPU recovery checks."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

import choose
import train
from r3_support import sha
from schedules import NAMES, Schedule, ScheduleSpec


def fixture(path):
    rng = np.random.default_rng(17)
    n = 12
    mask = np.zeros((n, 64, 128), dtype=bool)
    mask[:, 31:34, 60:68] = True
    with h5py.File(path, "w") as f:
        f["x"] = rng.uniform(-1, 1, (n, 4)).astype(np.float32)
        f["u"] = rng.normal(0, .1, (n, 2, 64, 128)).astype(np.float32) * (~mask[:, None])
        f["mask"] = mask
        f["y_over_h"] = np.tile(np.linspace(-6, 6, 128, dtype=np.float32), (n, 1))
        f["z_over_h"] = np.tile(np.linspace(0, 1, 64, dtype=np.float32) ** 2 * 8, (n, 1))
        for name, rows in {"train": np.arange(4), "val": np.arange(4, 8), "test": np.arange(8, 10), "test_extrap": np.arange(10, 12)}.items():
            f[f"split/{name}"] = rows
        f["analysis/sample_id"] = np.arange(100, 112)


def rates(spec):
    schedule = Schedule(spec)
    result = []
    for _ in range(spec.total_updates):
        result.append(schedule.next_lr)
        schedule.advance()
    return result


class ScheduleTests(unittest.TestCase):
    def test_constant_and_warmup_controls(self):
        self.assertEqual(rates(ScheduleSpec("constant", 10)), [1e-4] * 10)
        values = rates(ScheduleSpec("warmup-constant", 10, 2))
        self.assertAlmostEqual(values[0], 1e-5)
        self.assertEqual(values[1:], [1e-4] * 9)

    def test_cosine_endpoints_and_monotonic_tail(self):
        values = rates(ScheduleSpec("warmup-cosine", 10, 2))
        self.assertAlmostEqual(values[0], 1e-5)
        self.assertAlmostEqual(values[1], 1e-4)
        self.assertAlmostEqual(values[2], 1e-4)
        self.assertAlmostEqual(values[-1], 1e-6)
        self.assertTrue(all(a >= b for a, b in zip(values[2:], values[3:])))

    def test_multistep_exact_update_boundaries(self):
        values = rates(ScheduleSpec("warmup-multistep", 20, 2))
        self.assertAlmostEqual(values[11], 1e-4)
        self.assertAlmostEqual(values[12], 1e-5)
        self.assertAlmostEqual(values[16], 1e-5)
        self.assertAlmostEqual(values[17], 1e-6)

    def test_plateau_uses_only_postwarmup_field_observations(self):
        scheduler = Schedule(ScheduleSpec("warmup-plateau", 20, 2))
        scheduler.observe(.1)
        self.assertIsNone(scheduler.best)
        scheduler.advance(); scheduler.advance()
        scheduler.observe(.1)
        scheduler.observe(.1); scheduler.observe(.1)
        self.assertEqual(scheduler.next_lr, 1e-4)
        scheduler.observe(.1)
        self.assertEqual(scheduler.next_lr, 5e-5)
        scheduler.observe(.09)
        self.assertEqual(scheduler.bad_checks, 0)

    def test_scheduler_resume_matches_uninterrupted_and_rejects_horizon_change(self):
        for name in NAMES:
            spec = ScheduleSpec(name, 20, 0 if name == "constant" else 2)
            a, b = Schedule(spec), Schedule(spec)
            for _ in range(7):
                a.advance()
            a.observe(.2)
            b.load_state_dict(a.state_dict())
            for _ in range(13):
                self.assertEqual(a.next_lr, b.next_lr)
                a.advance(); b.advance()
                a.observe(.2); b.observe(.2)
            with self.assertRaises(ValueError):
                Schedule(replace(spec, total_updates=21)).load_state_dict(a.state_dict())


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "fixture.h5"
        fixture(self.data)

    def tearDown(self):
        self.temp.cleanup()

    def arguments(self, path, name):
        return ["--data", str(self.data), "--output", str(path), "--schedule", name,
                "--base", "4", "--device", "cpu", "--smoke"]

    def test_all_schedules_finish_same_number_of_updates(self):
        hashes = set()
        for name in NAMES:
            output = self.root / name
            train.main(self.arguments(output, name))
            cfg = json.loads((output / "config.json").read_text())
            val = json.loads((output / "validation.json").read_text())
            self.assertEqual(val["updates_in_round3"], 4)
            self.assertTrue(val["equal_update_budget_complete"])
            self.assertFalse(cfg["early_stopping"])
            hashes.add(cfg["initial_model_sha256"])
        self.assertEqual(len(hashes), 1)

    def test_mid_epoch_resume_preserves_optimizer_scheduler_and_rng(self):
        for name in ("warmup-cosine", "warmup-plateau"):
            full, partial = self.root / (name + "-full"), self.root / (name + "-partial")
            train.main(self.arguments(full, name))
            args = self.arguments(partial, name)
            train.main(args + ["--stop-after-updates", "1"])
            state = torch.load(partial / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(state["progress"]["cursor"], 2)
            self.assertEqual(json.loads((partial / "complete.json").read_text())["status"], "incomplete")
            train.main(args + ["--resume", str(partial / "last.pt")])
            a = torch.load(full / "last.pt", map_location="cpu", weights_only=False)
            b = torch.load(partial / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertTrue(torch.equal(a["torch_rng"], b["torch_rng"]))
            for key, value in a["model"].items():
                torch.testing.assert_close(value, b["model"][key], atol=0, rtol=0)

    def fake_candidate(self, name, seed, score, phase):
        # Manufactured metadata checks selection logic only; these are not model results.
        path = self.root / f"candidate-{name}-{seed}"
        path.mkdir()
        campaign = json.loads(choose.CAMPAIGN.read_text())
        spec = ScheduleSpec(name, 220500, 0 if name == "constant" else 2205)
        cfg = {"round": 3, "smoke_test_only": False, "base": 36, "backbone": "paired", "batch_size": 16,
               "learning_rate": .0001, "min_lr": .000001, "mixed_precision": False, "loss_weighting": "none",
               "max_updates": 220500, "data_sha256": campaign["dataset_sha256"], "split_hashes": {},
               "validation_rows": list(range(128)), "field_interval": 20, "field_steps": 50,
               "source_sha256": {}, "gpu": "fixture-gpu", "initial_model_sha256": str(seed),
               "schedule": name, "schedule_config": asdict(spec), "seed": seed, "phase": phase}
        (path / "config.json").write_text(json.dumps(cfg))
        checkpoint = path / "best-field.pt"
        checkpoint.write_bytes(f"{name}-{seed}".encode())
        val = {"smoke_test_only": False, "equal_update_budget_complete": True, "updates_in_round3": 220500,
               "checkpoints": {"best-field": {"epoch": 500, "sha256": sha(checkpoint),
                   "steps": {"20": {"designs": 881, "masked_mse_global": score},
                             "50": {"designs": 881, "masked_mse_global": score * 1.01}}}}}
        (path / "validation.json").write_text(json.dumps(val))
        (path / "complete.json").write_text(json.dumps({"status": "complete"}))
        return path

    def test_screen_then_seed_averaged_confirmation(self):
        paths = [self.fake_candidate(name, 6180, score, "screen")
                 for name, score in zip(NAMES, [.01, .009, .006, .008, .007])]
        plan = choose.screen(paths)
        self.assertEqual(plan["challenger"], "warmup-cosine")
        self.assertEqual(len(plan["confirmation_tasks"]), 4)
        confirm_paths = [paths[0], paths[2]]
        confirm_paths += [self.fake_candidate(name, seed, score, "confirm")
                          for name, score in [("constant", .012), ("warmup-cosine", .008)] for seed in (6181, 6182)]
        result = choose.confirm(confirm_paths, plan)
        self.assertEqual(result["winner"]["schedule"], "warmup-cosine")
        self.assertEqual(result["winner"]["steps"], 20)
        (paths[0] / "complete.json").write_text(json.dumps({"status": "incomplete"}))
        with self.assertRaisesRegex(ValueError, "update budget"):
            choose.screen(paths)


if __name__ == "__main__":
    unittest.main()
