"""Scientific checks and a CPU integration run on synthetic, explicitly marked data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

import draw_study
import train
from support import (FieldData, ROOT, initial_noise, legacy, load_weights, make_model,
                     masked_epsilon_objective, sample, sha, tensors, timestep_weights)
from freeze_selection import choose


def fixture(path):
    rng = np.random.default_rng(17)
    count = 12
    x = rng.uniform(-1, 1, (count, 4)).astype(np.float32)
    y = np.tile(np.linspace(-6, 6, 128, dtype=np.float32), (count, 1))
    z = np.tile((np.linspace(0, 1, 64, dtype=np.float32) ** 2) * 8, (count, 1))
    u = rng.normal(0, .1, (count, 2, 64, 128)).astype(np.float32)
    mask = np.zeros((count, 64, 128), dtype=bool)
    mask[:, 31:34, 60:68] = True
    u *= ~mask[:, None]
    with h5py.File(path, "w") as f:
        for key, value in {"x": x, "u": u, "mask": mask, "y_over_h": y, "z_over_h": z}.items():
            f[key] = value
        for key, value in {"train": np.arange(4), "val": np.arange(4, 8), "test": np.arange(8, 10), "test_extrap": np.arange(10, 12)}.items():
            f[f"split/{key}"] = value
        f["analysis/sample_id"] = np.arange(100, 112)


class RoundTwoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "fixture.h5"
        fixture(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_training_statistics_and_target_access_exclude_holdout(self):
        before = FieldData(self.path)
        statistics = before.statistics()
        with h5py.File(self.path, "a") as f:
            f["u"][8:] = 1e9
        after = FieldData(self.path)
        for a, b in zip(statistics, after.statistics()):
            np.testing.assert_array_equal(a, b)
        self.assertLess(float(after.arrays["u"].max()), 1)
        with self.assertRaises(ValueError):
            after.batch([8], "cpu")

    def test_noise_is_design_keyed_and_rng_isolated(self):
        torch.manual_seed(42)
        rng = torch.get_rng_state().clone()
        together = initial_noise([123, 456], 6180, "cpu")
        separately = torch.cat([initial_noise([456], 6180, "cpu"), initial_noise([123], 6180, "cpu")]).flip(0)
        self.assertTrue(torch.equal(together, separately))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertFalse(torch.equal(together, initial_noise([123, 456], 6181, "cpu")))

    def test_sampler_oracle_and_mask(self):
        alpha = legacy.schedule("cpu")
        clean = torch.full((1, 2, 64, 128), .2)
        class Oracle(torch.nn.Module):
            def forward(self, noisy, t, design):
                a = alpha[t, None, None, None]
                return (noisy - a.sqrt() * clean) / (1 - a).sqrt()
        batch = {"x": torch.zeros(1, 4), "mask": torch.zeros(1, 64, 128, dtype=torch.bool)}
        batch["mask"][:, 31:34, 60:68] = True
        mean, std = tensors([0, 0], [1, 1], "cpu")
        out = sample(Oracle(), batch, [12], 6180, mean, std, alpha, 4)
        expected = clean.masked_fill(batch["mask"][:, None], 0)
        torch.testing.assert_close(out, expected, atol=2e-5, rtol=2e-5)

    def test_streaming_ensemble_variance_is_unbiased(self):
        data = FieldData(self.path, ("val",))
        rows = data.splits["val"][:2]
        def fake_sample(model, batch, ids, seed, *args):
            return torch.full((len(ids), 2, 64, 128), (seed - 6179) * .1)
        with patch.object(draw_study, "sample", side_effect=fake_sample):
            ens, var, singles, draws, maps, _ = draw_study.study(None, None, None, data, rows, "cpu", 2, [1, 2, 4], 1, 6180)
        self.assertIsNone(var[1][0]["valid_pixel_variance"])
        self.assertAlmostEqual(var[2][0]["valid_pixel_variance"], .005, places=7)
        self.assertAlmostEqual(var[4][0]["valid_pixel_variance"], np.var([.1, .2, .3, .4], ddof=1), places=7)
        self.assertEqual(draws.shape, (4, 2, 2, 64, 128))
        self.assertEqual(len(singles), 4)

    def test_min_snr_epsilon_formula_and_limits(self):
        # SNR 100 -> gamma/SNR = .05; SNR 1 -> 1. Check both limiting cases.
        alpha = torch.tensor([100 / 101, .5, 0., 1.])
        actual = timestep_weights(alpha, "min-snr", 5.)
        torch.testing.assert_close(actual, torch.tensor([.05, 1., 1., 0.]), rtol=1e-5, atol=1e-7)
        with self.assertRaises(ValueError):
            timestep_weights(alpha, "min-snr", 0.)

    def test_min_snr_mask_gradient_and_uniform_control(self):
        pred = torch.ones(2, 2, 2, 3, requires_grad=True)
        noise = torch.zeros_like(pred)
        mask = torch.zeros(2, 2, 3, dtype=torch.bool)
        mask[0, 0, 0] = True
        alpha = torch.tensor([100 / 101, .5])
        objective, plain, weights = masked_epsilon_objective(pred, noise, mask, alpha, "min-snr", 5.)
        expected = (weights[0] * 10 + weights[1] * 12) / 22
        torch.testing.assert_close(objective, expected)
        self.assertEqual(float(plain.detach()), 1.)
        objective.backward()
        self.assertTrue((pred.grad[0, :, 0, 0] == 0).all())
        torch.testing.assert_close(pred.grad[0, 0, 1, 1], 2 * weights[0] / 22)
        changed = pred.detach().clone()
        changed[0, :, 0, 0] = 1e6
        after, _, _ = masked_epsilon_objective(changed, noise, mask, alpha, "min-snr", 5.)
        torch.testing.assert_close(objective.detach(), after)
        unweighted, _, _ = masked_epsilon_objective(changed, noise, mask, alpha)
        torch.testing.assert_close(unweighted, legacy.masked_noise_loss(changed, noise, mask), rtol=0, atol=0)

    def test_executed_architecture_is_reused(self):
        torch.manual_seed(6180)
        model = make_model("paired", 36)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 5_602_322)
        pair = json.loads((ROOT / "results/unet_regression/11604740/pair-config.json").read_text())
        self.assertEqual(train.state_hash(model), pair["initial_model_sha256"])
        original = make_model("original", 36)
        self.assertEqual(sum(p.numel() for p in original.parameters()), 5_601_674)

    def test_published_checkpoints_load_without_regression_dependency(self):
        for relative, expected in [
            ("results/remaining/11603443/best-direct.pt", "original"),
            ("results/unet_regression/11604740/best-diffusion.pt", "paired"),
        ]:
            model, mean, std, checkpoint, backbone = load_weights(ROOT / relative, "cpu")
            self.assertEqual(backbone, expected)
            self.assertEqual(mean.shape, (1, 2, 1, 1))
            self.assertTrue(torch.isfinite(std).all())
            self.assertEqual(sum(p.numel() for p in model.parameters()), checkpoint["config"]["parameters"])

    def test_resume_rejects_weights_only_or_changed_protocol(self):
        with self.assertRaisesRegex(ValueError, "optimizer"):
            train.validate_resume({"model": {}}, {})
        cfg = {"base": 36, "seed": 6180, "batch_size": 16, "backbone": "paired", "learning_rate": .0002,
               "mixed_precision": False, "round": 2, "data_sha256": "abc", "source_sha256": {},
               "split_hashes": {}, "validation_rows": [4, 5], "field_steps": 50,
               "field_interval": 20, "patience_evaluations": 5, "smoke_test_only": False}
        source = {k: None for k in ("model", "ema", "optimizer", "scaler", "epoch", "elapsed", "numpy_rng", "torch_rng")}
        source["config"] = copy.deepcopy(cfg)
        train.validate_resume(source, cfg)
        source["config"]["learning_rate"] = .0001
        with self.assertRaisesRegex(ValueError, "learning rate"):
            train.validate_resume(source, cfg)
        source["config"]["learning_rate"] = .0002
        changed_loss = {**cfg, "loss_weighting": "min-snr", "snr_gamma": 5.}
        with self.assertRaisesRegex(ValueError, "loss weighting"):
            train.validate_resume(source, changed_loss)
        source["config"].update(loss_weighting="min-snr", snr_gamma=1.)
        with self.assertRaisesRegex(ValueError, "gamma"):
            train.validate_resume(source, changed_loss)

    def test_min_snr_cpu_training_and_logging(self):
        output = self.root / "min-snr"
        # Force reweighting for every timestep in this tiny smoke run.
        # The formula test above separately covers the campaign's gamma=5.
        train.main(["--data", str(self.path), "--output", str(output), "--base", "4",
                    "--device", "cpu", "--smoke", "--loss-weighting", "min-snr", "--snr-gamma", "0.000001"])
        cfg = json.loads((output / "config.json").read_text())
        self.assertEqual(cfg["experiment"], "R2-D")
        self.assertEqual(cfg["snr_gamma"], .000001)
        history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
        for row in history:
            if "train_objective" in row:
                self.assertLessEqual(row["train_objective"], row["train_noise_mse"] + 1e-7)
                self.assertTrue(0 < row["mean_timestep_weight"] < 1)

    def test_cpu_training_draws_and_native_resume(self):
        output = self.root / "train"
        command = ["--data", str(self.path), "--output", str(output), "--base", "4", "--device", "cpu", "--smoke"]
        train.main(command)
        before = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        legacy_state = copy.deepcopy(before)
        legacy_state.pop("progress")
        legacy_state["epoch"] = 2
        legacy_state["config"].pop("round")
        legacy_state["config"]["source_sha256"] = {}
        legacy_path = self.root / "legacy-last.pt"
        torch.save(legacy_state, legacy_path)
        train.main(["--data", str(self.path), "--output", str(self.root / "migrated"),
                    "--base", "4", "--device", "cpu", "--smoke", "--epochs", "3", "--resume", str(legacy_path)])
        migrated = json.loads((self.root / "migrated/config.json").read_text())
        self.assertTrue(migrated["legacy_migration"]["epoch_boundary_only"])
        train.main(command + ["--resume", str(output / "last.pt"), "--epochs", "3"])
        after = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        self.assertGreater(after["progress"]["updates"], before["progress"]["updates"])
        self.assertEqual(after["progress"]["epoch"], 4)
        validation = json.loads((output / "validation.json").read_text())
        self.assertTrue(validation["smoke_test_only"])
        self.assertIn("best-noise", validation["checkpoints"])
        with self.assertRaisesRegex(ValueError, "Smoke"):
            choose([output], [1, 3])
        draw_study.main(["--data", str(self.path), "--output", str(self.root / "draws"),
                         "--checkpoint", str(output / "best-field.pt"), "--device", "cpu", "--smoke"])
        variance = json.loads((self.root / "draws/variance_2_per_design.json").read_text())
        self.assertEqual(len(variance), 2)
        self.assertIsNotNone(variance[0]["valid_pixel_variance"])

    def test_selection_uses_validation_and_detects_changed_checkpoint(self):
        folder = self.root / "candidate"
        folder.mkdir()
        cfg = {"smoke_test_only": False, "data_sha256": "course", "split_hashes": {},
               "validation_rows": [4, 5], "field_steps": 50, "learning_rate": .0002, "backbone": "paired"}
        validation = {"smoke_test_only": False, "checkpoints": {}}
        for name, value in [("best-field", .01), ("best-noise", .02)]:
            path = folder / f"{name}.pt"
            path.write_bytes(name.encode())
            validation["checkpoints"][name] = {"sha256": sha(path), "epoch": 20,
                "steps": {"20": {"designs": 881, "masked_mse_global": value},
                          "50": {"designs": 881, "masked_mse_global": value}}}
        for name, value in [("config.json", cfg), ("validation.json", validation),
                            ("complete.json", {"status": "complete", "smoke_test_only": False})]:
            (folder / name).write_text(json.dumps(value))
        winner = choose([folder], [1, 3])
        self.assertEqual(winner["steps"], 20)
        self.assertEqual(winner["validation_field_mse"], .01)
        (folder / "best-field.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            choose([folder], [1, 3])

    def test_holdout_requires_frozen_selection(self):
        with self.assertRaises(SystemExit):
            draw_study.main(["--checkpoint", "/missing.pt", "--output", str(self.root / "forbidden"),
                             "--split", "test", "--confirm-test"])
        self.assertFalse((self.root / "forbidden").exists())


if __name__ == "__main__":
    unittest.main()
