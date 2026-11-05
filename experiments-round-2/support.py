"""Round-two data, checkpoint, deterministic sampling and metric helpers."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = "/courses/CS6180.202710/shared/final_proj/em2d_task.h5"
METRICS = ("relative_l2", "edge_relative_l2", "gradient_relative_l2", "masked_mse")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load_module("round2_legacy_models", ROOT / "experiments/remaining/models.py")
metrics = load_module("round2_original_metrics", ROOT / "experiments/pca_gp/metric_utils.py")


class PositionUNet(legacy.ConditionalUNet):
    """State-dict-compatible with the executed 4R pair; no Fourier dependency."""

    def __init__(self, base=36):
        super().__init__(residual=True, base=base)
        row = torch.linspace(-1, 1, 64)[:, None].expand(64, 128)
        col = torch.linspace(-1, 1, 128)[None, :].expand(64, 128)
        self.register_buffer("position", torch.stack((row, col))[None], persistent=False)

    def forward(self, field, timestep, design):
        return super().forward(field, timestep, design, self.position.to(field).expand(len(field), -1, -1, -1))


def make_model(backbone, base):
    if backbone == "paired":
        return PositionUNet(base)
    if backbone == "original":
        return legacy.ConditionalUNet(base=base)
    raise ValueError(f"Unknown backbone: {backbone}")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def setup(seed, device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu only for a smoke test")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(int(os.getenv("SLURM_CPUS_PER_TASK", "4")), 8))
    return torch.device(device)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def provenance(device):
    paths = list(Path(__file__).parent.glob("*.py")) + [
        ROOT / "experiments/remaining/models.py", ROOT / "experiments/pca_gp/metric_utils.py"]
    return {
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "hostname": socket.gethostname(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "metric_convention": "run 11597463: valid pixels; Euclidean 3-pixel edge; forward coordinate gradients with valid endpoints",
    }


class FieldData:
    """CPU-resident targets for explicitly requested splits; GPU transfers per batch."""

    def __init__(self, path, allowed=("train", "val")):
        with h5py.File(path, "r") as f:
            self.splits = {s: f[f"split/{s}"][:].astype(np.int64) for s in ("train", "val", "test", "test_extrap")}
            n = len(f["x"])
            joined = np.concatenate(list(self.splits.values()))
            if len(joined) != n or not np.array_equal(np.sort(joined), np.arange(n)):
                raise ValueError("Splits must be disjoint and cover the dataset")
            if f["u"].shape != (n, 2, 64, 128):
                raise ValueError("Round two preserves the 64 x 128 dataset")
            self.rows = np.sort(np.concatenate([self.splits[s] for s in allowed]))
            self.lookup = np.full(n, -1, dtype=np.int64)
            self.lookup[self.rows] = np.arange(len(self.rows))
            self.arrays = {k: f[k][self.rows] for k in ("x", "u", "mask", "y_over_h", "z_over_h")}
            self.arrays["mask"] = self.arrays["mask"].astype(bool)
            self.sample_ids = f["analysis/sample_id"][:] if "analysis/sample_id" in f else np.arange(n)
        if len(np.unique(self.sample_ids)) != n:
            raise ValueError("Sample IDs must be unique for reproducible per-design noise")
        if self.arrays["x"].shape != (len(self.rows), 4):
            raise ValueError("Expected four design inputs")
        if not all(np.isfinite(a).all() for a in self.arrays.values()):
            raise ValueError("Non-finite data")
        if not (~self.arrays["mask"]).any(axis=(1, 2)).all():
            raise ValueError("A design has no valid pixels")
        for k in ("y_over_h", "z_over_h"):
            if not (np.diff(self.arrays[k], axis=1) > 0).all():
                raise ValueError("Coordinates must increase")

    def local(self, rows):
        ids = self.lookup[np.asarray(rows, dtype=np.int64)]
        if (ids < 0).any():
            raise ValueError("Targets for this split were not authorized by the command")
        return ids

    def batch(self, rows, device):
        ids = self.local(rows)
        return {k: torch.as_tensor(self.arrays[k][ids], device=device) for k in ("x", "u", "mask")}

    def statistics(self):
        total = np.zeros(2, dtype=np.float64)
        square = total.copy()
        count = 0
        for start in range(0, len(self.splits["train"]), 64):
            ids = self.local(self.splits["train"][start:start + 64])
            a = self.arrays["u"][ids].astype(np.float64)
            valid = (~self.arrays["mask"][ids])[:, None]
            total += (a * valid).sum(axis=(0, 2, 3))
            square += (a * a * valid).sum(axis=(0, 2, 3))
            count += valid.sum()
        mean = total / count
        std = np.sqrt(np.maximum(square / count - mean * mean, 1e-12))
        return mean.astype(np.float32), std.astype(np.float32)

    def split_hashes(self):
        return {s: hashlib.sha256(a.tobytes()).hexdigest() for s, a in self.splits.items()}


def validation_rows(data, count=128):
    rows = data.splits["val"]
    if count <= 0 or count >= len(rows):
        return rows.copy()
    chosen = np.random.default_rng(9137).choice(rows, count, replace=False)
    return rows[np.isin(rows, chosen)]


def tensors(mean, std, device):
    return tuple(torch.as_tensor(a, dtype=torch.float32, device=device).reshape(1, 2, 1, 1) for a in (mean, std))


def timestep_weights(alpha_bar, weighting="none", gamma=5.):
    """Epsilon-prediction Min-SNR: min(SNR, gamma)/SNR, not min(SNR, gamma).

    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training
    The equivalent ratio below is stable at both alpha_bar endpoints.
    """
    if weighting == "none":
        return torch.ones_like(alpha_bar, dtype=torch.float32)
    if weighting != "min-snr" or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Expected none or min-snr weighting with finite positive gamma")
    a = alpha_bar.float()
    return (gamma * (1 - a) / a.clamp_min(torch.finfo(a.dtype).tiny)).clamp(max=1.)


def masked_epsilon_objective(prediction, noise, mask, alpha_bar, weighting="none", gamma=5.):
    """Preserve the control's valid-element denominator; do not renormalize weights."""
    valid = (~mask)[:, None]
    square = (prediction.float() - noise.float()).square() * valid
    denominator = (valid.sum() * 2).clamp_min(1)
    unweighted = square.sum() / denominator
    weights = timestep_weights(alpha_bar, weighting, gamma)
    objective = unweighted if weighting == "none" else (square * weights[:, None, None, None]).sum() / denominator
    return objective, unweighted, weights


def load_weights(path, device, allow_smoke=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if cfg.get("mode", "direct") != "direct" or cfg.get("experiment") == "4R":
        raise ValueError("Round-two commands accept direct diffusion checkpoints only")
    if (cfg.get("smoke") or cfg.get("smoke_test_only")) and not allow_smoke:
        raise ValueError("A smoke checkpoint cannot produce full experiment results")
    if cfg.get("timesteps", 1000) != 1000:
        raise ValueError("This campaign uses the recorded 1,000-step training schedule")
    backbone = cfg.get("backbone", "paired" if cfg.get("position_conditioning") else "original")
    model = make_model(backbone, cfg["base"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mean, std = tensors(cfg["target_mean"], cfg["target_std"], device)
    return model, mean, std, checkpoint, backbone


def initial_noise(sample_ids, seed, device):
    # CPU generation fixes noise across batch sizes, row ordering, and CPU/CUDA.
    fields = []
    for sample_id in sample_ids:
        key = hashlib.sha256(f"round2:{int(sample_id)}:{int(seed)}".encode()).digest()
        generator = torch.Generator().manual_seed(int.from_bytes(key[:8], "little") % (2**63 - 1))
        fields.append(torch.randn((2, 64, 128), generator=generator))
    return torch.stack(fields).to(device)


@torch.inference_mode()
def sample(model, batch, sample_ids, seed, mean, std, alpha, steps, noise=None):
    model.eval()
    valid = (~batch["mask"])[:, None]
    noisy = (initial_noise(sample_ids, seed, batch["x"].device) if noise is None else noise) * valid
    grid = np.linspace(len(alpha) - 1, 0, steps, dtype=np.int64)
    if steps < 2 or steps > len(alpha) or len(np.unique(grid)) != steps:
        raise ValueError("DDIM steps must be between 2 and the number of training timesteps")
    def bounded_clean(clean):
        return ((clean * std + mean).clamp(-1, 1) - mean) / std
    for i, timestep in enumerate(grid):
        t = torch.full((len(noisy),), int(timestep), device=noisy.device, dtype=torch.long)
        epsilon = model(noisy, t, batch["x"])
        previous = alpha[int(grid[i + 1])] if i + 1 < steps else alpha.new_tensor(1.)
        noisy = legacy.ddim_update(noisy, epsilon, alpha[int(timestep)], previous, batch["mask"], bounded_clean)
    return (noisy * std + mean).masked_fill(batch["mask"][:, None], 0)


def records(predictions, rows, data):
    result = []
    for p, row, local in zip(predictions, rows, data.local(rows)):
        a = data.arrays
        result.append({"row_index": int(row), "sample_id": int(data.sample_ids[row]),
                       **metrics.design_metrics(p, a["u"][local], a["mask"][local], a["y_over_h"][local], a["z_over_h"][local])})
    return result


def summarize(rows):
    return {**{k: {"mean": float(np.mean([r[k] for r in rows])),
                   "median": float(np.median([r[k] for r in rows])),
                   "std": float(np.std([r[k] for r in rows]))} for k in METRICS}, "designs": len(rows)}


@torch.inference_mode()
def field_validation(model, data, rows, mean, std, device, steps=50, seeds=(6180,), batch_size=16):
    alpha = legacy.schedule(device)
    all_records = []
    squared_error = 0.
    elements = 0
    for seed in seeds:
        seed_records = []
        for start in range(0, len(rows), batch_size):
            ids = rows[start:start + batch_size]
            batch = data.batch(ids, device)
            pred = sample(model, batch, data.sample_ids[ids], seed, mean, std, alpha, steps)
            valid = (~batch["mask"])[:, None]
            squared_error += float(((pred - batch["u"]).square() * valid).double().sum())
            elements += int(valid.sum()) * 2
            seed_records.extend(records(pred.cpu().numpy(), ids, data))
        all_records.append(seed_records)
    result = {k: float(np.mean([summarize(r)[k]["mean"] for r in all_records])) for k in METRICS}
    result["masked_mse_global"] = squared_error / elements
    result["designs"] = len(rows)
    return result


def timed(call, device, warmups=2, repeats=5):
    for _ in range(warmups):
        call()
    sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    sync(device)
    return (time.perf_counter() - start) / repeats
