"""Plain coordinate-MLP DeepONet for the CS 6180 microstrip task."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time

import h5py
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch import nn

SPLITS = ("train", "val", "test", "test_extrap")


class DeepONet(nn.Module):
    def __init__(self, width=128, rank=64):
        super().__init__()
        self.rank = rank
        def mlp(n):
            return nn.Sequential(nn.Linear(n, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, 2 * rank))
        self.branch = mlp(4)
        self.trunk = mlp(2)
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, x, coords):
        # coords: (batch, points, 2), in units of h. Fixed rescaling, no fit.
        b = self.branch(x).reshape(-1, 2, self.rank)
        t = self.trunk(coords / 8.0).reshape(x.shape[0], -1, 2, self.rank)
        return torch.einsum("bcp,bkcp->bck", b, t) + self.bias[None, :, None]


def masked_mse(prediction, target, mask):
    valid = (~mask).unsqueeze(1)
    return ((prediction - target).square() * valid).sum() / (valid.sum() * 2).clamp_min(1)


def load_data(path, device):
    with h5py.File(path, "r") as f:
        arrays = {k: f[k][:] for k in ("x", "u", "mask", "y_over_h", "z_over_h")}
        splits = {k: np.asarray(f[f"split/{k}"][:], dtype=np.int64) for k in SPLITS}
        source_ids = f["analysis/sample_id"][:] if "analysis/sample_id" in f else np.arange(len(arrays["x"]))
    n = len(arrays["x"])
    assert arrays["x"].shape == (n, 4)
    assert arrays["u"].shape == (n, 2, 64, 128)
    assert arrays["mask"].shape == (n, 64, 128)
    assert arrays["y_over_h"].shape == (n, 128)
    assert arrays["z_over_h"].shape == (n, 64)
    all_rows = np.concatenate(list(splits.values()))
    assert len(all_rows) == n and len(np.unique(all_rows)) == n
    assert all_rows.min() == 0 and all_rows.max() == n - 1
    for k, value in arrays.items():
        assert np.isfinite(value).all(), k
    assert np.all(np.diff(arrays["y_over_h"], axis=1) > 0)
    assert np.all(np.diff(arrays["z_over_h"], axis=1) > 0)
    assert (~arrays["mask"]).any(axis=(1, 2)).all()
    tensors = {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}
    tensors["mask"] = tensors["mask"].bool()
    return arrays, tensors, splits, source_ids


def coordinates(data, ids, pixels=None):
    if pixels is None:
        y = data["y_over_h"][ids, None, :].expand(-1, 64, -1)
        z = data["z_over_h"][ids, :, None].expand(-1, -1, 128)
        return torch.stack((y, z), dim=-1).flatten(1, 2)
    y = data["y_over_h"][ids].gather(1, pixels % 128)
    z = data["z_over_h"][ids].gather(1, pixels // 128)
    return torch.stack((y, z), dim=-1)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def validation_loss(model, data, indices, batch_size=4):
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(indices), batch_size):
        ids = torch.as_tensor(indices[start:start + batch_size], device=data["x"].device)
        pred = model(data["x"][ids], coordinates(data, ids)).reshape(-1, 2, 64, 128)
        valid = (~data["mask"][ids]).unsqueeze(1)
        total += float(((pred - data["u"][ids]).square() * valid).sum())
        count += int(valid.sum()) * 2
    return total / count


def relative_norm(error, truth):
    return float(np.linalg.norm(error) / max(float(np.linalg.norm(truth)), 1e-12))


def design_metrics(pred, truth, mask, y, z):
    valid = ~mask
    edge = valid & (distance_transform_edt(valid) <= 3)
    # Forward differences on the nonuniform physical coordinate grid (units h).
    # Both endpoints must be valid; gradients never cross the excluded trace.
    gy_valid = valid[:, 1:] & valid[:, :-1]
    gz_valid = valid[1:, :] & valid[:-1, :]
    dpy = np.diff(pred, axis=2) / np.diff(y)[None, None, :]
    dty = np.diff(truth, axis=2) / np.diff(y)[None, None, :]
    dpz = np.diff(pred, axis=1) / np.diff(z)[None, :, None]
    dtz = np.diff(truth, axis=1) / np.diff(z)[None, :, None]
    grad_error = np.concatenate(((dpy - dty)[:, gy_valid].ravel(),
                                 (dpz - dtz)[:, gz_valid].ravel()))
    grad_truth = np.concatenate((dty[:, gy_valid].ravel(), dtz[:, gz_valid].ravel()))
    return {
        "relative_l2": relative_norm((pred - truth)[:, valid], truth[:, valid]),
        "edge_relative_l2": relative_norm((pred - truth)[:, edge], truth[:, edge]),
        "gradient_relative_l2": relative_norm(grad_error, grad_truth),
        "masked_mse": float(np.mean((pred[:, valid] - truth[:, valid]) ** 2)),
    }


@torch.inference_mode()
def evaluate(model, data, arrays, indices, output, name, source_ids):
    model.eval()
    device = data["x"].device
    records, example_predictions = [], []
    for row in indices:
        ids = torch.tensor([int(row)], device=device)
        pred = model(data["x"][ids], coordinates(data, ids)).reshape(2, 64, 128).cpu().numpy()
        records.append({"row_index": int(row), "sample_id": int(source_ids[row]),
                        **design_metrics(pred, arrays["u"][row], arrays["mask"][row],
                                         arrays["y_over_h"][row], arrays["z_over_h"][row])})
        if len(example_predictions) < 4:
            example_predictions.append(pred)
    keys = ("relative_l2", "edge_relative_l2", "gradient_relative_l2", "masked_mse")
    summary = {key: {"mean": float(np.mean([r[key] for r in records])),
                     "median": float(np.median([r[key] for r in records])),
                     "std": float(np.std([r[key] for r in records]))} for key in keys}
    summary["designs"] = len(records)
    example_rows = np.asarray(indices[:len(example_predictions)])
    np.savez_compressed(output / f"{name}_examples.npz", row_indices=example_rows,
                        prediction=np.stack(example_predictions), truth=arrays["u"][example_rows],
                        mask=arrays["mask"][example_rows], y_over_h=arrays["y_over_h"][example_rows],
                        z_over_h=arrays["z_over_h"][example_rows])
    (output / f"{name}_per_design.json").write_text(json.dumps(records, indent=2) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("/courses/CS6180.202710/shared/final_proj/em2d_task.h5"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--pixels", type=int, default=1024)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=6180)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--smoke", action="store_true", help="Run two training batches and a small evaluation")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose an unused output directory: {args.output}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to run the GPU experiment on a CPU")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), 8))
    args.output.mkdir(parents=True)
    arrays, data, splits, source_ids = load_data(args.data, device)
    if args.smoke:
        splits = {k: v[:min(len(v), 4)] for k, v in splits.items()}
        args.epochs = 1
    model = DeepONet(args.width, args.rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"model": "DeepONet, plain SiLU coordinate trunk, no Fourier features",
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "torch_version": torch.__version__, "numpy_version": np.__version__,
                   "h5py_version": h5py.__version__, "hostname": socket.gethostname(),
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU smoke test",
                   "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "split_sizes": {k: len(v) for k, v in splits.items()},
                   "metric_convention": "Per-design relative norms; Euclidean 3-pixel edge band; forward physical-coordinate gradients with valid endpoints"})
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config), flush=True)
    best, best_epoch, stale = float("inf"), 0, 0
    sync(device)
    started = time.perf_counter()
    with (args.output / "history.jsonl").open("w", buffering=1) as history:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            order = np.random.permutation(splits["train"])
            loss_sum, batches = 0.0, 0
            for start in range(0, len(order), args.batch_size):
                ids = torch.as_tensor(order[start:start + args.batch_size], device=device)
                valid = (~data["mask"][ids]).flatten(1)
                pixels = torch.multinomial(valid.float(), args.pixels, replacement=True)
                target = data["u"][ids].flatten(2).gather(2, pixels[:, None, :].expand(-1, 2, -1))
                prediction = model(data["x"][ids], coordinates(data, ids, pixels))
                loss = (prediction - target).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item()
                batches += 1
                if args.smoke and batches >= 2:
                    break
            val = validation_loss(model, data, splits["val"])
            sync(device)
            record = {"epoch": epoch, "sampled_train_mse": loss_sum / batches,
                      "validation_masked_mse": val, "seconds": time.perf_counter() - epoch_start}
            print(json.dumps(record), flush=True)
            history.write(json.dumps(record) + "\n")
            if val < best:
                best, best_epoch, stale = val, epoch, 0
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "epoch": epoch, "validation_masked_mse": val, "config": config},
                           args.output / "best.pt")
            else:
                stale += 1
            if stale >= args.patience:
                break
    sync(device)
    train_seconds = time.perf_counter() - started
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    # Fixed validation design for batch-one model-only latency, excluding transfers.
    ids = torch.tensor([int(splits["val"][0])], device=device)
    sample_x, sample_coords = data["x"][ids], coordinates(data, ids)
    with torch.inference_mode():
        for _ in range(10):
            model(sample_x, sample_coords)
        sync(device)
        tic = time.perf_counter()
        for _ in range(50):
            model(sample_x, sample_coords)
        sync(device)
        latency = (time.perf_counter() - tic) / 50
    results = {"best_epoch": best_epoch, "best_validation_masked_mse": best,
               "train_seconds": train_seconds, "train_gpu_hours": train_seconds / 3600 if device.type == "cuda" else None,
               "batch_one_prediction_seconds": latency, "smoke_test_only": args.smoke}
    for split in ("test", "test_extrap"):
        results[split] = evaluate(model, data, arrays, splits[split], args.output, split, source_ids)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
