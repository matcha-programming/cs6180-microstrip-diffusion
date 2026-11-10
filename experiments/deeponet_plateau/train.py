"""Continue the published plain DeepONet until a validation plateau."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "experiments/deeponet_plain/train.py"
spec = importlib.util.spec_from_file_location("plain", ORIGINAL)
plain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plain)
DATA_SHA = "7670297247077ee223a572b724e45236a6970a1f38da02f9a96e074ca0215af2"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj, binary=False):
    tmp = path.with_suffix(path.suffix + ".tmp")
    if binary:
        torch.save(obj, tmp)
    else:
        tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def window_update(state, window=50, threshold=.01):
    if state["new_epochs"] % window:
        return None
    gain = (state["anchor"] - state["best"]) / max(state["anchor"], 1e-30)
    state["streak"] = state["streak"] + 1 if gain < threshold else 0
    state["anchor"] = state["best"]
    return gain


def train(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing a production CPU run")
    if args.device == "cpu" and not args.smoke:
        raise RuntimeError("CPU is only allowed for smoke checks")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "run.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.output / "metrics.json").exists():
        raise RuntimeError("This run is already evaluated")
    last = args.output / "last.pt"
    if last.exists() != args.resume:
        raise RuntimeError("Use --resume only when last.pt exists; never overwrite a run")
    if not args.resume and (args.output / "config.json").exists():
        raise RuntimeError("Partial initialization exists; inspect before recovery")
    sha = digest(args.data)
    if not args.smoke and sha != DATA_SHA:
        raise RuntimeError("Dataset hash mismatch")
    if not args.smoke and (digest(ORIGINAL) != "32ca458eb54f3060fd9abf99625f6c7d9bf2e746b19250333d423dc3568e25b4"
            or digest(args.initial) != "b3870986a8f8243c6b22cde2686abd865ce611ca5b361542dc29f90c8bab6fe7"):
        raise RuntimeError("Published source/checkpoint hash mismatch")
    device = torch.device(args.device)
    torch.set_num_threads(min(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), 8))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(6180)
    np.random.seed(6180)
    torch.manual_seed(6180)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(6180)
    checkpoint = torch.load(last if args.resume else args.initial, map_location=device, weights_only=False)
    old = checkpoint["config"]
    for key, expected in {"width": 128, "rank": 64, "batch_size": 16,
                          "pixels": 1024, "lr": .001, "seed": 6180}.items():
        if old[key] != expected:
            raise RuntimeError("Original hyperparameter mismatch: " + key)
    model = plain.DeepONet(old["width"], old["rank"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=old["lr"], weight_decay=1e-6)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if not optimizer.state:
        raise RuntimeError("Optimizer state is missing")
    arrays, data, splits, ids = plain.load_data(args.data, device)
    if args.smoke:
        splits = {k: v[:2] for k, v in splits.items()}
    if args.resume:
        cfg = json.loads((args.output / "config.json").read_text())
        if (cfg["data_sha256"] != sha or cfg["original_source_sha256"] != digest(ORIGINAL)
                or cfg["continuation_source_sha256"] != digest(__file__)
                or cfg["smoke_test_only"] != args.smoke):
            raise RuntimeError("Continuation provenance mismatch")
        state = checkpoint["state"]
        best_weights = checkpoint["best_model"]
        save(args.output / "best.pt", {"model": best_weights, "epoch": state["best_epoch"],
             "validation_masked_mse": state["best"], "config": cfg}, True)
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])
        # Remove any uncommitted history tail after an interrupted atomic checkpoint save.
        records = [json.loads(s) for s in (args.output / "history.jsonl").read_text().splitlines()]
        (args.output / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records if r["epoch"] <= state["epoch"]))
    else:
        initial_val = plain.validation_loss(model, data, splits["val"])
        if not args.smoke and not np.isclose(initial_val, checkpoint["validation_masked_mse"], rtol=.01):
            raise RuntimeError("Original validation score did not reproduce")
        cfg = {**old, "epochs": None, "patience": None, "output": str(args.output),
               "historical_config": old, "data_sha256": sha, "initial_checkpoint_sha256": digest(args.initial),
               "initial_epoch": checkpoint["epoch"], "original_validation_mse": initial_val,
               "original_source_sha256": digest(ORIGINAL), "continuation_source_sha256": digest(__file__),
               "plateau_window": 50, "plateau_relative_gain": .01, "consecutive_windows": 2,
               "smoke_test_only": args.smoke, "epoch_cap": None,
               "continuation": "Model and AdamW state from published best epoch; old RNG state unavailable; new RNG seed 6180. Not the separate-machine trial.",
               "initial_job_id": os.environ.get("SLURM_JOB_ID"),
               "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU smoke",
               "torch_version": str(torch.__version__)}
        save(args.output / "config.json", cfg)
        state = {"epoch": checkpoint["epoch"], "best_epoch": checkpoint["epoch"],
                 "best": initial_val, "anchor": initial_val, "streak": 0, "new_epochs": 0,
                 "train_seconds": 0., "stop_reason": "training"}
        best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        save(args.output / "best.pt", {"model": model.state_dict(), "epoch": state["epoch"],
                                      "validation_masked_mse": initial_val, "config": cfg}, True)
        (args.output / "history.jsonl").touch()
    print(json.dumps({"event": "start", "job_id": os.environ.get("SLURM_JOB_ID"),
                      "hostname": socket.gethostname(), "state": state, "config": cfg}), flush=True)
    started = time.monotonic()
    while state["stop_reason"] != "validation_plateau":
        tic = time.monotonic()
        model.train()
        order = np.random.permutation(splits["train"])
        losses = []
        for start in range(0, len(order), old["batch_size"]):
            rows = torch.as_tensor(order[start:start + old["batch_size"]], device=device)
            valid = (~data["mask"][rows]).flatten(1)
            pixels = torch.multinomial(valid.float(), old["pixels"], replacement=True)
            truth = data["u"][rows].flatten(2).gather(2, pixels[:, None, :].expand(-1, 2, -1))
            prediction = model(data["x"][rows], plain.coordinates(data, rows, pixels))
            loss = (prediction - truth).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(loss.item())
            if args.smoke:
                break
        val = plain.validation_loss(model, data, splits["val"])
        plain.sync(device)
        if not np.isfinite(val):
            raise RuntimeError("Nonfinite validation loss")
        state["epoch"] += 1
        state["new_epochs"] += 1
        state["train_seconds"] += time.monotonic() - tic
        if val < state["best"]:
            state.update(best=val, best_epoch=state["epoch"])
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save(args.output / "best.pt", {"model": model.state_dict(), "epoch": state["epoch"],
                 "validation_masked_mse": val, "config": cfg}, True)
        gain = window_update(state)
        state["stop_reason"] = "validation_plateau" if state["streak"] >= 2 else "training"
        if state["stop_reason"] == "training" and time.monotonic() - started >= args.segment_seconds:
            state["stop_reason"] = "segment_budget_resume_required"
        if args.smoke and state["new_epochs"] >= args.smoke_epochs:
            state["stop_reason"] = "smoke_complete"
        record = {**state, "validation_masked_mse": val, "sampled_train_mse": float(np.mean(losses)),
                  "window_relative_gain": gain, "job_id": os.environ.get("SLURM_JOB_ID")}
        with (args.output / "history.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        save(last, {"model": model.state_dict(), "best_model": best_weights,
                    "optimizer": optimizer.state_dict(), "config": cfg,
                    "state": dict(state), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}, True)
        save(args.output / "status.json", record)
        print(json.dumps(record), flush=True)
        if state["stop_reason"] != "training":
            break
    if state["stop_reason"] in ("validation_plateau", "smoke_complete"):
        best = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model"])
        result = {**state, "smoke_test_only": args.smoke,
                  "validation_mse_improvement_factor": cfg["original_validation_mse"] / state["best"]}
        for split in ("test", "test_extrap"):
            result[split] = plain.evaluate(model, data, arrays, splits[split], args.output, split, ids)
        save(args.output / "metrics.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--segment-seconds", type=float, default=7 * 3600)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-epochs", type=int, default=2)
    train(p.parse_args())
