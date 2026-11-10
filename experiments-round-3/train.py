"""Round three: fresh direct-DDPM runs with controlled learning-rate schedules."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from r3_support import (DEFAULT_DATA, FieldData, dump, field_validation, legacy, make_model, masked_epsilon_objective,
                     provenance, save_checkpoint, setup, sha, sync, tensors, validation_rows)

from schedules import NAMES, Schedule, ScheduleSpec


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def noise_validation(model, data, mean, std, alpha, device, batch_size):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(9137)
    total, count = 0., 0
    rows = data.splits["val"]
    for start in range(0, len(rows), batch_size):
        batch = data.batch(rows[start:start + batch_size], device)
        valid = (~batch["mask"])[:, None]
        target = ((batch["u"] - mean) / std) * valid
        t = torch.randint(len(alpha), (len(target),), device=device, generator=generator)
        noise = torch.randn(target.shape, device=device, generator=generator) * valid
        a = alpha[t, None, None, None]
        pred = model((a.sqrt() * target + (1 - a).sqrt() * noise) * valid, t, batch["x"])
        total += float(((pred - noise).square() * valid).sum())
        count += int(valid.sum()) * 2
    return total / count


def validate_resume(checkpoint, config):
    required = {"model", "ema", "optimizer", "scaler", "epoch", "elapsed", "numpy_rng", "torch_rng", "config", "progress", "scheduler"}
    if not required.issubset(checkpoint) or checkpoint["config"].get("round") != 3:
        raise ValueError("Round three resumes only its own full recovery checkpoints; start schedule comparisons fresh")
    old = checkpoint["config"]
    keys = ("base", "seed", "batch_size", "backbone", "learning_rate", "mixed_precision",
            "data_sha256", "source_sha256", "split_hashes", "validation_rows", "field_steps",
            "field_interval", "schedule_config", "max_updates", "smoke_test_only", "phase")
    for key in keys:
        if old[key] != config[key]:
            raise ValueError(f"Resume protocol mismatch: {key}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Full optimizer checkpoint; never a weights-only best.pt")
    parser.add_argument("--backbone", choices=["paired", "original"], default="paired")
    parser.add_argument("--lr", type=float, default=1e-4, help="Common peak LR")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--schedule", choices=NAMES, default="constant")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--phase", choices=["screen", "confirm"], default="screen")
    parser.add_argument("--base", type=int, default=36)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-train-hours", type=float, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=6180)
    parser.add_argument("--field-interval", type=int, default=20)
    parser.add_argument("--validation-count", type=int, default=128)
    parser.add_argument("--field-steps", type=int, default=50)
    parser.add_argument("--stop-after-updates", type=int, help="Smoke-only interruption check; does not change schedule horizon")
    parser.add_argument("--amp", action="store_true", help="Default is FP32, matching the paired P100 run")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    args.loss_weighting, args.snr_gamma = "none", 5.
    if args.schedule == "constant":
        args.warmup_epochs = 0
    if args.device == "cpu" and not args.smoke:
        parser.error("CPU is reserved for --smoke")
    if min(args.lr, args.epochs, args.max_train_hours, args.batch_size, args.field_interval) <= 0:
        parser.error("Learning rate, budgets, batch size and intervals must be positive")
    if args.stop_after_updates is not None and (not args.smoke or args.stop_after_updates <= 0):
        parser.error("--stop-after-updates is a positive smoke-only interruption check")
    if args.warmup_epochs < 0:
        parser.error("Warmup epochs must be nonnegative")
    if args.base < 4 or args.base % 4 or not 2 <= args.field_steps <= 1000 or args.validation_count < 0:
        parser.error("Base must be divisible by four; steps must be 2..1000; validation count must be nonnegative")
    if args.smoke:
        args.epochs = 2
        args.field_interval, args.field_steps, args.validation_count, args.batch_size = 1, 2, 2, 2
    if args.output.exists() and any(args.output.iterdir()):
        if not args.resume or args.resume.resolve() != (args.output / "last.pt").resolve():
            parser.error("Use an unused output directory, or resume its own last.pt")
    args.output.mkdir(parents=True, exist_ok=True)
    device = setup(args.seed, args.device)
    data = FieldData(args.data)
    if args.smoke:
        data.splits = {k: v[:4] for k, v in data.splits.items()}
    rows = validation_rows(data, args.validation_count)
    mean_array, std_array = data.statistics()
    mean, std = tensors(mean_array, std_array, device)
    steps_per_epoch = math.ceil(len(data.splits["train"]) / args.batch_size)
    max_updates = steps_per_epoch * args.epochs
    warmup = 0 if args.schedule == "constant" else args.warmup_epochs * steps_per_epoch
    if args.smoke:
        warmup = min(warmup, max_updates - 2)
    spec = ScheduleSpec(args.schedule, max_updates, warmup, args.lr, args.min_lr)
    scheduler = Schedule(spec)
    model = make_model(args.backbone, args.base).to(device)
    initial_hash = state_hash(model)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"round": 3, "mode": "direct", "experiment": "R3-" + args.schedule,
                   "schedule_config": asdict(spec), "max_updates": max_updates,
                   "full_validation_steps": [2, 3] if args.smoke else [20, 50],
                   "full_validation_sampling_seeds": [6180] if args.smoke else [6180, 6181, 6182],
                   "early_stopping": False,
                   "learning_rate": args.lr, "mixed_precision": args.amp and device.type == "cuda",
                   "snr_gamma": args.snr_gamma if args.loss_weighting == "min-snr" else None,
                   "training_objective": "masked epsilon MSE" if args.loss_weighting == "none" else "masked epsilon MSE times min(SNR, gamma)/SNR; original valid-element denominator",
                   "smoke_test_only": args.smoke, "initial_model_sha256": initial_hash,
                   "parameters": sum(p.numel() for p in model.parameters()), "validation_rows": rows.tolist(),
                   "target_mean": mean_array.tolist(), "target_std": std_array.tolist(),
                   "data_sha256": sha(args.data), "split_hashes": data.split_hashes(),
                   "timesteps": 1000, "beta_schedule": "linear 0.0001 to 0.02", "ema_decay": .999,
                   "selection": "fixed validation panel global masked field MSE; DDIM eta=0; draw seed 6180",
                   "noise_policy": "round2 sample-ID/seed keyed validation; original random training corruption",
                   **provenance(device)})
    epoch, cursor, order, elapsed, updates = 1, 0, None, 0., 0
    best_field, best_noise = float("inf"), float("inf")
    if args.resume:
        last = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume(last, config)
        if not np.allclose(last["config"]["target_mean"], mean_array, atol=1e-7, rtol=1e-5) or not np.allclose(last["config"]["target_std"], std_array, atol=1e-7, rtol=1e-5):
            raise ValueError("Training-only channel statistics differ from the source run")
        # Continue with the exact recorded transform after independently verifying it.
        config["target_mean"] = last["config"]["target_mean"]
        config["target_std"] = last["config"]["target_std"]
        mean, std = tensors(config["target_mean"], config["target_std"], device)
        model.load_state_dict(last["model"])
        ema.load_state_dict(last["ema"])
        optimizer.load_state_dict(last["optimizer"])
        scaler.load_state_dict(last["scaler"])
        elapsed = last["elapsed"]
        epoch, cursor, order, updates = (last["progress"][k] for k in ("epoch", "cursor", "order", "updates"))
        best_field, best_noise = last["best_field"], last["best_noise"]
        scheduler.load_state_dict(last["scheduler"])
        if scheduler.completed != updates:
            raise ValueError("Optimizer-update and scheduler counters differ")
        if args.resume.resolve() != (args.output / "last.pt").resolve():
            raise ValueError("Resume in the original round-three output directory")
        config["resumed_checkpoint_sha256"] = sha(args.resume)
        np.random.set_state(last["numpy_rng"])
        torch.set_rng_state(last["torch_rng"].cpu())
        if device.type == "cuda":
            if not last.get("cuda_rng") or len(last["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("CUDA RNG state/device-count mismatch")
            torch.cuda.set_rng_state_all([s.cpu() for s in last["cuda_rng"]])
    if elapsed >= args.max_train_hours * 3600 or updates >= max_updates:
        raise ValueError("The total budget/epoch limit must exceed the source run's consumed budget")
    if (args.output / "complete.json").exists():
        (args.output / "complete.json").unlink()
    dump(args.output / "config.json", config)
    alpha = legacy.schedule(device)
    sync(device)
    begin = time.perf_counter()
    reason = "updates"

    def weights(recorded_epoch=None):
        return {"model": ema.state_dict(), "config": config, "epoch": epoch if recorded_epoch is None else recorded_epoch,
                "updates_in_round3": updates, "selection_field_mse": best_field,
                "scheduler": scheduler.state_dict(),
                "validation_noise_mse": best_noise if np.isfinite(best_noise) else None}

    def save_last():
        save_checkpoint(args.output / "last.pt", {
            "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "config": config, "epoch": epoch,
            "progress": {"epoch": epoch, "cursor": cursor, "order": order, "updates": updates},
            "elapsed": elapsed + time.perf_counter() - begin, "best_field": best_field,
            "best_noise": best_noise, "scheduler": scheduler.state_dict(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        })

    with (args.output / "history.jsonl").open("a", buffering=1) as history:
        # Equal checkpoint opportunities for every schedule, including the initial state.
        if not (args.output / "best-field.pt").exists():
            value = field_validation(ema, data, rows, mean, std, device, args.field_steps, batch_size=args.batch_size)
            best_field = value["masked_mse_global"]
            save_checkpoint(args.output / "best-field.pt", weights(epoch - 1))
            history.write(json.dumps({"event": "initial_field_validation", "epoch": epoch - 1, "field": value}) + "\n")
        while updates < max_updates:
            if elapsed + time.perf_counter() - begin >= args.max_train_hours * 3600:
                reason = "training_budget"
                break
            if order is None:
                order = np.random.permutation(data.splits["train"])
                cursor = 0
            model.train()
            losses = []
            unweighted_losses, mean_weights = [], []
            lrs = []
            while cursor < len(order):
                batch = data.batch(order[cursor:cursor + args.batch_size], device)
                valid = (~batch["mask"])[:, None]
                target = ((batch["u"] - mean) / std) * valid
                t = torch.randint(1000, (len(target),), device=device)
                noise = torch.randn_like(target) * valid
                a = alpha[t, None, None, None]
                noisy = (a.sqrt() * target + (1 - a).sqrt() * noise) * valid
                lr_used = scheduler.apply(optimizer)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=config["mixed_precision"]):
                    loss, unweighted, weights_for_t = masked_epsilon_objective(
                        model(noisy, t, batch["x"]), noise, batch["mask"], alpha[t], args.loss_weighting, args.snr_gamma)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    with torch.no_grad():
                        for averaged, current in zip(ema.parameters(), model.parameters()):
                            averaged.lerp_(current, .001)
                    updates += 1
                    scheduler.advance()
                    lrs.append(lr_used)
                cursor += len(target)
                losses.append(loss.item())
                unweighted_losses.append(unweighted.item())
                mean_weights.append(float(weights_for_t.mean()))
                if (updates >= max_updates or
                        (args.stop_after_updates and updates >= args.stop_after_updates) or
                        elapsed + time.perf_counter() - begin >= args.max_train_hours * 3600):
                    break
            complete_epoch = cursor >= len(order)
            budget_hit = elapsed + time.perf_counter() - begin >= args.max_train_hours * 3600
            interrupted = bool(args.stop_after_updates and updates >= args.stop_after_updates and updates < max_updates)
            if not complete_epoch:
                # No off-cadence validation/scheduler observations on preemption.
                history.write(json.dumps({"event": "partial_epoch", "epoch": epoch, "cursor": cursor,
                    "updates_in_round3": updates, "lr_next": scheduler.next_lr}) + "\n")
                save_last()
                if budget_hit or interrupted:
                    reason = "training_budget" if budget_hit else "smoke_interruption"
                    break
                continue
            noise_score = noise_validation(ema, data, mean, std, alpha, device, args.batch_size)
            if noise_score < best_noise:
                best_noise = noise_score
                save_checkpoint(args.output / "best-noise.pt", weights())
            row = {"epoch": epoch, "complete_epoch": complete_epoch, "cursor": cursor,
                   "train_noise_mse": float(np.mean(unweighted_losses)), "train_objective": float(np.mean(losses)),
                   "mean_timestep_weight": float(np.mean(mean_weights)), "validation_noise_mse": noise_score,
                   "updates_in_round3": updates, "field": None,
                   "lr_first": lrs[0] if lrs else None, "lr_last": lrs[-1] if lrs else None}
            if epoch % args.field_interval == 0 or updates >= max_updates:
                value = field_validation(ema, data, rows, mean, std, device, args.field_steps, batch_size=args.batch_size)
                row["field"] = value
                if value["masked_mse_global"] < best_field:
                    best_field = value["masked_mse_global"]
                    save_checkpoint(args.output / "best-field.pt", weights())
                scheduler.observe(value["masked_mse_global"])
            row["lr_next"] = scheduler.next_lr
            row["scheduler"] = scheduler.state_dict()
            sync(device)
            row["total_train_seconds"] = elapsed + time.perf_counter() - begin
            history.write(json.dumps(row, allow_nan=False) + "\n")
            print("ROUND3_EPOCH", json.dumps(row), flush=True)
            if complete_epoch:
                epoch, cursor, order = epoch + 1, 0, None
            save_last()
            if budget_hit:
                reason = "training_budget"
                break
            if interrupted:
                reason = "smoke_interruption"
                break
        save_last()
    duration = elapsed + time.perf_counter() - begin
    # Full validation confirmation also quantifies field-vs-noise checkpoint selection.
    validation = {"smoke_test_only": args.smoke, "stop_reason": reason,
                  "total_train_seconds": duration,
                  "new_train_seconds": duration - elapsed, "updates_in_round3": updates, "checkpoints": {}}
    validation["target_updates"] = max_updates
    validation["equal_update_budget_complete"] = updates == max_updates
    steps_list = [2, 3] if args.smoke else [20, 50]
    val_seeds = (6180,) if args.smoke else (6180, 6181, 6182)
    confirmation_begin = time.perf_counter()
    for name in ("best-field", "best-noise"):
        if not (args.output / f"{name}.pt").exists():
            continue
        chosen = torch.load(args.output / f"{name}.pt", map_location="cpu", weights_only=False)
        ema.load_state_dict(chosen["model"])
        validation["checkpoints"][name] = {
            "epoch": chosen["epoch"], "sha256": sha(args.output / f"{name}.pt"),
            "steps": {str(s): field_validation(ema, data, data.splits["val"], mean, std, device, s, val_seeds, args.batch_size)
                      for s in steps_list}}
    validation["full_validation_seconds"] = time.perf_counter() - confirmation_begin
    dump(args.output / "validation.json", validation)
    dump(args.output / "complete.json", {"status": "complete" if updates == max_updates else "incomplete",
                                        "smoke_test_only": args.smoke, "updates": updates, "target_updates": max_updates})
    print("ROUND3_TRAIN_COMPLETE", str(args.output), flush=True)


if __name__ == "__main__":
    main()
