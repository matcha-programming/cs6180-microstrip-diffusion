"""R2-B/C/D: direct DDPM continuation, learning rates, and optional Min-SNR ablation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from support import (DEFAULT_DATA, FieldData, dump, field_validation, legacy, make_model, masked_epsilon_objective,
                     provenance, save_checkpoint, setup, sha, sync, tensors, validation_rows)

ROUND1_DATA_SHA = "7670297247077ee223a572b724e45236a6970a1f38da02f9a96e074ca0215af2"


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
    required = {"model", "ema", "optimizer", "scaler", "epoch", "elapsed", "numpy_rng", "torch_rng", "config"}
    if not required.issubset(checkpoint):
        raise ValueError("True continuation requires last.pt with optimizer, EMA and RNG state; best.pt is weights only")
    old = checkpoint["config"]
    if old.get("mode", "direct") != "direct" or old.get("experiment") == "4R":
        raise ValueError("Only direct diffusion recovery checkpoints can be continued")
    paired = bool(old.get("position_conditioning")) or old.get("backbone") == "paired"
    for key in ("base", "seed", "batch_size"):
        if old[key] != config[key]:
            raise ValueError(f"Resume mismatch: {key}")
    if ("paired" if paired else "original") != config["backbone"]:
        raise ValueError("Resume backbone mismatch")
    if old.get("learning_rate", old.get("lr")) != config["learning_rate"]:
        raise ValueError("Continuation keeps the original learning rate; use a fresh R2-C run to change it")
    if old["mixed_precision"] != config["mixed_precision"]:
        raise ValueError("Resume precision mismatch; keep the source AMP policy")
    if old.get("loss_weighting", "none") != config.get("loss_weighting", "none"):
        raise ValueError("Resume loss weighting mismatch; loss ablations must start fresh")
    if old.get("loss_weighting", "none") == "min-snr" and old.get("snr_gamma") != config.get("snr_gamma"):
        raise ValueError("Resume Min-SNR gamma mismatch")
    if "round" in old:
        for key in ("data_sha256", "source_sha256", "split_hashes", "validation_rows", "field_steps", "field_interval", "patience_evaluations"):
            if old[key] != config[key]:
                raise ValueError(f"Resume protocol mismatch: {key}")
        if old["smoke_test_only"] != config["smoke_test_only"]:
            raise ValueError("Cannot mix smoke and full runs")
    else:
        if not config["smoke_test_only"] and config["data_sha256"] != ROUND1_DATA_SHA:
            raise ValueError("Round-one migration requires the verified course dataset")
        # Verify the legacy implementations rather than silently resuming changed code.
        from support import ROOT
        for group, folder in (("source_sha256", "remaining"), ("paired_source_sha256", "unet_regression")):
            for name, expected in old.get(group, {}).items():
                if sha(ROOT / "experiments" / folder / name) != expected:
                    raise ValueError(f"Legacy source mismatch: {folder}/{name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Full optimizer checkpoint; never a weights-only best.pt")
    parser.add_argument("--backbone", choices=["paired", "original"], default="paired")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--loss-weighting", choices=["none", "min-snr"], default="none")
    parser.add_argument("--snr-gamma", type=float, default=5.)
    parser.add_argument("--base", type=int, default=36)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max-train-hours", type=float, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=6180)
    parser.add_argument("--field-interval", type=int, default=20)
    parser.add_argument("--validation-count", type=int, default=128)
    parser.add_argument("--field-steps", type=int, default=50)
    parser.add_argument("--patience-evaluations", type=int, default=5)
    parser.add_argument("--amp", action="store_true", help="Default is FP32, matching the paired P100 run")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.device == "cpu" and not args.smoke:
        parser.error("CPU is reserved for --smoke")
    if min(args.lr, args.epochs, args.max_train_hours, args.batch_size, args.field_interval, args.patience_evaluations) <= 0:
        parser.error("Learning rate, budgets, batch size and intervals must be positive")
    if not np.isfinite(args.snr_gamma) or args.snr_gamma <= 0:
        parser.error("Min-SNR gamma must be finite and positive")
    if args.base < 4 or args.base % 4 or not 2 <= args.field_steps <= 1000 or args.validation_count < 0:
        parser.error("Base must be divisible by four; steps must be 2..1000; validation count must be nonnegative")
    if args.smoke:
        args.epochs = 2 if not args.resume else args.epochs
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
    model = make_model(args.backbone, args.base).to(device)
    initial_hash = state_hash(model)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"round": 2, "mode": "direct", "experiment": "R2-B" if args.resume else "R2-D" if args.loss_weighting == "min-snr" else "R2-C",
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
    best_field, best_noise, stale = float("inf"), float("inf"), 0
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
        if "progress" in last:
            config["experiment"] = last["config"]["experiment"]
            epoch, cursor, order, updates = (last["progress"][k] for k in ("epoch", "cursor", "order", "updates"))
            if "best_field" in last:
                best_field, best_noise, stale = last["best_field"], last["best_noise"], last["stale"]
                # Native resume stays in the directory containing its selected checkpoints.
                if args.resume.resolve() != (args.output / "last.pt").resolve():
                    raise ValueError("Resume a native round-two run in its existing output directory")
        else:
            epoch = last["epoch"] + 1
            config["legacy_migration"] = {
                "source_epoch": last["epoch"], "historical_train_seconds": elapsed,
                "epoch_boundary_only": True,
                "note": "Legacy checkpoint has no minibatch cursor; the partial final epoch cannot be replayed exactly. Start the next epoch; historical exact optimizer-update count is unavailable.",
            }
        config["resumed_checkpoint_sha256"] = sha(args.resume)
        np.random.set_state(last["numpy_rng"])
        torch.set_rng_state(last["torch_rng"].cpu())
        if device.type == "cuda":
            if not last.get("cuda_rng") or len(last["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("CUDA RNG state/device-count mismatch")
            torch.cuda.set_rng_state_all([s.cpu() for s in last["cuda_rng"]])
    if elapsed >= args.max_train_hours * 3600 or epoch > args.epochs:
        raise ValueError("The total budget/epoch limit must exceed the source run's consumed budget")
    if (args.output / "complete.json").exists():
        (args.output / "complete.json").unlink()
    dump(args.output / "config.json", config)
    alpha = legacy.schedule(device)
    sync(device)
    begin = time.perf_counter()
    reason = "epochs"

    def weights(recorded_epoch=None):
        return {"model": ema.state_dict(), "config": config, "epoch": epoch if recorded_epoch is None else recorded_epoch,
                "updates_in_round2": updates, "selection_field_mse": best_field,
                "validation_noise_mse": best_noise if np.isfinite(best_noise) else None}

    def save_last():
        save_checkpoint(args.output / "last.pt", {
            "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "config": config, "epoch": epoch,
            "progress": {"epoch": epoch, "cursor": cursor, "order": order, "updates": updates},
            "elapsed": elapsed + time.perf_counter() - begin, "best_field": best_field,
            "best_noise": best_noise, "stale": stale, "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        })

    with (args.output / "history.jsonl").open("a", buffering=1) as history:
        # Measure the starting EMA, so continuation has a field-scored reference.
        if not (args.output / "best-field.pt").exists():
            value = field_validation(ema, data, rows, mean, std, device, args.field_steps, batch_size=args.batch_size)
            best_field = value["masked_mse_global"]
            save_checkpoint(args.output / "best-field.pt", weights(epoch - 1))
            history.write(json.dumps({"event": "initial_field_validation", "epoch": epoch - 1, "field": value}) + "\n")
        while epoch <= args.epochs:
            if elapsed + time.perf_counter() - begin >= args.max_train_hours * 3600:
                reason = "training_budget"
                break
            if order is None:
                order = np.random.permutation(data.splits["train"])
                cursor = 0
            model.train()
            losses = []
            unweighted_losses, mean_weights = [], []
            while cursor < len(order):
                batch = data.batch(order[cursor:cursor + args.batch_size], device)
                valid = (~batch["mask"])[:, None]
                target = ((batch["u"] - mean) / std) * valid
                t = torch.randint(1000, (len(target),), device=device)
                noise = torch.randn_like(target) * valid
                a = alpha[t, None, None, None]
                noisy = (a.sqrt() * target + (1 - a).sqrt() * noise) * valid
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
                cursor += len(target)
                losses.append(loss.item())
                unweighted_losses.append(unweighted.item())
                mean_weights.append(float(weights_for_t.mean()))
                if args.smoke or elapsed + time.perf_counter() - begin >= args.max_train_hours * 3600:
                    break
            complete_epoch = cursor >= len(order)
            budget_hit = elapsed + time.perf_counter() - begin >= args.max_train_hours * 3600
            noise_score = noise_validation(ema, data, mean, std, alpha, device, args.batch_size)
            if noise_score < best_noise:
                best_noise = noise_score
                save_checkpoint(args.output / "best-noise.pt", weights())
            row = {"epoch": epoch, "complete_epoch": complete_epoch, "cursor": cursor,
                   "train_noise_mse": float(np.mean(unweighted_losses)), "train_objective": float(np.mean(losses)),
                   "mean_timestep_weight": float(np.mean(mean_weights)), "validation_noise_mse": noise_score,
                   "updates_in_round2": updates, "field": None}
            if epoch % args.field_interval == 0 or epoch == args.epochs or budget_hit or args.smoke:
                value = field_validation(ema, data, rows, mean, std, device, args.field_steps, batch_size=args.batch_size)
                row["field"] = value
                if value["masked_mse_global"] < best_field:
                    best_field, stale = value["masked_mse_global"], 0
                    save_checkpoint(args.output / "best-field.pt", weights())
                else:
                    stale += 1
            sync(device)
            row["total_train_seconds"] = elapsed + time.perf_counter() - begin
            history.write(json.dumps(row, allow_nan=False) + "\n")
            print("ROUND2_EPOCH", json.dumps(row), flush=True)
            if complete_epoch or args.smoke:
                epoch, cursor, order = epoch + 1, 0, None
            save_last()
            if budget_hit:
                reason = "training_budget"
                break
            if stale >= args.patience_evaluations:
                reason = "field_early_stopping"
                break
        save_last()
    duration = elapsed + time.perf_counter() - begin
    # Full validation confirmation also quantifies field-vs-noise checkpoint selection.
    validation = {"smoke_test_only": args.smoke, "stop_reason": reason,
                  "historical_plus_round2_train_seconds": duration,
                  "new_train_seconds": duration - elapsed, "updates_in_round2": updates, "checkpoints": {}}
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
    dump(args.output / "complete.json", {"status": "complete", "smoke_test_only": args.smoke})
    print("ROUND2_TRAIN_COMPLETE", str(args.output), flush=True)


if __name__ == "__main__":
    main()
