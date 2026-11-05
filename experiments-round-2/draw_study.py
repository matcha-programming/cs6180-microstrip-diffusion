"""R2-A: nested ensemble sizes and unbiased per-design sampling variance."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from support import (DEFAULT_DATA, FieldData, dump, initial_noise, legacy, load_weights, provenance,
                     records, sample, setup, sha, summarize, sync, timed, validation_rows)


def study(model, mean, std, data, rows, device, steps, sizes, batch_size, seed_start):
    """Welford accumulation uses O(batch * pixels), independent of draw count."""
    ensembles = {k: [] for k in sizes}
    variances = {k: [] for k in sizes}
    singles = [[] for _ in range(max(sizes))]
    examples = []
    example_variance = []
    alpha = legacy.schedule(device)
    sync(device)
    begin = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        ids = rows[start:start + batch_size]
        batch = data.batch(ids, device)
        running = np.zeros((len(ids), 2, 64, 128), dtype=np.float64)
        m2 = np.zeros_like(running)
        saved = []
        for draw in range(1, max(sizes) + 1):
            pred = sample(model, batch, data.sample_ids[ids], seed_start + draw - 1, mean, std, alpha, steps).cpu().numpy()
            singles[draw - 1].extend(records(pred, ids, data))
            delta = pred - running
            running += delta / draw
            m2 += delta * (pred - running)
            if start < 4:
                saved.append(pred[:max(0, 4 - start)])
            if draw not in sizes:
                continue
            ensembles[draw].extend(records(running.astype(np.float32), ids, data))
            for j, (row, local) in enumerate(zip(ids, data.local(ids))):
                valid = ~data.arrays["mask"][local]
                edge = valid & (distance_transform_edt(valid) <= 3)
                var = m2[j] / (draw - 1) if draw > 1 else None
                variances[draw].append({
                    "row_index": int(row), "sample_id": int(data.sample_ids[row]),
                    "normalized_design": data.arrays["x"][local].tolist(),
                    "valid_pixel_variance": float(var[:, valid].mean()) if var is not None else None,
                    "edge_pixel_variance": float(var[:, edge].mean()) if var is not None and edge.any() else None,
                    "channel_variance": var[:, valid].mean(axis=1).tolist() if var is not None else None,
                    "estimated_ensemble_variance": float(var[:, valid].mean() / draw) if var is not None else None,
                })
        if start < 4:
            examples.append(np.stack(saved))
            example_variance.append((m2 / (max(sizes) - 1))[:max(0, 4 - start)] if max(sizes) > 1 else np.zeros_like(m2[:max(0, 4 - start)]))
    sync(device)
    elapsed = time.perf_counter() - begin
    return ensembles, variances, singles, np.concatenate(examples, axis=1), np.concatenate(example_variance), elapsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--split", choices=["val", "test", "test_extrap"], default="val")
    parser.add_argument("--limit", type=int, default=128, help="0 selects the full split")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=6180)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--confirm-test", action="store_true", help="Explicit final evaluation after validation selection")
    parser.add_argument("--selection", type=Path, help="Frozen freeze_selection.py manifest, required for held-out reporting")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.split != "val" and not args.confirm_test:
        parser.error("Held-out evaluation requires --confirm-test after freezing validation choices")
    selection = None
    if args.split != "val":
        import json
        if args.smoke:
            parser.error("Smoke output cannot be a frozen held-out report")
        if not args.selection:
            parser.error("Held-out evaluation also requires --selection from freeze_selection.py")
        selection = json.loads(args.selection.read_text())
        if selection.get("selection_split") != "val" or selection.get("smoke_test_only"):
            parser.error("Expected a full-data validation selection manifest")
        if sha(args.checkpoint) != selection["checkpoint_sha256"] or args.steps != selection["steps"]:
            parser.error("Checkpoint or sampler differs from the frozen manifest")
        if sorted(set(args.sizes)) != selection["report_sizes"] or args.limit != 0:
            parser.error("Final reporting requires --limit 0 and exactly the frozen --sizes")
        if args.seed_start != selection["seed_start"]:
            parser.error("Sampling seeds differ from the frozen manifest")
    if args.device == "cpu" and not args.smoke:
        parser.error("CPU execution is reserved for --smoke")
    if min(args.sizes) < 1 or args.limit < 0 or args.batch_size < 1:
        parser.error("Positive draw counts and batch size, and nonnegative limit required")
    args.sizes = sorted(set(args.sizes))
    args.output.mkdir(parents=True, exist_ok=False)
    device = setup(6180, args.device)
    data = FieldData(args.data, (args.split,))
    rows = validation_rows(data, args.limit) if args.split == "val" else data.splits[args.split]
    if args.split != "val" and args.limit:
        rows = rows[:args.limit]
    if args.smoke:
        rows = rows[:2]
        args.sizes = [1, 2]
        args.steps = 2
    model, mean, std, checkpoint, backbone = load_weights(args.checkpoint, device, args.smoke)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"experiment": "R2-A", "backbone": backbone, "checkpoint_sha256": sha(args.checkpoint),
                   "data_sha256": sha(args.data), "split_hashes": data.split_hashes(),
                   "row_indices": rows.tolist(), "checkpoint_epoch": checkpoint.get("epoch"),
                   "smoke_test_only": args.smoke, "sampling_noise": "round2 sample-ID/seed keyed CPU Gaussian",
                   "variance_ddof": 1, "sampler": "DDIM", "eta": 0, **provenance(device)})
    if selection and (config["data_sha256"] != selection["data_sha256"] or config["split_hashes"] != selection["split_hashes"]):
        raise ValueError("Final dataset differs from the validation selection")
    if selection:
        config["selection_manifest_sha256"] = sha(args.selection)
    dump(args.output / "config.json", config)
    ens, var, single, draws, maps, wall = study(model, mean, std, data, rows, device, args.steps, args.sizes, args.batch_size, args.seed_start)
    for j, r in enumerate(single):
        dump(args.output / f"draw_{args.seed_start + j}_per_design.json", r)
    result = {"experiment": "R2-A", "smoke_test_only": args.smoke, "split": args.split,
              "checkpoint_sha256": config["checkpoint_sha256"], "evaluation_wall_seconds": wall, "sizes": {}}
    batch = data.batch(rows[:1], device)
    alpha = legacy.schedule(device)
    noise = initial_noise(data.sample_ids[rows[:1]], args.seed_start, device)
    seconds = timed(lambda: sample(model, batch, data.sample_ids[rows[:1]], args.seed_start, mean, std,
                                   alpha, args.steps, noise=noise), device)
    result["timing_convention"] = "batch one synchronized DDIM; cached Gaussian input on GPU; excludes noise construction, transfers, scoring and I/O"
    for k in args.sizes:
        dump(args.output / f"ensemble_{k}_per_design.json", ens[k])
        dump(args.output / f"variance_{k}_per_design.json", var[k])
        result["sizes"][str(k)] = {
            "ensemble": summarize(ens[k]),
            "single_draw_mean": {metric: float(np.mean([summarize(s)[metric]["mean"] for s in single[:k]]))
                                 for metric in ("relative_l2", "edge_relative_l2", "gradient_relative_l2", "masked_mse")},
            "batch_one_single_draw_seconds": seconds,
            "estimated_serial_ensemble_seconds": seconds * k,
            "mean_design_variance": float(np.mean([r["valid_pixel_variance"] for r in var[k]])) if k > 1 else None,
        }
    local = data.local(rows[:4])
    np.savez_compressed(args.output / "fixed_examples.npz", row_indices=rows[:4], draws=draws,
                        sample_seeds=np.arange(args.seed_start, args.seed_start + max(args.sizes)),
                        unbiased_variance=maps, **{key: data.arrays[key][local] for key in ("u", "mask", "y_over_h", "z_over_h")})
    dump(args.output / "metrics.json", result)
    dump(args.output / "complete.json", {"status": "complete", "smoke_test_only": args.smoke})
    print("DRAW_STUDY_COMPLETE", str(args.output), flush=True)


if __name__ == "__main__":
    main()
