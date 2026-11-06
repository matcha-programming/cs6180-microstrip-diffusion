"""Audit fixed split support without reading field targets or changing the split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import Delaunay, cKDTree


def features(raw):
    """Quasi-static similarity hypothesis: (log(w/h), log(t/h), epsilon_r)."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 4 or not np.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError("Expected positive finite h, w/h, t, epsilon_r")
    return np.column_stack((np.log(raw[:, 1]), np.log(raw[:, 2] / raw[:, 0]), raw[:, 3]))


def support(train, query):
    """All scaling and support geometry are fitted using training metadata only."""
    low, high = train.min(0), train.max(0)
    span = high - low
    if (span <= 0).any():
        raise ValueError("Every coordinate must vary in training")
    a, b = (train - low) / span, (query - low) / span
    hull = Delaunay(a)
    distance, neighbor = cKDTree(a).query(b)
    training_distance = cKDTree(a).query(a, k=2)[0][:, 1]
    return {
        "within_training_coordinate_bounds": ((b >= -1e-8) & (b <= 1 + 1e-8)).all(1),
        "inside_training_convex_hull": hull.find_simplex(b, tol=1e-8) >= 0,
        "nearest_training_distance": distance,
        "nearest_training_index": neighbor,
    }, {"training_min": low.tolist(), "training_max": high.tolist(),
        "training_leave_one_out_nn_p95": float(np.quantile(training_distance, .95))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with h5py.File(args.data, "r") as f:
        raw, original = f["analysis/params_raw"][:], f["x"][:]
        ids = f["analysis/sample_id"][:]
        splits = {k: f["split/" + k][:].astype(int) for k in ("train", "val", "test", "test_extrap")}
    joined = np.concatenate(list(splits.values()))
    if not np.array_equal(np.sort(joined), np.arange(len(raw))):
        raise ValueError("Splits must partition all design rows")
    transformed = features(raw)
    args.output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with args.data.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    result = {"dataset_sha256": digest.hexdigest(), "uses_field_targets": False,
              "hypothesis": "Scale invariance of peak-normalized electrostatics under common length scaling; not a verified description of the staff solver.",
              "limitations": "Convex-hull membership measures geometric support, not density, interpolation accuracy, or proof of the physical hypothesis. The official h-held-out split remains unchanged and separately labeled.",
              "spaces": {}}
    records = []
    train_rows = splits["train"]
    for space, values in (("original_4d", original.astype(float)), ("quasistatic_3d", transformed)):
        full, fit = support(values[train_rows], values)
        summary = {"fit": fit, "splits": {}}
        for split, rows in splits.items():
            selected = {k: v[rows] for k, v in full.items()}
            summary["splits"][split] = {
                "count": len(rows),
                "raw_h_min_mm": float(raw[rows, 0].min()), "raw_h_max_mm": float(raw[rows, 0].max()),
                "within_training_bounds_fraction": float(selected["within_training_coordinate_bounds"].mean()),
                "inside_training_convex_hull_fraction": float(selected["inside_training_convex_hull"].mean()),
                "nearest_training_distance_median": float(np.median(selected["nearest_training_distance"])),
                "nearest_training_distance_p95": float(np.quantile(selected["nearest_training_distance"], .95)),
            }
            if split != "train":
                for row in rows:
                    records.append({"space": space, "split": split, "row": int(row), "sample_id": int(ids[row]),
                                    "nearest_training_row": int(train_rows[full["nearest_training_index"][row]]),
                                    **{k: float(v[row]) for k, v in full.items() if k != "nearest_training_index"}})
        result["spaces"][space] = summary
    (args.output / "support.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.output / "per-design.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
