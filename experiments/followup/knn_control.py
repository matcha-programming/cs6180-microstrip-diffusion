"""Validation-only distance-weighted field kNN controls; no held-out target reads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

from audit_support import features

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/pca_gp"))
from metric_utils import design_metrics


def weights(distance):
    distance = np.atleast_1d(distance)
    exact = distance <= 1e-12
    w = exact.astype(float) if exact.any() else 1 / distance
    return w / w.sum()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with h5py.File(args.data, "r") as f:
        train, val = f["split/train"][:].astype(int), f["split/val"][:].astype(int)
        raw, x, ids = f["analysis/params_raw"][:], f["x"][:], f["analysis/sample_id"][:]
        allowed = np.sort(np.r_[train, val])
        lookup = np.full(len(x), -1, int)
        lookup[allowed] = np.arange(len(allowed))
        a = {k: f[k][allowed] for k in ("u", "mask", "y_over_h", "z_over_h")}
    result = {"selection_split": "val", "heldout_targets_read": False,
              "candidate_features": ["original_4d", "quasistatic_3d"], "candidate_k": [1, 5, 15],
              "metric_source_sha256": hashlib.sha256((ROOT / "experiments/pca_gp/metric_utils.py").read_bytes()).hexdigest(),
              "candidates": []}
    for space, values in (("original_4d", x.astype(float)), ("quasistatic_3d", features(raw))):
        low, span = values[train].min(0), np.ptp(values[train], axis=0)
        tree = cKDTree((values[train] - low) / span)
        distance, neighbor = tree.query((values[val] - low) / span, k=15)
        for k in (1, 5, 15):
            start, records, squared, count = time.perf_counter(), [], 0., 0
            for j, row in enumerate(val):
                w = weights(distance[j, :k])
                pred = np.einsum("i,iczy->czy", w, a["u"][lookup[train[neighbor[j, :k]]]])
                at = lookup[row]
                mask = a["mask"][at].astype(bool)
                pred[:, mask] = 0
                truth = a["u"][at]
                m = design_metrics(pred, truth, mask, a["y_over_h"][at], a["z_over_h"][at])
                records.append({"row": int(row), "sample_id": int(ids[row]), **m})
                squared += float(((pred - truth)[:, ~mask] ** 2).sum())
                count += 2 * int((~mask).sum())
            name = f"{space}-k{k}"
            (args.output / (name + "-per-design.json")).write_text(json.dumps(records) + "\n")
            candidate = {"feature_space": space, "k": k, "global_masked_mse": squared / count,
                         "seconds_with_metrics": time.perf_counter() - start,
                         "metrics": {key: {"mean": float(np.mean([r[key] for r in records])),
                                            "p95": float(np.quantile([r[key] for r in records], .95))}
                                     for key in ("relative_l2", "edge_relative_l2", "gradient_relative_l2", "masked_mse")},
                         "training_min": low.tolist(), "training_span": span.tolist()}
            result["candidates"].append(candidate)
            print(name, json.dumps(candidate), flush=True)
    result["selected"] = min(result["candidates"], key=lambda v: (v["global_masked_mse"], v["k"], v["feature_space"]))
    (args.output / "validation-selection.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
