"""Run the unchanged staff baseline and save comparison-compatible artifacts."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket
import sys
import time

import h5py
import numpy as np
import scipy
import sklearn
from metric_utils import design_metrics

STAFF_SHA = "4cf231c86d4270a0a3a6662655e094eabef5d5a409982b183920026a83ae6014"
SPLITS = ("train", "val", "test", "test_extrap")


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def audit_data(path, output):
    with h5py.File(path, "r") as f:
        n = len(f["x"])
        expected = {"x": (n, 4), "u": (n, 2, 64, 128), "mask": (n, 64, 128),
                    "y_over_h": (n, 128), "z_over_h": (n, 64)}
        assert n == 9789
        assert all(f[k].shape == shape for k, shape in expected.items())
        splits = {k: f[f"split/{k}"][:] for k in SPLITS}
        combined = np.concatenate(list(splits.values()))
        assert np.array_equal(np.sort(combined), np.arange(n))
        assert [len(splits[k]) for k in SPLITS] == [7048, 881, 881, 979]
        report = {"shapes": {k: list(f[k].shape) for k in expected}, "splits": {},
                  "finite": True, "masked_targets_zero": True, "increasing_coordinates": True}
        for k, ids in splits.items():
            report["splits"][k] = {"count": len(ids), "indices_sha256": hashlib.sha256(ids.tobytes()).hexdigest()}
        counts, peaks = [], []
        for start in range(0, n, 256):
            a = {k: f[k][start:start + 256] for k in expected}
            assert all(np.isfinite(v).all() for v in a.values())
            assert np.all(np.diff(a["y_over_h"], axis=1) > 0)
            assert np.all(np.diff(a["z_over_h"], axis=1) > 0)
            assert np.all(np.where(a["mask"][:, None], a["u"], 0) == 0)
            counts.extend(a["mask"].sum(axis=(1, 2)).tolist())
            peaks.extend(np.sqrt((a["u"] ** 2).sum(axis=1)).max(axis=(1, 2)).tolist())
        report["masked_pixels_per_design"] = {"min": min(counts), "max": max(counts)}
        report["normalized_peak_magnitude"] = {"min": min(peaks), "max": max(peaks)}
        report["y_grid_varies_between_designs"] = bool(not np.array_equal(f["y_over_h"][0], f["y_over_h"][-1]))
        save_json(output / "data-inspection.json", report)
    print("DATA_AUDIT_PASSED", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    staff_path = here / "staff_pca_gp.py"
    assert hashlib.sha256(staff_path.read_bytes()).hexdigest() == STAFF_SHA
    audit_data(args.data, args.output)
    save_json(args.output / "config.json", {
        "experiment": 1, "model": "Staff PCA-32 + independent ARD RBF Gaussian processes",
        "seed": 0, "k": 32, "hp_subset": 1500, "floor_ks": [8, 16, 32, 64],
        "data": args.data, "hostname": socket.gethostname(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "cpus": os.getenv("SLURM_CPUS_PER_TASK"), "gpu_hours": 0,
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__, "h5py_version": h5py.__version__,
        "staff_sha256": STAFF_SHA,
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metric_utils_sha256": hashlib.sha256((here / "metric_utils.py").read_bytes()).hexdigest(),
        "fit_scope": "PCA mean/components, coefficient scaling, GP hyperparameter subset, final GPs, and kNN use train only",
        "comparison_metric_convention": "Identical to DeepONet run 11597463: Euclidean 3-pixel edge band; forward coordinate gradients; valid endpoints; per-design aggregation",
        "staff_metric_convention": "Chebyshev 3-pixel edge band and pixel-index forward gradients, retained unchanged in staff-results.json"
    })
    spec = importlib.util.spec_from_file_location("staff_pca_gp", staff_path)
    staff = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(staff)
    original_metrics = staff.metrics
    comparison = {"smoke_test_only": False, "k": 32}

    def record_metrics(prediction, truth, mask):
        caller = inspect.currentframe().f_back
        local = caller.f_locals
        # Staff evaluates PCA floors, GP, then kNN. Capture only GP predictions.
        if local.get("u_hat") is prediction:
            name, rows = local["name"], local["idx"]
            with h5py.File(args.data, "r") as f:
                yy, zz = f["y_over_h"][:], f["z_over_h"][:]
                ids = f["analysis/sample_id"][:]
            records = [{"row_index": int(row), "sample_id": int(ids[row]),
                        **design_metrics(prediction[j], truth[j], mask[j], yy[row], zz[row])}
                       for j, row in enumerate(rows)]
            keys = ("relative_l2", "edge_relative_l2", "gradient_relative_l2", "masked_mse")
            comparison[name] = {k: {"mean": float(np.mean([r[k] for r in records])),
                                    "median": float(np.median([r[k] for r in records])),
                                    "std": float(np.std([r[k] for r in records]))} for k in keys}
            comparison[name]["designs"] = len(records)
            save_json(args.output / f"{name}_per_design.json", records)
            np.savez_compressed(args.output / f"{name}_examples.npz", row_indices=rows[:4],
                                prediction=prediction[:4], truth=truth[:4], mask=mask[:4],
                                y_over_h=yy[rows[:4]], z_over_h=zz[rows[:4]])
            if name == "val":
                def predict_one():
                    c = np.stack([g.predict(local["x"][rows[:1]]) for g in local["gps"]], axis=1)
                    return ((c * local["c_sd"] + local["c_mu"]) @ local["comps"][:32] + local["mean"]).reshape(2, 64, 128)
                for _ in range(5):
                    predict_one()
                begin = time.perf_counter()
                for _ in range(20):
                    predict_one()
                comparison["batch_one_prediction_seconds"] = (time.perf_counter() - begin) / 20
            save_json(args.output / "metrics.json", comparison)
            print("COMPARISON_METRICS", name, comparison[name], flush=True)
        del caller
        return original_metrics(prediction, truth, mask)

    staff.metrics = record_metrics
    sys.argv = [str(staff_path), "--src", args.data, "--k", "32", "--hp-subset", "1500",
                "--seed", "0", "--out", str(args.output / "staff-results.json")]
    begin = time.perf_counter()
    staff.main()
    comparison["total_wrapper_seconds"] = time.perf_counter() - begin
    assert all(k in comparison for k in ("val", "test", "test_extrap"))
    save_json(args.output / "metrics.json", comparison)
    print("BASELINE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
