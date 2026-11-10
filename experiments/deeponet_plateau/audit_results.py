"""Audit a completed continuation against the fixed HDF5 and reproduce it on CPU."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("plain", ROOT / "experiments/deeponet_plain/train.py")
plain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plain)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run = args.run
    cfg = json.loads((run / "config.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    hist = [json.loads(s) for s in (run / "history.jsonl").read_text().splitlines()]
    assert not cfg["smoke_test_only"] and not metrics["smoke_test_only"]
    assert sha(args.data) == cfg["data_sha256"]
    assert sha(ROOT / "experiments/deeponet_plain/train.py") == cfg["original_source_sha256"]
    assert sha(ROOT / "experiments/deeponet_plateau/train.py") == cfg["continuation_source_sha256"]
    assert sha(ROOT / "results/deeponet_plain/11597463/best.pt") == cfg["initial_checkpoint_sha256"]
    assert [r["epoch"] for r in hist] == list(range(96, 1146))
    best, anchor, streak = cfg["original_validation_mse"], cfg["original_validation_mse"], 0
    windows = []
    for i, r in enumerate(hist, 1):
        assert r["new_epochs"] == i and np.isfinite(r["validation_masked_mse"])
        best = min(best, r["validation_masked_mse"])
        assert np.isclose(best, r["best"], rtol=1e-12)
        if i % 50 == 0:
            gain = (anchor - best) / anchor
            streak = streak + 1 if gain < .01 else 0
            windows.append({"epoch": r["epoch"], "relative_gain": gain, "streak": streak})
            assert np.isclose(gain, r["window_relative_gain"], rtol=1e-12, atol=1e-15)
            anchor = best
        assert r["streak"] == streak
        assert streak < 2 or i == len(hist), "Training continued after its stopping rule"
    assert streak == 2 and metrics["stop_reason"] == "validation_plateau"
    assert metrics["best_epoch"] == hist[int(np.argmin([r["validation_masked_mse"] for r in hist]))]["epoch"]
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == metrics["best_epoch"]
    assert np.isclose(ckpt["validation_masked_mse"], best, rtol=1e-12)
    model = plain.DeepONet(cfg["width"], cfg["rank"]).eval()
    model.load_state_dict(ckpt["model"])
    torch.set_num_threads(4)
    result = {"status": "passed", "dataset_sha256": sha(args.data),
              "best_checkpoint_sha256": sha(run / "best.pt"), "history_records": len(hist),
              "tolerances": {"per_design_metric_rtol": .002, "per_design_metric_atol": 1e-5,
                  "example_prediction_atol": 1e-5, "validation_global_mse_rtol": .001,
                  "aggregate_rtol": 1e-12, "aggregate_atol": 1e-15},
              "plateau_windows": windows, "full_cpu_reproduction": {}, "example_checks": {}}
    keys = ("relative_l2", "edge_relative_l2", "gradient_relative_l2", "masked_mse")
    with h5py.File(args.data, "r") as f, torch.inference_mode():
        x, ys, zs = (f[k][:] for k in ("x", "y_over_h", "z_over_h"))
        source_ids = f["analysis/sample_id"][:]
        for split, count in (("val", 881), ("test", 881), ("test_extrap", 979)):
            rows = f["split/" + split][:].astype(int)
            assert len(rows) == count and len(set(rows)) == count
            records = json.loads((run / (split + "_per_design.json")).read_text()) if split != "val" else None
            if records:
                assert [r["row_index"] for r in records] == rows.tolist()
                assert [r["sample_id"] for r in records] == source_ids[rows].tolist()
                for key in keys:
                    values = np.array([r[key] for r in records])
                    assert np.isfinite(values).all() and (values >= 0).all()
                    for name, fn in (("mean", np.mean), ("median", np.median), ("std", np.std)):
                        assert np.isclose(fn(values), metrics[split][key][name], rtol=1e-12, atol=1e-15)
            example = np.load(run / (split + "_examples.npz")) if records else None
            if example:
                assert np.array_equal(example["row_indices"], rows[:4])
                for j, row in enumerate(rows[:4]):
                    for arr, dataset in (("truth", "u"), ("mask", "mask"), ("y_over_h", "y_over_h"), ("z_over_h", "z_over_h")):
                        assert np.array_equal(example[arr][j], f[dataset][row])
            numerator, denominator, max_metric_diff, max_pred_diff = 0., 0, 0., 0.
            for begin in range(0, count, 4):
                rr = rows[begin:begin + 4]
                y = torch.as_tensor(ys[rr])[:, None, :].expand(-1, 64, -1)
                z = torch.as_tensor(zs[rr])[:, :, None].expand(-1, -1, 128)
                coords = torch.stack((y, z), dim=-1).flatten(1, 2)
                predictions = model(torch.as_tensor(x[rr]), coords).reshape(-1, 2, 64, 128).numpy()
                assert np.isfinite(predictions).all()
                for j, row in enumerate(rr):
                    truth, mask = f["u"][row], f["mask"][row].astype(bool)
                    valid = ~mask
                    numerator += np.square((predictions[j] - truth)[:, valid], dtype=np.float64).sum()
                    denominator += 2 * int(valid.sum())
                    if records:
                        got = plain.design_metrics(predictions[j], truth, mask, ys[row], zs[row])
                        for key in keys:
                            expected = records[begin + j][key]
                            max_metric_diff = max(max_metric_diff, abs(got[key] - expected))
                            assert np.isclose(got[key], expected, rtol=.002, atol=1e-5), (split, int(row), key)
                        if begin + j < 4:
                            max_pred_diff = max(max_pred_diff, float(np.max(abs(predictions[j] - example["prediction"][begin + j]))))
            reproduced = numerator / denominator
            if split == "val":
                assert np.isclose(reproduced, best, rtol=.001)
            result["full_cpu_reproduction"][split] = {"designs": count, "valid_component_values": denominator,
                "global_masked_mse": reproduced, "max_per_design_metric_absolute_difference": max_metric_diff}
            if example:
                assert max_pred_diff < 1e-5
                result["example_checks"][split] = {"examples": 4, "mask_truth_grid_match": True,
                    "max_prediction_absolute_difference": max_pred_diff}
            print(split, result["full_cpu_reproduction"][split], flush=True)
    (args.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    comparisons = [("PCA-32 + GP", ROOT / "results/pca_gp/11601747/metrics.json"),
                   ("Plain DeepONet, original", ROOT / "results/deeponet_plain/11597463/metrics.json"),
                   ("Plain DeepONet, plateau", run / "metrics.json"),
                   ("Fourier DeepONet, round one", ROOT / "results/remaining/11603443/fourier-metrics.json"),
                   ("4R PositionUNet", ROOT / "results/unet_regression/11604740/regression-metrics.json")]
    with (args.output / "comparison.csv").open("w") as out:
        writer = csv.DictWriter(out, fieldnames=["method", "split", *keys], lineterminator="\n")
        writer.writeheader()
        for name, path in comparisons:
            m = json.loads(path.read_text())
            for split in ("test", "test_extrap"):
                writer.writerow({"method": name, "split": split, **{k: m[split][k]["mean"] for k in keys}})


if __name__ == "__main__":
    main()
