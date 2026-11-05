"""Build figures for the independently audited plain DeepONet continuation."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads((a.run / "config.json").read_text())
    history = [json.loads(s) for s in (a.run / "history.jsonl").read_text().splitlines()]
    epochs = np.array([r["epoch"] for r in history])
    val = np.array([r["validation_masked_mse"] for r in history])
    best = np.array([r["best"] for r in history])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    ax.semilogy(epochs, val, color="#9fb7c7", linewidth=.7, label="Full validation")
    ax.semilogy(epochs, best, color="#096b72", linewidth=1.6, label="Best checkpoint")
    ax.axhline(cfg["original_validation_mse"], color="#9b5335", linestyle="--", linewidth=1,
               label="Published original checkpoint")
    ax.axvspan(1045, 1095, color="#e6b856", alpha=.16)
    ax.axvspan(1095, 1145, color="#e6b856", alpha=.29, label="Final two 50-epoch windows")
    ax.set(xlabel="Epoch (continuation starts after epoch 95)", ylabel="Global masked validation MSE",
           title="Plain DeepONet: validation plateau after 1,050 additional epochs")
    ax.grid(alpha=.15)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(a.output / "learning-curve.png", dpi=180)
    plt.close(fig)
    e = np.load(a.run / "test_examples.npz")
    fig, axes = plt.subplots(4, 3, figsize=(11, 10), constrained_layout=True)
    for i in range(2):
        for channel, cname in enumerate(("Ey", "Ez")):
            row = 2 * i + channel
            truth, pred = e["truth"][i, channel], e["prediction"][i, channel]
            mask = e["mask"][i]
            limit = max(float(np.max(np.abs(truth[~mask]))), float(np.max(np.abs(pred[~mask]))))
            for col, values in enumerate((truth, pred, np.abs(pred - truth))):
                ax = axes[row, col]
                kwargs = {"vmin": 0, "cmap": "magma"} if col == 2 else {"vmin": -limit, "vmax": limit, "cmap": "RdBu_r"}
                im = ax.pcolormesh(e["y_over_h"][i], e["z_over_h"][i], np.ma.array(values, mask=mask),
                                   shading="auto", rasterized=True, **kwargs)
                ax.set(xlabel="y/h", ylabel="z/h", title=f"ID {int(e['row_indices'][i])}, {cname}: " + ("target", "prediction", "absolute error")[col])
                fig.colorbar(im, ax=ax, shrink=.8, pad=.02)
    fig.suptitle("First two fixed test designs · fields in target-normalized units (E/s) · excluded trace pixels masked", fontsize=11)
    fig.savefig(a.output / "test-fixed-examples.png", dpi=160)
    plt.close(fig)
    rows = list(csv.DictReader((a.output / "comparison.csv").open()))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for ax, split, title in zip(axes, ("test", "test_extrap"), ("Test · 881 designs", "Official h holdout · 979 designs")):
        vals = [100 * float(next(r for r in rows if r["method"] == n and r["split"] == split)["relative_l2"]) for n in methods]
        bars = ax.barh(methods, vals, color=["#8395a7", "#bdc8cf", "#096b72", "#5b829e", "#56704d"])
        ax.invert_yaxis()
        ax.bar_label(bars, labels=[f"{v:.2f}%" for v in vals], padding=4, fontsize=9)
        ax.set(xlabel="Mean per-design relative L2 (%)", title=title, xlim=(0, max(vals) * 1.2))
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Same fixed splits and stored targets · historical runs have different training budgets", fontsize=11)
    fig.savefig(a.output / "comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
