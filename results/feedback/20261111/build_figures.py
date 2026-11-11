"""Rebuild figures from the verified JSON evidence in this directory."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / "screen-report/summary.json").read_text())["primary"]
labels = [row["run"].replace("-", " ") for row in rows]
colors = ["#176c69" if row["method"] == "regression" else
          "#99581b" if row["method"] == "flow" else "#4778a5" for row in rows]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
y = np.arange(len(rows))
for ax, key, scale, title, fmt in [
    (axes[0], "masked_mse_global", 1e5, "Global masked MSE (×10⁻⁵) · primary", ".2f"),
    (axes[1], "relative_l2", 100, "Mean relative L2 (%) · secondary", ".2f"),
]:
    values = [row[key] * scale for row in rows]
    ax.barh(y, values, color=colors, height=.62)
    ax.set_xlabel(title)
    ax.set_xlim(0, max(values) * 1.17)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.18)
    for pos, value in zip(y, values):
        ax.text(value + max(values) * .02, pos, format(value, fmt), va="center", fontsize=9)
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
fig.suptitle("100-epoch training screen · 881 validation designs", fontsize=14)
fig.text(.5, .015, "One training seed; regression: 1 pass; flow/diffusion: 20 evaluations, mean of 3 single-draw scores.",
         ha="center", fontsize=9)
fig.tight_layout(rect=(0, .05, 1, .96))
fig.savefig(ROOT / "validation-comparison.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
for row in rows:
    history = [json.loads(line) for line in (ROOT / "screen" / row["run"] / "history.jsonl").read_text().splitlines()]
    points = [point for point in history if "field" in point]
    ax.plot([point["epoch"] for point in points],
            [point["field"]["masked_mse_global"] for point in points],
            marker="o", markersize=4, label=row["run"].replace("-", " "))
ax.set(yscale="log", xlabel="Epoch", ylabel="Global masked field MSE (log scale)",
       title="Checkpoint-selection panel · 128 validation designs")
ax.set_xticks([20, 40, 60, 80, 100])
ax.grid(alpha=.18)
ax.legend(fontsize=8, ncol=2)
fig.text(.5, .015, "Panel scores guide checkpoint selection; the main comparison uses all 881 validation designs.",
         ha="center", fontsize=8)
fig.tight_layout(rect=(0, .04, 1, 1))
fig.savefig(ROOT / "validation-panel-history.png")
plt.close(fig)
