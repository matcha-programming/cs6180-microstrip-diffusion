"""Render data previews and compatibility diagnostics from verified artifacts."""
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path("data/palace-pilot")
manifest = json.loads((root / "cases.json").read_text())
audit = json.loads((root / "dataset-verification.json").read_text())
gate = json.loads((root / "quality-gate.json").read_text())
assert audit["status"] == "passed"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 160})

# Fixed row indices, independent of model error or compatibility scores.
selected = [0, 85, 170]
fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("#bdbdbd")
with h5py.File(root / "palace-pilot.h5") as data:
    for column, row in enumerate(selected):
        h, wh, t, er = data["analysis/params_raw"][row]
        for channel in range(2):
            ax = axes[channel, column]
            field = np.ma.masked_array(data["u"][row, channel], data["mask"][row])
            mesh = ax.pcolormesh(data["y_over_h"][row], data["z_over_h"][row], field,
                                 shading="auto", cmap=cmap, vmin=-1, vmax=1)
            ax.set(xlabel="y / h", ylabel="z / h")
            if channel == 0:
                ax.set_title(f"new-{row:04d}\nh={h:.3f} mm, w/h={wh:.2f}, t/h={t/h:.3f}, εr={er:.2f}", fontsize=9)
        axes[0, column].text(.02, .95, "Ey / s", transform=axes[0, column].transAxes, va="top")
        axes[1, column].text(.02, .95, "Ez / s", transform=axes[1, column].transAxes, va="top")
fig.colorbar(mesh, ax=axes, label="Normalized field component", shrink=.8)
fig.suptitle("Palace samples · fixed rows 0, 85, 170 · conductor mask in gray")
fig.savefig(root / "sample-fields.png")
plt.close(fig)

references = [c for c in manifest["cases"] if c["kind"] == "reference"]
refinements = [c for c in manifest["cases"] if c["kind"] == "refinement"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
axes[0].scatter([c["raw"][2] / c["raw"][0] for c in references],
                np.asarray(audit["reference_errors"]) * 100, color="#226b87")
axes[0].set(title=f'24 training references · mean {100*gate["reference_mean"]:.2f}% · p95 {100*gate["reference_p95"]:.2f}%', xlabel="Trace thickness / substrate height (t/h)",
            ylabel="Raw relative L2 (%)")
axes[0].title.set_fontsize(10)
axes[1].scatter([c["raw"][2] / c["raw"][0] for c in refinements],
                np.asarray(audit["refinement_errors"]) * 100, color="#9b571a")
axes[1].axhline(2, color="#a52a2a", linestyle="--", label="2% refinement limit")
axes[1].set(title="Eight mesh-refinement pairs", xlabel="Trace thickness / substrate height (t/h)",
            ylabel="Coarse-to-fine relative L2 (%)")
axes[1].legend(fontsize=8)
for ax in axes:
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.18)
fig.suptitle("Compatibility gate passed" if gate["passed"] else
             "Compatibility gate failed · keep synthetic data separate")
fig.savefig(root / "compatibility-diagnostics.png")
plt.close(fig)
