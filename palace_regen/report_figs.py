"""Figures (SVG) and tables (JSON) for the Palace recreation report.

Usage: report_figs.py --h5 DATA.h5 --runs RUNS_DIR --out OUT_DIR
Reads the run directories written by regen.py and the staff_u_*.npz files written by
staffpipe.py, and writes interface.svg, maps.svg, rows.svg and tables.json to OUT_DIR.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402
import regen  # noqa: E402

plt.rcParams.update({"svg.fonttype": "none", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 9, "axes.titleweight": "normal"})
ROWS = [0, 4862, 3592, 8053, 52, 1, 500, 8317, 7185]
FIXED = "es_p1_uni1t"
C_DATA, C_DIRECT, C_CHORD, C_TRUE = "#1f2328", "#c2410c", "#2563eb", "#9ca3af"


def best_scale(up, ud, valid):
    return float((up[:, valid] * ud[:, valid]).sum() / (up[:, valid] ** 2).sum())


def load_u(path):
    return np.load(path)["u"] if os.path.exists(path) else None


def staff_path(runs, row, tag):
    return os.path.join(runs, f"row{row:05d}", tag, "staff_u_trace_chord.npz")


def fig_interface(d, prof, u_direct, u_chord, col, path):
    """E_z against z/h at one column away from the trace, around the dielectric interface."""
    E, zf, yv = prof["E"], prof["z"] / d["h"], prof["y"]
    j = int(np.argmin(np.abs(yv - d["y"][col])))
    zg = d["z"] / d["h"]
    ud = d["u"]
    far = np.abs(d["y"]) > d["w"] / 2 + d["h"]
    sub = (zg > 0.05) & (zg < 0.9)
    # scale the converged physical profile onto the data's normalisation using substrate rows
    Ez_rows = np.array([E[1, int(np.argmin(np.abs(zf - z))), j] for z in zg[sub]])
    k = float((ud[1, sub, col] * Ez_rows).sum() / (Ez_rows ** 2).sum())
    valid = ~d["mask"]
    a_dir = best_scale(u_direct, ud, valid)
    a_chd = best_scale(u_chord, ud, valid)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    sel = (zf > 0.6) & (zf < 2.4)
    ax.plot(zf[sel], k * E[1, sel, j], color=C_TRUE, lw=2.2, label="Palace E, converged")
    rows = (zg > 0.6) & (zg < 2.4)
    ax.plot(zg[rows], a_dir * u_direct[1, rows, col], "s", color=C_DIRECT, ms=5, mfc="none",
            label="Palace E at the grid points")
    ax.plot(zg[rows], a_chd * u_chord[1, rows, col], "o", color=C_CHORD, ms=9, mfc="none", mew=1.4,
            label="chord differences of Palace V")
    ax.plot(zg[rows], ud[1, rows, col], "x", color=C_DATA, ms=4, mew=1.2, label="dataset")
    ax.axvline(1.0, color=C_TRUE, ls=":", lw=1)
    for r in (30, 31, 32, 33):
        ax.annotate(f"row {r}", (zg[r], ud[1, r, col]), textcoords="offset points",
                    xytext=(6, -11 if r != 31 else 6), fontsize=8, color=C_DATA)
    ax.set_xlabel("z / h")
    ax.set_ylabel("E_z (dataset units)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return dict(col=col, y_over_h=float(d["y"][col] / d["h"]))


def fig_maps(items, path):
    """For each design: dataset E_z, reproduction E_z, difference, in pixel space."""
    n = len(items)
    fig, axs = plt.subplots(n, 3, figsize=(7.2, 1.55 * n + 0.3), constrained_layout=True)
    axs = np.atleast_2d(axs)
    for i, (d, u, label) in enumerate(items):
        ud = d["u"]
        a = best_scale(u, ud, ~d["mask"])
        lim = np.percentile(np.abs(ud[1]), 99.5)
        diff = a * u[1] - ud[1]
        dl = max(np.percentile(np.abs(diff), 99.5), 1e-6)
        for jj, (img, cmap, vlim, title) in enumerate([
                (ud[1], "RdBu_r", lim, "dataset"), (a * u[1], "RdBu_r", lim, "reproduced"),
                (diff, "PuOr", dl, "difference")]):
            ax = axs[i, jj]
            ax.imshow(np.where(d["mask"], np.nan, img), origin="lower", cmap=cmap, vmin=-vlim,
                      vmax=vlim, aspect="auto", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(title)
            if jj == 2:
                ax.text(0.03, 0.9, f"scale ±{dl:.2g}", transform=ax.transAxes, fontsize=7.5)
            if jj == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.savefig(path)
    plt.close(fig)


def fig_rows(d, u_direct, u_chord, path):
    ud, m = d["u"], d["mask"]

    def per_row(u):
        a = best_scale(u, ud, ~m)
        return np.array([np.linalg.norm((a * u - ud)[:, r, ~m[r]]) / np.linalg.norm(ud[:, r, ~m[r]])
                         for r in range(64)])

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.plot(per_row(u_direct), color=C_DIRECT, lw=1.6, label="Palace E at the grid points")
    ax.plot(per_row(u_chord), color=C_CHORD, lw=1.6, label="chord differences of Palace V")
    ax.axvspan(30.5, 32.5, color=C_TRUE, alpha=0.25, lw=0)
    ax.set_xlabel("grid row (z)")
    ax.set_ylabel("relative error per row")
    ax.set_xlim(0, 63)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def scaled_without_worst(u, d, k=3):
    """Scaled relative error after dropping the k pixels with the largest squared error."""
    ud, m = d["u"], d["mask"]
    a = best_scale(u, ud, ~m)
    e2 = ((a * u - ud) ** 2).sum(0)
    e2[m] = 0
    worst = np.argsort(e2.ravel())[::-1][:k]
    keep = np.ones(e2.size, bool)
    keep[worst] = False
    keep &= ~m.ravel()
    ref = (ud ** 2).sum(0).ravel()[~m.ravel()].sum()
    pix = [divmod(int(i), e2.shape[1]) for i in worst]
    return float(np.sqrt(e2.ravel()[keep].sum() / ref)), pix, float(e2.ravel()[worst].sum() / e2.sum())


def signature(u, m):
    a = u.astype(np.float32)
    out = {}
    for name, rows in (("ground", slice(0, 11)), ("air", slice(34, 54))):
        for c, lab in ((0, "Ey"), (1, "Ez")):
            x = a[c, rows]
            mm = m[rows]
            pair = (x[:, 1:] == x[:, :-1]) & ~mm[:, 1:] & ~mm[:, :-1] & (x[:, 1:] != 0)
            out[f"{name}_{lab}"] = float(pair.mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    R = args.runs
    D = {r: regen.load_row(args.h5, r) for r in ROWS}
    import h5py
    with h5py.File(args.h5, "r") as f:
        split = {}
        for name in ("train", "val", "test", "test_extrap"):
            for i in f["split"][name][:]:
                split[int(i)] = name
    tables = {}

    # 1. staff-pipeline reproduction across mesh variants
    variants = {}
    for fn in glob.glob(os.path.join(R, "scan*", "row*", "*", "staff_u_trace_chord.npz")):
        row = int(fn.split(os.sep)[-3][3:])
        tag = fn.split(os.sep)[-2]
        if row in D:
            variants.setdefault(tag, {})[row] = compare.metrics(np.load(fn)["u"], D[row])
    mesh_rows = []
    for tag, by in sorted(variants.items()):
        common = [r for r in ROWS if r in by]
        mesh_rows.append(dict(tag=tag, n=len(common), scaled={r: by[r]["rel_all_scaled"] for r in common},
                              raw={r: by[r]["rel_all"] for r in common}))
    tables["mesh_scan"] = mesh_rows
    # best variant per design by scaled error
    best = {}
    for r in ROWS:
        cand = [(by[r]["rel_all_scaled"], tag) for tag, by in variants.items() if r in by]
        if cand:
            best[r] = min(cand)[1]

    # 2. per-design table
    design_rows = []
    for r in ROWS:
        d = D[r]
        u_dir = load_u(os.path.join(R, f"row{r:05d}", "es_p3_ref0.5", "fields.npz"))
        m_dir = compare.metrics(u_dir, d) if u_dir is not None else None
        m_chd = variants[best[r]][r]
        sub = "scan" if best[r].startswith("es_p1_uni") else "scan2"
        u_best = np.load(staff_path(os.path.join(R, sub), r, best[r]))["u"]
        e3, pix3, share3 = scaled_without_worst(u_best, d)
        # one fixed recipe for every design: uniform elements of about t
        u_fix = np.load(staff_path(os.path.join(R, "scan"), r, FIXED))["u"]
        m_fix = compare.metrics(u_fix, d)
        f3, fpix3, fshare3 = scaled_without_worst(u_fix, d)
        design_rows.append(dict(
            row=r, sid=d["sid"], split=split.get(r), t_over_h=d["t"] / d["h"],
            w_over_h=d["w"] / d["h"], eps_r=d["er"], h_mm=d["h"],
            direct_scaled=m_dir["rel_all_scaled"] if m_dir else None,
            direct_raw=m_dir["rel_all"] if m_dir else None,
            best_mesh=best[r], chord_scaled=m_chd["rel_all_scaled"], chord_raw=m_chd["rel_all"],
            chord_far=m_chd["rel_far"], chord_band=m_chd["rel_band"], chord_scaled_drop3=e3,
            worst3=pix3, worst3_share=share3,
            fixed_raw=m_fix["rel_all"], fixed_scaled=m_fix["rel_all_scaled"], fixed_drop3=f3,
            fixed_far=m_fix["rel_far"], fixed_band=m_fix["rel_band"], fixed_worst3=fpix3,
            fixed_worst3_share=fshare3, ratio_fixed=m_fix["ez_ratio_31_30"]["palace"],
            ratio_data=m_chd["ez_ratio_31_30"]["data"], ratio_chord=m_chd["ez_ratio_31_30"]["palace"],
            ratio_direct=m_dir["ez_ratio_31_30"]["palace"] if m_dir else None,
            flips_data=m_chd["flips_under_trace"]["data"], flips_chord=m_chd["flips_under_trace"]["palace"]))
    tables["designs"] = design_rows

    # 3. solver variants, direct E on graded meshes
    solver = {}
    for tag in ("es_p1_ref1", "es_p3_ref0.5", "bm1GHz_p1_ref1", "bm10GHz_p1_ref1", "bm30GHz_p1_ref1",
                "bm10GHz_p3_ref0.5"):
        vals = {}
        for r in ROWS:
            u = load_u(os.path.join(R, f"row{r:05d}", tag, "fields.npz"))
            if u is not None:
                vals[r] = compare.metrics(u, D[r])["rel_all_scaled"]
        solver[tag] = vals
    tables["solvers"] = solver

    # 4. texture signatures
    sig = {}
    for r in (7185, 500):
        d = D[r]
        s = dict(data=signature(d["u"], d["mask"]))
        for tag in ("es_p1_uni1t", "es_p1_uni2t"):
            ud_ = load_u(os.path.join(R, "scan", f"row{r:05d}", tag, "fields.npz"))
            uc_ = load_u(staff_path(os.path.join(R, "scan"), r, tag))
            if ud_ is not None:
                s[f"direct_{tag}"] = signature(ud_, d["mask"])
            if uc_ is not None:
                s[f"chord_{tag}"] = signature(uc_, d["mask"])
        sig[r] = s
    tables["signatures"] = sig

    # figures
    d0 = D[0]
    tagf = "es_p1_uni1t"
    u_dir0 = load_u(os.path.join(R, "scan", "row00000", tagf, "fields.npz"))
    u_chd0 = load_u(staff_path(os.path.join(R, "scan"), 0, tagf))
    prof = np.load(os.path.join(R, "profiles", "row00000_es_p3_full", "profile.npz"))
    tables["interface_fig"] = fig_interface(d0, prof, u_dir0, u_chd0, 20,
                                            os.path.join(args.out, "interface.svg"))
    fig_rows(d0, u_dir0, u_chd0, os.path.join(args.out, "rows.svg"))
    items = []
    for r in (0, 8053, 500):
        rd = os.path.join(R, "scan" if best[r].startswith("es_p1_uni") else "scan2", f"row{r:05d}",
                          best[r])
        items.append((D[r], np.load(os.path.join(rd, "staff_u_trace_chord.npz"))["u"],
                      f"t/h {D[r]['t'] / D[r]['h']:.3f}"))
    fig_maps(items, os.path.join(args.out, "maps.svg"))
    with open(os.path.join(args.out, "tables.json"), "w") as fh:
        json.dump(tables, fh, indent=1, default=float)
    print(json.dumps({k: (v if k != "mesh_scan" else len(v)) for k, v in tables.items()
                      if k in ("interface_fig", "mesh_scan")}))


if __name__ == "__main__":
    sys.exit(main())
