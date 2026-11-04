#!/usr/bin/env python3
"""PCA + Gaussian-process baseline for the field-prediction task (predict u from x).

PCA is fit on the training images; one GP per coefficient, each with an ARD squared-exponential
kernel, maps x to the standardized coefficients. Hyperparameters are fit on a random subset of the
training designs, then each GP is fit on all training designs with them fixed. A 5-nearest-neighbour
regressor on the same coefficients is reported as a sanity check.

Metrics are computed on u over unmasked pixels, per design, then summarized:
  rel_l2       ||u_hat - u|| / ||u||
  rel_l2_edge  the same over unmasked pixels within 3 pixels of the mask
  rel_grad     relative error of the finite-difference gradient (both directions, both channels),
               over pixel pairs where both pixels are unmasked
The PCA floor is the same metric for the true u projected onto K components (no GP).

Usage: python3 pca_gp.py [--src em2d_task.h5] [--k 32] [--hp-subset 1500] [--out pca_gp.json]
Needs numpy, scipy, h5py and scikit-learn. Run it on a compute node with about 8 CPUs and 32 GB;
it takes about 25 minutes. By default it reads em2d_task.h5 from its own folder (or the repository's
dataset/ folder) and writes pca_gp.json to the current directory.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

HERE = Path(__file__).parent


def edge_band(mask: np.ndarray, width: int = 3) -> np.ndarray:
    """Unmasked pixels within `width` pixels (Chebyshev distance) of the mask, per design."""
    st = np.ones((1, 3, 3), bool)
    return binary_dilation(mask, structure=st, iterations=width) & ~mask


def metrics(u_hat: np.ndarray, u: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    keep = ~mask[:, None]                                  # (N, 1, 64, 128)
    band = edge_band(mask)[:, None]
    d = (u_hat - u)

    def rel(sel):
        num = np.sqrt(((d * sel) ** 2).sum(axis=(1, 2, 3)))
        den = np.sqrt(((u * sel) ** 2).sum(axis=(1, 2, 3)))
        return num / den

    def grads(a):
        return a[..., 1:, :] - a[..., :-1, :], a[..., :, 1:] - a[..., :, :-1]

    gz_h, gy_h = grads(u_hat)
    gz, gy = grads(u)
    kz = keep[..., 1:, :] & keep[..., :-1, :]
    ky = keep[..., :, 1:] & keep[..., :, :-1]
    num = np.sqrt((((gz_h - gz) * kz) ** 2).sum(axis=(1, 2, 3)) + (((gy_h - gy) * ky) ** 2).sum(axis=(1, 2, 3)))
    den = np.sqrt(((gz * kz) ** 2).sum(axis=(1, 2, 3)) + ((gy * ky) ** 2).sum(axis=(1, 2, 3)))
    return {"rel_l2": rel(keep), "rel_l2_edge": rel(band), "rel_grad": num / den}


def summarize(m: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {k: {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.quantile(v, 0.9))}
            for k, v in m.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here_data = HERE / "em2d_task.h5"
    ap.add_argument("--src", default=str(here_data if here_data.exists() else HERE.parent / "dataset/em2d_task.h5"))
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--floor-ks", default="8,16,32,64")
    ap.add_argument("--hp-subset", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pca_gp.json")
    args = ap.parse_args()
    t0 = time.time()

    with h5py.File(args.src, "r") as f:
        x = f["x"][:].astype(np.float64)
        u = f["u"][:]                                      # float32 (N, 2, 64, 128)
        mask = f["mask"][:]
        split = {k: f[f"split/{k}"][:] for k in ("train", "val", "test", "test_extrap")}
    n_pix = u[0].size
    tr = split["train"]
    U = u.reshape(len(u), n_pix).astype(np.float64)
    print(f"loaded {len(u)} designs ({time.time() - t0:.0f} s)", flush=True)

    # PCA on training images
    mean = U[tr].mean(axis=0)
    kmax = max([args.k] + [int(k) for k in args.floor_ks.split(",")])
    _, sv, vt = np.linalg.svd(U[tr] - mean, full_matrices=False)
    comps = vt[:kmax]                                      # (kmax, n_pix)
    var_ratio = (sv ** 2) / (sv ** 2).sum()
    print(f"PCA done ({time.time() - t0:.0f} s); variance explained by {args.k}: "
          f"{var_ratio[:args.k].sum():.6f}", flush=True)

    results = {"k": args.k, "n_train": int(len(tr)),
               "variance_explained": {str(k): float(var_ratio[:int(k)].sum()) for k in args.floor_ks.split(",")},
               "pca_floor": {}, "pca_gp": {}}

    # PCA floor: project the true images
    for k in [int(k) for k in args.floor_ks.split(",")]:
        results["pca_floor"][str(k)] = {}
        for name in ("test", "test_extrap"):
            idx = split[name]
            c = (U[idx] - mean) @ comps[:k].T
            rec = (c @ comps[:k] + mean).reshape(u[idx].shape) * ~mask[idx][:, None]
            results["pca_floor"][str(k)][name] = summarize(metrics(rec, u[idx].astype(np.float64), mask[idx]))
    print(f"PCA floors done ({time.time() - t0:.0f} s)", flush=True)

    # GP from x to standardized coefficients
    C = (U[tr] - mean) @ comps[:args.k].T
    c_mu, c_sd = C.mean(axis=0), C.std(axis=0)
    Cs = (C - c_mu) / c_sd
    # One GP per coefficient: a shared kernel across all coefficients collapsed to "all noise",
    # because the harder high-order coefficients dominate the joint likelihood.
    rng = np.random.default_rng(args.seed)
    sub = rng.choice(len(tr), size=min(args.hp_subset, len(tr)), replace=False)
    gps, kernels = [], []
    for j in range(args.k):
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=np.ones(4), length_scale_bounds=(5e-2, 5e1)) \
            + WhiteKernel(1e-4, (1e-8, 1e0))
        hp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, random_state=args.seed)
        hp.fit(x[tr][sub], Cs[sub, j])
        gp = GaussianProcessRegressor(kernel=hp.kernel_, optimizer=None)
        gp.fit(x[tr], Cs[:, j])
        gps.append(gp)
        kernels.append(str(hp.kernel_))
        print(f"  coefficient {j:2d}: {hp.kernel_} ({time.time() - t0:.0f} s)", flush=True)
    results["gp_kernels"] = kernels

    # sanity check: distance-weighted k-nearest neighbours on the same coefficients
    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors=5, weights="distance").fit(x[tr], Cs)
    results["knn5"] = {}

    for name in ("val", "test", "test_extrap"):
        idx = split[name]
        ts = time.time()
        c_hat = np.stack([g.predict(x[idx]) for g in gps], axis=1) * c_sd + c_mu
        u_hat = (c_hat @ comps[:args.k] + mean).reshape(u[idx].shape) * ~mask[idx][:, None]
        dt = (time.time() - ts) / len(idx)
        results["pca_gp"][name] = summarize(metrics(u_hat, u[idx].astype(np.float64), mask[idx]))
        results["pca_gp"][name]["ms_per_design"] = 1e3 * dt
        c_knn = knn.predict(x[idx]) * c_sd + c_mu
        u_knn = (c_knn @ comps[:args.k] + mean).reshape(u[idx].shape) * ~mask[idx][:, None]
        results["knn5"][name] = summarize(metrics(u_knn, u[idx].astype(np.float64), mask[idx]))
    results["wall_s"] = time.time() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(json.dumps({"pca_gp": results["pca_gp"], "knn5": results["knn5"], "pca_floor": results["pca_floor"].get(str(args.k))}, indent=1))
    print(f"wrote {out} ({results['wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
