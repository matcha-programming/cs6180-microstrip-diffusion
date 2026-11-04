"""Put a Palace electrostatic solution on the dataset grid: exact interpolation of the order-1
potential V from Palace's ParaView output ("Electrostatic": {"Save": 1}), then E = -grad V by
finite differences on the grid.
"""

import glob
import os

import numpy as np


def grid_xy(d):
    Z, Y = np.meshgrid(d["z"], d["y"], indexing="ij")
    return np.c_[Y.ravel(), Z.ravel()]


def potential_on_grid(run_dir, d, inside=np.nan, k=16):
    """Palace's order-1 potential V interpolated exactly at the grid points.

    Reads the ParaView output, finds the triangle holding each point (k nearest centroids,
    barycentric test) and interpolates linearly. vtkProbeFilter is not used: with VTK 9.6.1
    it silently misses some points that lie well inside Palace's Lagrange-triangle cells and
    returns 0 for them. Points in no triangle (inside the trace) get `inside`.
    Returns V and the mask of points found."""
    import vtk
    from scipy.spatial import cKDTree
    from vtk.util.numpy_support import vtk_to_numpy
    pv = sorted(glob.glob(os.path.join(run_dir, "postpro", "paraview", "electrostatic",
                                       "Cycle000001", "data.pvtu")))
    assert pv, f"no electrostatic ParaView output under {run_dir}"
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(pv[0])
    r.Update()
    g = r.GetOutput()
    P = vtk_to_numpy(g.GetPoints().GetData())[:, :2].astype(np.float64)
    Vp = vtk_to_numpy(g.GetPointData().GetArray("V")).astype(np.float64)
    off = vtk_to_numpy(g.GetCells().GetOffsetsArray())
    assert np.all(np.diff(off) == 3), "expected order-1 triangles (Solver.Order 1)"
    T = vtk_to_numpy(g.GetCells().GetConnectivityArray()).reshape(-1, 3)
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    tree = cKDTree((A + B + C) / 3)
    # The VTU stores float32 coordinates, so its walls sit up to ~1e-7 h inside the true
    # domain; clamp the grid to the stored bounds so wall points stay inside their cells.
    xy = grid_xy(d)
    lo, hi = P.min(0), P.max(0)
    xy = np.clip(xy, lo, hi)
    _, cand = tree.query(xy, k=k)
    a, b, c = A[cand], B[cand], C[cand]
    v0, v1, v2 = b - a, c - a, xy[:, None, :] - a
    den = v0[..., 0] * v1[..., 1] - v1[..., 0] * v0[..., 1]
    l1 = (v2[..., 0] * v1[..., 1] - v1[..., 0] * v2[..., 1]) / den
    l2 = (v0[..., 0] * v2[..., 1] - v2[..., 0] * v0[..., 1]) / den
    l0 = 1 - l1 - l2
    score = np.minimum(np.minimum(l0, l1), l2)
    j = np.argmax(score, axis=1)
    rows = np.arange(len(xy))
    ok = score[rows, j] > -1e-6  # points on edges and walls count as inside
    Z, Y = np.meshgrid(d["z"], d["y"], indexing="ij")
    in_trace = ((np.abs(Y) < d["w"] / 2) & (Z > d["h"]) & (Z < d["h"] + d["t"])).ravel()
    assert not (~ok & ~in_trace).any(), "grid points outside the trace not located in the mesh"
    tri = T[cand[rows, j]]
    lam = np.stack([l0[rows, j], l1[rows, j], l2[rows, j]], axis=1)
    V = np.where(ok, (lam * Vp[tri]).sum(1), inside)
    shape = d["u"].shape[1:]
    return V.reshape(shape), ok.reshape(shape)


def gradient_field(V, d, scheme="chord"):
    """E = -grad V by finite differences on the (non-uniform) dataset grid.
    chord:      (V[i+1] - V[i-1]) / (x[i+1] - x[i-1]), one-sided at the ends, which is
                np.gradient(V) / np.gradient(x)
    nonuniform: np.gradient(V, x), the second-order formula for uneven spacing"""
    if scheme == "chord":
        dVdz = np.gradient(V, axis=0) / np.gradient(d["z"])[:, None]
        dVdy = np.gradient(V, axis=1) / np.gradient(d["y"])[None, :]
    else:
        dVdz, dVdy = np.gradient(V, d["z"], d["y"])
    return np.stack([-dVdy, -dVdz])
