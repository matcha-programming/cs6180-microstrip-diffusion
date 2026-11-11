"""Re-simulate em2d_task.h5 designs with Palace.

For each dataset row: build a 2-D Gmsh mesh of the microstrip cross-section, write a Palace
config whose probes are the row's 64 x 128 sampling grid, run Palace, and save Palace's own E
field at those points in the dataset's u convention (channels E_y, E_z; rows z, columns y;
u = E / max |E| over unmasked pixels; masked pixels 0) as fields.npz. That is not yet what the
dataset stores: run with --save 1 and pass the run to staffpipe.py to reproduce the targets.

Geometry, in mm (the dataset's h, w = (w/h) h and t, from /analysis/params_raw):
  window  y in [-(w/2 + 6h), w/2 + 6h], z in [0, 8h], read from y_over_h and z_over_h
  ground  z = 0 (Ground)
  substrate eps_r for 0 < z < h, air above
  trace   w x t rectangle on the substrate, h < z < h + t, cut out of the mesh (Terminal, 1 V)
  walls   sides and top carry no condition (natural, zero normal D)

Usage:
  regen.py --h5 DATA.h5 --palace PALACE_BIN --out RUNS_DIR --rows 0 1 500 [--order 1 2 3]
           [--problem electrostatic|boundarymode] [--freq GHz ...] [--save 1] [--np 4]
           [--ref 1.0 | --uniform C | --points A B]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import h5py
import numpy as np


def load_row(h5, row):
    with h5py.File(h5, "r") as f:
        h, wh, t, er = (float(v) for v in f["analysis/params_raw"][row])
        # The stored float32 y grid is linspace(-(w/2 + 6h), w/2 + 6h, 128); rebuild it in
        # float64 so the end columns sit exactly on the side walls, not 1e-7 outside them.
        y32 = f["y_over_h"][row].astype(np.float64) * h
        Y = wh * h / 2 + 6 * h
        y = np.linspace(-Y, Y, y32.size)
        assert np.abs(y - y32).max() < 1e-6 * Y, np.abs(y - y32).max()
        z = np.clip(f["z_over_h"][row].astype(np.float64) * h, 0.0, 8 * h)
        return dict(
            row=row, sid=int(f["analysis/sample_id"][row]), h=h, w=wh * h, t=t, er=er, y=y, z=z,
            u=f["u"][row].astype(np.float64), mask=f["mask"][row].astype(bool))


def _generate(path):
    """Mesh the current Gmsh model and write it in the format Palace reads; returns triangles."""
    import gmsh
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    ntri = sum(len(e) for e in gmsh.model.mesh.getElements(2)[1])
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 1)
    gmsh.write(path)
    return ntri


def build_mesh(d, path, ref=1.0, verbose=0, uniform=None, points=None):
    """2-D mesh with the trace cut out. Sizes scale with h and t; ref < 1 refines everywhere.
    uniform=c instead gives every element size about c * t (Gmsh point-based sizing), which is
    what the run-length pattern of the dataset suggests the staff mesh looked like.
    points=(a, b) sets Gmsh point sizes a * t at the trace corners and b * t at every other
    corner, interpolated by Gmsh's defaults: fine at the trace, coarse at the walls."""
    import gmsh
    h, w, t = d["h"], d["w"], d["t"]
    Y, H = w / 2 + 6 * h, 8 * h
    lc_corner = ref * min(t, w) / 6
    lc_trace = ref * min(t, h / 10)
    lc_sub = ref * h / 12
    lc_far = ref * h / 2
    gmsh.initialize(["-noenv"])
    try:
        gmsh.option.setNumber("General.Verbosity", verbose)
        gmsh.model.add("microstrip")
        occ = gmsh.model.occ
        sub = occ.addRectangle(-Y, 0, 0, 2 * Y, h)
        air = occ.addRectangle(-Y, h, 0, 2 * Y, H - h)
        tr = occ.addRectangle(-w / 2, h, 0, w, t)
        air_cut, _ = occ.cut([(2, air)], [(2, tr)])
        occ.fragment([(2, sub)], air_cut)
        occ.synchronize()
        surfs = [tag for _, tag in gmsh.model.getEntities(2)]
        sub_s = [s for s in surfs if occ.getCenterOfMass(2, s)[1] < h]
        air_s = [s for s in surfs if s not in sub_s]
        eps = 1e-3 * min(t, w, h)  # OCC bounding boxes are padded by ~1e-7
        ground, trace, outer = [], [], []
        for _, c in gmsh.model.getEntities(1):
            x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, c)
            if y1 < eps:
                ground.append(c)
            elif x0 > -w / 2 - eps and x1 < w / 2 + eps and y0 > h - eps and y1 < h + t + eps:
                trace.append(c)
            elif x1 < -Y + eps or x0 > Y - eps or y0 > H - eps:
                outer.append(c)
        assert len(sub_s) == 1 and len(air_s) == 1 and len(ground) == 1, (sub_s, air_s, ground)
        assert len(trace) == 4 and len(outer) == 5, (trace, outer)  # sides split at z = h
        gmsh.model.addPhysicalGroup(2, sub_s, 1, "substrate")
        gmsh.model.addPhysicalGroup(2, air_s, 2, "air")
        gmsh.model.addPhysicalGroup(1, ground, 3, "ground")
        gmsh.model.addPhysicalGroup(1, trace, 4, "trace")
        gmsh.model.addPhysicalGroup(1, outer, 5, "outer")
        corners = [p for _, p in gmsh.model.getEntities(0)
                   if abs(abs(gmsh.model.getValue(0, p, [])[0]) - w / 2) < eps]
        if points is not None:
            a, b = points
            for _, p in gmsh.model.getEntities(0):
                x, y, _ = gmsh.model.getValue(0, p, [])
                on_trace = abs(abs(x) - w / 2) < eps and h - eps < y < h + t + eps
                gmsh.model.mesh.setSize([(0, p)], (a if on_trace else b) * t)
            return _generate(path)
        if uniform is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMin", uniform * t)
            gmsh.option.setNumber("Mesh.MeshSizeMax", uniform * t)
            return _generate(path)
        fld = gmsh.model.mesh.field
        fld.add("Distance", 1)
        fld.setNumbers(1, "PointsList", corners)
        fld.add("Threshold", 2)
        fld.setNumber(2, "InField", 1)
        fld.setNumber(2, "SizeMin", lc_corner)
        fld.setNumber(2, "SizeMax", lc_far)
        fld.setNumber(2, "DistMin", 0.5 * lc_corner)
        fld.setNumber(2, "DistMax", 4 * h)
        fld.add("Distance", 3)
        fld.setNumbers(3, "CurvesList", trace)
        fld.setNumber(3, "Sampling", 400)
        fld.add("Threshold", 4)
        fld.setNumber(4, "InField", 3)
        fld.setNumber(4, "SizeMin", lc_trace)
        fld.setNumber(4, "SizeMax", lc_far)
        fld.setNumber(4, "DistMin", lc_trace)
        fld.setNumber(4, "DistMax", 5 * h)
        fld.add("Box", 5)
        fld.setNumber(5, "VIn", lc_sub)
        fld.setNumber(5, "VOut", lc_far)
        fld.setNumber(5, "XMin", -Y - 1)
        fld.setNumber(5, "XMax", Y + 1)
        fld.setNumber(5, "YMin", -1)
        fld.setNumber(5, "YMax", h)
        fld.setNumber(5, "Thickness", 2 * h)
        fld.add("Min", 6)
        fld.setNumbers(6, "FieldsList", [2, 4, 5])
        fld.setAsBackgroundMesh(6)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        return _generate(path)
    finally:
        gmsh.finalize()


def probe_points(d):
    """All grid points except those strictly inside the trace; returns (flat indices, xy)."""
    Z, Yg = np.meshgrid(d["z"], d["y"], indexing="ij")
    tol = 1e-9 * d["h"]
    inside = (np.abs(Yg) < d["w"] / 2 - tol) & (Z > d["h"] + tol) & (Z < d["h"] + d["t"] - tol)
    keep = np.flatnonzero(~inside.ravel())
    return keep, np.c_[Yg.ravel()[keep], Z.ravel()[keep]]


def eps_eff(d):
    """Hammerstad quasi-static effective permittivity (zero thickness), for the mode target."""
    u, er = d["w"] / d["h"], d["er"]
    a = (1 + 12 / u) ** -0.5 + (0.04 * (1 - u) ** 2 if u < 1 else 0.0)
    return (er + 1) / 2 + (er - 1) / 2 * a


def write_config(d, mesh, outdir, keep, xy, order, path, problem="electrostatic", freq=10.0,
                 save=0):
    # The schema wants three coordinates; a 2-D mesh uses the first two.
    probes = [{"Index": int(k) + 1, "Center": [float(a), float(b), 0.0]}
              for k, (a, b) in zip(keep, xy)]
    cfg = {
        "Problem": {"Type": "Electrostatic", "Verbose": 1, "Output": outdir},
        "Model": {"Mesh": mesh, "L0": 1.0e-3},
        "Domains": {
            "Materials": [{"Attributes": [1], "Permittivity": d["er"]},
                          {"Attributes": [2], "Permittivity": 1.0}],
            "Postprocessing": {"Probe": probes}},
        "Boundaries": {"Ground": {"Attributes": [3]},
                       "Terminal": [{"Index": 1, "Attributes": [4]}]},
        "Solver": {"Order": order, "Device": "CPU", "Electrostatic": {"Save": save},
                   "Linear": {"Type": "BoomerAMG", "KSPType": "CG", "Tol": 1.0e-12,
                              "MaxIts": 1000}}}
    if problem == "boundarymode":
        # 2-D mode analysis: PEC trace and ground, natural (PMC) sides and top.
        cfg["Problem"]["Type"] = "BoundaryMode"
        cfg["Boundaries"] = {"PEC": {"Attributes": [3, 4]}}
        cfg["Solver"] = {"Order": order, "Device": "CPU",
                         "BoundaryMode": {"Freq": freq, "N": 1, "Save": save,
                                          "Target": float(np.sqrt(eps_eff(d))), "Tol": 1.0e-10}}
    with open(path, "w") as fh:
        json.dump(cfg, fh)


def read_probes(csv_path, keep, shape):
    """probe-E.csv (first solution row) -> complex (E_y, E_z) on the grid; unprobed points 0."""
    with open(csv_path) as fh:
        rows = list(csv.reader(fh))
    head = [c.strip() for c in rows[0]]
    vals = np.array([float(v) for v in rows[1]])
    col = {c: j for j, c in enumerate(head)}
    cplx = "Re{E_x[%d]} (V/m)" % (keep[0] + 1) in col
    E = np.zeros((2,) + shape, complex)
    flat = E.reshape(2, -1)
    for k in keep:
        for c, lab in ((0, "E_x"), (1, "E_y")):
            if cplx:
                flat[c, k] = (vals[col["Re{%s[%d]} (V/m)" % (lab, k + 1)]]
                              + 1j * vals[col["Im{%s[%d]} (V/m)" % (lab, k + 1)]])
            else:
                flat[c, k] = vals[col["%s[%d] (V/m)" % (lab, k + 1)]]
    return E


def realify(E):
    """Rotate a complex mode by the global phase that makes it most nearly real, keep Re, and
    fix the sign so E_z under the trace centre is negative (the dataset's convention)."""
    phi = 0.5 * np.angle((E ** 2).sum())
    Er = (E * np.exp(-1j * phi)).real
    resid = float(np.linalg.norm((E * np.exp(-1j * phi)).imag) / np.linalg.norm(E))
    if Er[1, : Er.shape[1] // 2, Er.shape[2] // 2 - 1: Er.shape[2] // 2 + 1].mean() > 0:
        Er = -Er
    return Er, resid


def normalize(E, mask):
    mag = np.sqrt((E ** 2).sum(0))
    s = mag[~mask].max()
    u = E / s
    u[:, mask] = 0
    return u, s


def run_one(d, args, order, ref, tag, problem="electrostatic", freq=10.0, save=0, uniform=None,
            points=None):
    rd = os.path.abspath(os.path.join(args.out, f"row{d['row']:05d}", tag))
    os.makedirs(rd, exist_ok=True)
    mesh = os.path.join(rd, "mesh.msh")
    t0 = time.time()
    ntri = build_mesh(d, mesh, ref=ref, uniform=uniform, points=points)
    keep, xy = probe_points(d)
    cfg = os.path.join(rd, "config.json")
    write_config(d, mesh, os.path.join(rd, "postpro"), keep, xy, order, cfg, problem, freq,
                 save)
    t1 = time.time()
    with open(os.path.join(rd, "palace.log"), "w") as log:
        subprocess.run([args.palace, "-np", str(args.np), cfg], cwd=rd, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    t2 = time.time()
    Ec = read_probes(os.path.join(rd, "postpro", "probe-E.csv"), keep, d["u"].shape[1:])
    E, imag_resid = realify(Ec)
    u, s = normalize(E, d["mask"])
    np.savez_compressed(os.path.join(rd, "fields.npz"), E=E, Ec=Ec, u=u, s=s, keep=keep)
    return dict(tag=tag, problem=problem, uniform=uniform, points=points,
                freq=freq if problem == "boundarymode" else None, order=order, ref=ref, ntri=ntri,
                mesh_s=t1 - t0, palace_s=t2 - t1, s=s, imag_resid=imag_resid, dir=rd)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--palace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, nargs="+", required=True)
    ap.add_argument("--order", type=int, nargs="+", default=[1])
    ap.add_argument("--ref", type=float, nargs="+", default=[1.0])
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--problem", choices=["electrostatic", "boundarymode"],
                    default="electrostatic")
    ap.add_argument("--freq", type=float, nargs="+", default=[10.0], help="GHz, BoundaryMode")
    ap.add_argument("--save", type=int, default=0, help="also write ParaView output")
    ap.add_argument("--uniform", type=float, help="element size = UNIFORM * t everywhere")
    ap.add_argument("--points", type=float, nargs=2, metavar=("A", "B"),
                    help="Gmsh point sizes A * t at the trace corners, B * t elsewhere")
    args = ap.parse_args(argv)
    freqs = args.freq if args.problem == "boundarymode" else [None]
    for row in args.rows:
        d = load_row(args.h5, row)
        for order in args.order:
            for ref in args.ref:
                for f in freqs:
                    mesh = (f"pts{args.points[0]:g}-{args.points[1]:g}t" if args.points else
                            f"uni{args.uniform:g}t" if args.uniform is not None else f"ref{ref:g}")
                    tag = (f"es_p{order}_{mesh}" if f is None else f"bm{f:g}GHz_p{order}_{mesh}")
                    r = run_one(d, args, order, ref, tag, args.problem, f or 0.0, args.save,
                                args.uniform, args.points)
                    print(json.dumps({"row": row, "sid": d["sid"], **r}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
