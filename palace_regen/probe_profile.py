"""Sample one Palace solution densely in z around the dielectric interface, to test how the
dataset's rows 30-32 were produced. Writes profile.npz with E (2, n_z, 128) on z_fine x y_row.

Usage: probe_profile.py --h5 DATA.h5 --palace PALACE_BIN --out DIR --row ROW
       [--problem electrostatic|boundarymode] [--freq GHz] [--order 1] [--uniform C]
"""

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regen  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--palace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--problem", default="electrostatic")
    ap.add_argument("--freq", type=float, default=10.0)
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--ref", type=float, default=1.0)
    ap.add_argument("--uniform", type=float)
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--full", action="store_true", help="z/h 0..8 on every 4th column")
    args = ap.parse_args(argv)
    d = regen.load_row(args.h5, args.row)
    h = d["h"]
    # z/h from 0.80 to 1.60 in steps of 0.005, plus points 1e-6 h either side of the interface
    zf = np.unique(np.r_[np.arange(0.80, 1.6001, 0.005), 1 - 1e-6, 1 + 1e-6]) * h
    yv = d["y"]
    if args.full:
        zf = np.unique(np.r_[np.arange(0, 1, 0.0025), np.arange(1, 8.0001, 0.01),
                             1 - 1e-6, 1 + 1e-6]) * h
        yv = d["y"][::4]
    Z, Y = np.meshgrid(zf, yv, indexing="ij")
    tol = 1e-9 * h
    inside = (np.abs(Y) < d["w"] / 2 - tol) & (Z > h + tol) & (Z < h + d["t"] - tol)
    keep = np.flatnonzero(~inside.ravel())
    xy = np.c_[Y.ravel()[keep], Z.ravel()[keep]]
    rd = os.path.abspath(args.out)
    os.makedirs(rd, exist_ok=True)
    mesh = os.path.join(rd, "mesh.msh")
    regen.build_mesh(d, mesh, ref=args.ref, uniform=args.uniform)
    cfg = os.path.join(rd, "config.json")
    regen.write_config(d, mesh, os.path.join(rd, "postpro"), keep, xy, args.order, cfg,
                       args.problem, args.freq)
    with open(os.path.join(rd, "palace.log"), "w") as log:
        subprocess.run([args.palace, "-np", str(args.np), cfg], cwd=rd, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    Ec = regen.read_probes(os.path.join(rd, "postpro", "probe-E.csv"), keep, Z.shape)
    E, _ = regen.realify(Ec)
    k = int(np.argmin(np.abs(zf - (1 - 1e-6) * h)))  # just under the trace centre: E_z < 0
    if E[1, k, len(yv) // 2] > 0:
        E = -E
    np.savez_compressed(os.path.join(rd, "profile.npz"), E=E, z=zf, y=yv)
    print(rd)


if __name__ == "__main__":
    sys.exit(main())
