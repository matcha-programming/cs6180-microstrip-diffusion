"""The sampling pipeline that reproduces em2d_task.h5 from a Palace electrostatic solution.

Reconstructed from the data, not from staff code:
  1. Palace "Electrostatic", Solver.Order 1, trace as a 1 V terminal, ground plane at z = 0.
  2. The potential V (ParaView output) is interpolated at the 64 x 128 grid points. Points
     inside the trace are not in the mesh; they take the trace potential (--inside trace).
  3. E = -grad V by central chord differences on the grid, (V[i+1] - V[i-1]) / (x[i+1] - x[i-1]),
     one-sided at the edges.
  4. u = E / max |E| over unmasked pixels; masked pixels set to 0.

Usage: staffpipe.py --h5 DATA.h5 --runs RUNS_DIR [--inside trace|zero] [--json OUT.json]
Evaluates every run directory under RUNS_DIR/rowNNNNN/<tag>/ that has electrostatic ParaView
output, printing one JSON line per run.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402
import regen  # noqa: E402
import resample  # noqa: E402


def staff_u(run_dir, d, inside="trace", scheme="chord"):
    V, ok = resample.potential_on_grid(run_dir, d)
    V = np.where(ok, V, 0.0 if inside == "zero" else np.nanmax(np.where(ok, V, np.nan)))
    E = resample.gradient_field(V, d, scheme)
    u, s = regen.normalize(E, d["mask"])
    return u, E, s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--inside", choices=["zero", "trace"], default="trace")
    ap.add_argument("--scheme", choices=["chord", "nonuniform"], default="chord")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    out, cache = [], {}
    pat = os.path.join(args.runs, "row*", "*", "postpro", "paraview", "electrostatic")
    for pv in sorted(glob.glob(pat)):
        rd = os.path.dirname(os.path.dirname(os.path.dirname(pv)))
        row = int(os.path.basename(os.path.dirname(rd))[3:])
        d = cache.setdefault(row, regen.load_row(args.h5, row))
        u, E, s = staff_u(rd, d, args.inside, args.scheme)
        np.savez_compressed(os.path.join(rd, f"staff_u_{args.inside}_{args.scheme}.npz"), u=u, E=E,
                            s=s)
        m = dict(row=row, sid=d["sid"], tag=os.path.basename(rd), inside=args.inside,
                 scheme=args.scheme, h=d["h"], w_over_h=d["w"] / d["h"], t_over_h=d["t"] / d["h"],
                 eps_r=d["er"], **compare.metrics(u, d))
        out.append(m)
        print(json.dumps(m), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
