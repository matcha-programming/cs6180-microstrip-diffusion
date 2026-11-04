"""Compare re-simulated fields (runs/rowNNNNN/<tag>/fields.npz) with the dataset targets.

Usage: compare.py --h5 DATA.h5 --runs RUNS_DIR [--json OUT.json]
Prints one JSON line per run and optionally writes them all to OUT.json.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regen import load_row  # noqa: E402

BOX = np.ones((3, 3), bool)


def rel(a, b, sel):
    return float(np.linalg.norm((a - b)[:, sel]) / np.linalg.norm(b[:, sel]))


def metrics(up, d):
    ud, mask = d["u"], d["mask"]
    valid = ~mask
    band = binary_dilation(mask, BOX, iterations=3) & valid
    far = ~binary_dilation(mask, BOX, iterations=6)
    scale = float((up[:, valid] * ud[:, valid]).sum() / (up[:, valid] ** 2).sum())
    dy = d["y"][1] - d["y"][0]
    side = np.abs(d["y"]) > d["w"] / 2 + 3 * dy
    foot = np.abs(d["y"]) <= d["w"] / 2 + dy / 2

    def ratio(u):
        return float(np.median(u[1, 31, side] / u[1, 30, side]))

    def stair(u):
        a = u[:, :11, :].astype(np.float32)
        same = np.isclose(a[:, 1:], a[:, :-1], rtol=1e-6, atol=0) & (a[:, 1:] != 0)
        return float(same.mean())

    return dict(
        rel_all=rel(up, ud, valid), rel_far=rel(up, ud, far), rel_band=rel(up, ud, band),
        rel_Ey=float(np.linalg.norm((up - ud)[0, valid]) / np.linalg.norm(ud[0, valid])),
        rel_Ez=float(np.linalg.norm((up - ud)[1, valid]) / np.linalg.norm(ud[1, valid])),
        scale=scale, rel_all_scaled=rel(scale * up, ud, valid),
        ez_ratio_31_30=dict(data=ratio(ud), palace=ratio(up)),
        flips_under_trace=dict(data=int((ud[1, :31, :][:, foot] > 0).sum()),
                               palace=int((up[1, :31, :][:, foot] > 0).sum())),
        stair_rows0_10=dict(data=stair(ud), palace=stair(up)),
        peak_rc=dict(data=[int(v) for v in np.unravel_index(
                         np.argmax(np.where(mask, 0, np.hypot(*ud))), mask.shape)],
                     palace=[int(v) for v in np.unravel_index(
                         np.argmax(np.where(mask, 0, np.hypot(*up))), mask.shape)]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    out, cache = [], {}
    for fn in sorted(glob.glob(os.path.join(args.runs, "row*", "*", "fields.npz"))):
        row = int(os.path.basename(os.path.dirname(os.path.dirname(fn)))[3:])
        tag = os.path.basename(os.path.dirname(fn))
        d = cache.setdefault(row, load_row(args.h5, row))
        m = dict(row=row, sid=d["sid"], tag=tag, h=d["h"], w_over_h=d["w"] / d["h"],
                 t_over_h=d["t"] / d["h"], eps_r=d["er"], **metrics(np.load(fn)["u"], d))
        out.append(m)
        print(json.dumps(m), flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
