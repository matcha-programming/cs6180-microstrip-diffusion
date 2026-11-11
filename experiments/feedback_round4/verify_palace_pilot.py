"""Check every completed case in a Palace shard before expanding generation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(root, index):
    manifest = json.loads((root / "cases.json").read_text())
    cases = manifest["cases"][index::16]
    assert cases and len({case["id"] for case in cases}) == len(cases)
    checked = []
    for case in cases:
        folder = root / "cases" / case["id"]
        record = json.loads((folder / "complete.json").read_text())
        assert record["case"] == case, case["id"]
        sample = folder / "sample.npz"
        checksum = hashlib.sha256(sample.read_bytes()).hexdigest()
        assert checksum == record["sample_sha256"], case["id"]
        with np.load(sample, allow_pickle=False) as data:
            u, field, mask = data["u"], data["E"], data["mask"]
            scale = float(data["s"])
            assert u.shape == field.shape == (2, 64, 128)
            assert mask.shape == (64, 128) and mask.dtype == np.bool_
            assert (~mask).any() and mask.any()
            assert np.isfinite(u).all() and np.isfinite(field).all()
            assert np.isfinite(scale) and scale > 0
            assert np.count_nonzero(u[:, mask]) == 0
            assert np.allclose(u[:, ~mask], field[:, ~mask] / scale, rtol=2e-6, atol=1e-7)
            peak = float(np.linalg.norm(u[:, ~mask], axis=0).max())
            assert abs(peak - 1) < 1e-5
            assert data["y_over_h"].shape == (128,) and data["z_over_h"].shape == (64,)
            for key in ("y_over_h", "z_over_h"):
                assert np.isfinite(data[key]).all() and (np.diff(data[key]) > 0).all()
            assert np.array_equal(data["params_raw"], np.asarray(case["raw"]))
        solver_dir = Path(record["solver"]["dir"])
        assert solver_dir.is_relative_to(root.resolve())
        for name in ("palace.log", "config.json", "mesh.msh"):
            assert (solver_dir / name).stat().st_size > 0
        assert list(solver_dir.rglob("*.vtu")), case["id"]
        checked.append({"id": case["id"], "kind": case["kind"],
                        "sample_sha256": checksum, "normalized_peak": peak,
                        "s_v_per_m": scale})
    return {"status": "passed", "shard": index, "cases_verified": len(checked),
            "checks": ["case identity", "SHA256", "shapes", "finite fields",
                       "mask zeros", "E/s normalization", "coordinate ordering",
                       "input parameters", "solver log/config/mesh and VTU presence"],
            "scope": "Structural verification only; course compatibility requires the separate quality gate.",
            "cases": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0, choices=range(16))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root.resolve(), args.index)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}))
