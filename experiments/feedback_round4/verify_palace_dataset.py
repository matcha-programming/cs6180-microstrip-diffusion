"""Verify all generated cases and the packed dataset without changing the gate."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
from verify_palace_pilot import verify


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cases_check(root):
    manifest = json.loads((root / "cases.json").read_text())
    assert len(manifest["cases"]) == 288
    shards = [verify(root, index) for index in range(16)]
    records = []
    for case in manifest["cases"]:
        saved = json.loads((root / "cases" / case["id"] / "complete.json").read_text())
        lines = (Path(saved["solver"]["dir"]) / "palace.log").read_text().splitlines()
        convergence = [int(m.group(1)) for line in lines
                       if (m := re.search(r"PCG solver converged in (\d+) iterations", line))]
        assert convergence, case["id"]
        boundary = next(i for i, line in enumerate(lines)
                        if "Configuring Dirichlet BC at attributes:" in line)
        assert lines[boundary + 1].strip() == "3-4", case["id"]
        assert "Computing electrostatic fields for 1 terminal boundary" in "\n".join(lines)
        warnings = [i for i, line in enumerate(lines) if "Warning!" in line]
        for i in warnings:
            assert "external boundary attributes has no associated boundary condition" in lines[i + 1], case["id"]
        records.append({"id": case["id"], "kind": case["kind"],
                        "pcg_iterations": convergence, "dirichlet_attributes": [3, 4],
                        "known_boundary_warning_count": len(warnings),
                        "raw_reference_relative_l2": saved.get("raw_relative_l2"),
                        "sample_sha256": saved["sample_sha256"],
                        "solver_seconds": saved["solver"]["palace_s"],
                        "triangles": saved["solver"]["ntri"]})
    write(root / "all-case-verification.json", {
        "status": "passed", "cases_verified": sum(s["cases_verified"] for s in shards),
        "shard_checks": shards, "solver_checks": records,
        "scope": "Structural and solver checks; compatibility is evaluated separately."})
    print("All 288 cases passed structural and solver checks", flush=True)


def dataset_check(root):
    manifest = json.loads((root / "cases.json").read_text())
    cases = manifest["cases"]
    fresh = [case for case in cases if case["kind"] == "new"]
    complete = json.loads((root / "complete.json").read_text())
    gate = json.loads((root / "quality-gate.json").read_text())
    digest = hashlib.sha256((root / "palace-pilot.h5").read_bytes()).hexdigest()
    assert digest == complete["h5_sha256"]
    references, refinements = [], []
    for case in cases:
        folder = root / "cases" / case["id"]
        if case["kind"] == "reference":
            references.append(json.loads((folder / "complete.json").read_text())["raw_relative_l2"])
        elif case["kind"] == "refinement":
            with np.load(folder / "sample.npz") as fine, np.load(
                    root / "cases" / f'ref-{case["row"]:05}' / "sample.npz") as coarse:
                valid = ~fine["mask"]
                refinements.append(float(np.linalg.norm((fine["u"] - coarse["u"])[:, valid]) /
                                         np.linalg.norm(fine["u"][:, valid])))
    assert len(references) == 24 and len(refinements) == 8 and len(fresh) == 256
    values = {"reference_mean": float(np.mean(references)),
              "reference_p95": float(np.quantile(references, .95)),
              "refinement_max": max(refinements)}
    limits = {"reference_mean": .05, "reference_p95": .10, "refinement_max": .02}
    assert gate["thresholds"] == limits
    assert all(np.isclose(gate[key], value, rtol=1e-12, atol=1e-12) for key, value in values.items())
    passed = all(values[key] <= limit for key, limit in limits.items())
    assert passed == gate["passed"] == complete["training_eligible"]
    with h5py.File(root / "palace-pilot.h5", "r") as data:
        assert data["u"].shape == (256, 2, 64, 128)
        assert len(set(data["analysis/sample_id"][:].tolist())) == 256
        assert bool(data.attrs["course_compatible_gate"]) == passed
        assert data.attrs["source_dataset_sha256"] == manifest["source_dataset_sha256"]
        for i, case in enumerate(fresh):
            with np.load(root / "cases" / case["id"] / "sample.npz") as sample:
                for key in ("u", "mask", "y_over_h", "z_over_h", "s"):
                    actual = data[key][i]
                    assert np.isfinite(actual).all()
                    assert np.array_equal(actual, np.asarray(sample[key], dtype=data[key].dtype)), (case["id"], key)
                assert np.array_equal(data["analysis/params_raw"][i], np.asarray(case["raw"]))
                assert np.array_equal(data["x"][i], np.asarray(case["x"], dtype="f4"))
                assert int(data["analysis/sample_id"][i]) == case["sample_id"]
    write(root / "dataset-verification.json", {
        "status": "passed", "new_designs": 256, "reference_designs": 24,
        "refined_designs": 8, "h5_sha256": digest,
        "h5_bytes": (root / "palace-pilot.h5").stat().st_size,
        "packed_rows_match_verified_samples": True, "gate_recomputed": values,
        "reference_errors": references, "refinement_errors": refinements,
        "training_eligible": passed,
        "scope": "Passed describes artifact integrity; training eligibility is the separate gate result."})
    print(json.dumps({"artifact_integrity": "passed", "training_eligible": passed, **values}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("cases", "dataset"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    (cases_check if args.action == "cases" else dataset_check)(args.root.resolve())
