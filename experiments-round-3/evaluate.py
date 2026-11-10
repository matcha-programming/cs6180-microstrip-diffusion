"""Evaluate one frozen training-seed model through the unchanged round-two sampler."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from r3_support import ROUND2, dump, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "test_extrap"], required=True)
    parser.add_argument("--data", default="/courses/CS6180.202710/shared/final_proj/em2d_task.h5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.selection.read_text())
    if manifest.get("round") != 3 or manifest.get("smoke_test_only") or "confirmed_selection_sha256" not in manifest:
        parser.error("Use a confirmed round-three final-seed manifest")
    group = args.selection.parent / "confirmed-selection.json"
    if sha(group) != manifest["confirmed_selection_sha256"]:
        parser.error("Confirmation summary changed after selection")
    command = [sys.executable, str(ROUND2 / "draw_study.py"), "--data", args.data,
               "--checkpoint", manifest["checkpoint"], "--steps", str(manifest["steps"]),
               "--sizes", *map(str, manifest["report_sizes"]), "--seed-start", str(manifest["seed_start"]),
               "--split", args.split, "--limit", "0", "--confirm-test", "--selection", str(args.selection),
               "--output", str(args.output)]
    subprocess.run(command, check=True)
    for name in ("config.json", "metrics.json"):
        result = json.loads((args.output / name).read_text())
        result.update({"experiment": "R3-final", "schedule": manifest["schedule"],
                       "training_seed": manifest["training_seed"], "round3_selection_sha256": sha(args.selection)})
        dump(args.output / name, result)


if __name__ == "__main__":
    main()
