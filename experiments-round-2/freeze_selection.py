"""Freeze checkpoint/sampler choices using full validation, before held-out evaluation."""
import argparse
import json
from pathlib import Path

from support import dump, sha


def choose(runs, report_sizes):
    candidates = []
    reference = None
    for run in runs:
        cfg = json.loads((run / "config.json").read_text())
        result = json.loads((run / "validation.json").read_text())
        complete = json.loads((run / "complete.json").read_text())
        if cfg["smoke_test_only"] or result["smoke_test_only"] or complete.get("smoke_test_only"):
            raise ValueError("Smoke results cannot select a scientific result")
        if complete["status"] != "complete":
            raise ValueError("Run has not completed")
        signature = {k: cfg[k] for k in ("data_sha256", "split_hashes", "validation_rows", "field_steps")}
        if reference is not None and signature != reference:
            raise ValueError("Runs use different data or validation protocols")
        reference = signature
        for name, checkpoint in result["checkpoints"].items():
            path = (run / f"{name}.pt").resolve()
            if sha(path) != checkpoint["sha256"]:
                raise ValueError("Checkpoint changed after validation")
            for steps, metrics in checkpoint["steps"].items():
                if metrics["designs"] != 881:
                    raise ValueError("Selection requires all 881 validation designs")
                candidates.append({"checkpoint": str(path), "checkpoint_sha256": checkpoint["sha256"],
                                   "steps": int(steps), "validation_field_mse": metrics["masked_mse_global"],
                                   "run": str(run.resolve()), "epoch": checkpoint["epoch"],
                                   "learning_rate": cfg["learning_rate"], "backbone": cfg["backbone"],
                                   "loss_weighting": cfg.get("loss_weighting", "none"), "snr_gamma": cfg.get("snr_gamma")})
    winner = min(candidates, key=lambda r: (r["validation_field_mse"], r["steps"], r["checkpoint"]))
    return {"round": 2, "selection_split": "val", "selection_metric": "masked_mse_global",
            "smoke_test_only": False, **reference, **winner, "report_sizes": sorted(set(report_sizes)), "seed_start": 6180,
            "candidates": candidates, "note": "Freeze this file before --confirm-test; no test scores enter selection."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-sizes", type=int, nargs="+", default=[1, 3])
    args = parser.parse_args()
    if args.output.exists() or min(args.report_sizes) < 1:
        parser.error("Use a new manifest path and positive reporting draw counts")
    dump(args.output, choose(args.runs, args.report_sizes))
    print("SELECTION_FROZEN", args.output, flush=True)


if __name__ == "__main__":
    main()
