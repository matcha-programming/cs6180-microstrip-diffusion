"""Materialize one predeclared campaign task; --dry-run prints its exact command."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def command_for(args, campaign):
    common = ["--data", str(args.data), "--output"]
    if args.stage == "draws":
        item = campaign["draw_studies"][args.task_index]
        command = [sys.executable, str(HERE / "draw_study.py"), *common, str(args.output / item["name"]),
                   "--checkpoint", str(ROOT / item["checkpoint"]), "--steps", str(item["steps"]),
                   "--sizes", *map(str, campaign["draws"]["sizes"]),
                   "--limit", str(campaign["draws"]["validation_count"]),
                   "--seed-start", str(campaign["draws"]["seed_start"])]
    else:
        if args.stage == "continue":
            item = campaign["continuation"]
        elif args.stage == "min-snr":
            item = campaign["min_snr_trial"]
        else:
            item = campaign["learning_rate_trials"][args.task_index]
        budget = item if args.stage == "continue" else campaign["learning_rate_budget"]
        command = [sys.executable, str(HERE / "train.py"), *common, str(args.output / item["name"]),
                   "--lr", str(item["lr"]), "--epochs", str(budget["epochs"]),
                   "--max-train-hours", str(budget["max_train_hours"])]
        for key in ("backbone", "base", "batch_size", "seed", "field_interval", "validation_count", "field_steps", "patience_evaluations"):
            command += ["--" + key.replace("_", "-"), str(campaign["training"][key])]
        if args.stage == "min-snr":
            command += ["--loss-weighting", item["loss_weighting"], "--snr-gamma", str(item["snr_gamma"])]
        if args.stage == "continue":
            if not args.resume:
                raise ValueError("R2-B requires --resume /path/to/11604740/diffusion/last.pt (full optimizer state)")
            command += ["--resume", str(args.resume)]
    if args.smoke:
        command += ["--smoke", "--device", "cpu"]
        if args.stage != "draws":
            command += ["--base", "4"]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["draws", "continue", "lr", "min-snr"], required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--data", type=Path, default="/courses/CS6180.202710/shared/final_proj/em2d_task.h5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    campaign = json.loads((HERE / "campaign.json").read_text())
    counts = {"draws": len(campaign["draw_studies"]), "lr": len(campaign["learning_rate_trials"]), "continue": 1, "min-snr": 1}
    if not 0 <= args.task_index < counts[args.stage]:
        parser.error(f"Task index for {args.stage} must be 0..{counts[args.stage] - 1}")
    try:
        command = command_for(args, campaign)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"stage": args.stage, "task_index": args.task_index, "command": command}), flush=True)
    if args.dry_run:
        return
    # Hash the course file before dispatch; synthetic smoke fixtures are explicitly exempt.
    if not args.smoke:
        import hashlib
        digest = hashlib.sha256()
        with args.data.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != campaign["dataset_sha256"]:
            parser.error("Dataset checksum differs from the round-one course file")
    args.output.mkdir(parents=True, exist_ok=True)
    # Keep logs beside, rather than inside, the output directory that the worker creates.
    name = f"{args.stage}-{args.task_index}"
    log = args.output / f"{name}.log"
    manifest = args.output / f"{name}-launch.json"
    if manifest.exists():
        parser.error("Launch already exists; use a new campaign root or resume train.py explicitly")
    manifest.write_text(json.dumps({"command": command, "status": "running", "slurm_job_id": os.getenv("SLURM_JOB_ID")}, indent=2) + "\n")
    with log.open("x") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
    manifest.write_text(json.dumps({"command": command, "status": "complete" if result.returncode == 0 else "failed", "exit_code": result.returncode}, indent=2) + "\n")
    if result.returncode:
        raise SystemExit(f"Task failed; see {log}")


if __name__ == "__main__":
    main()
