"""Print or execute one predeclared round-three task; never submits scheduler jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def build(args, campaign):
    if args.phase == "screen":
        item = campaign["schedules"][args.task_index]
        seed = campaign["training"]["seed"]
    else:
        if args.plan is None:
            raise ValueError("Confirmation needs --plan from choose.py screen")
        plan = json.loads(args.plan.read_text())
        if plan["campaign_sha256"] != hashlib.sha256((HERE / "campaign.json").read_bytes()).hexdigest():
            raise ValueError("Campaign changed after the confirmation plan was frozen")
        task = plan["confirmation_tasks"][args.task_index]
        item = next(s for s in campaign["schedules"] if s["schedule"] == task["schedule"])
        seed = task["seed"]
    settings = {**campaign["training"], **item}
    output = args.output / f"{item['name']}-seed{seed}"
    command = [sys.executable, str(HERE / "train.py"), "--data", str(args.data), "--output", str(output),
               "--schedule", item["schedule"], "--seed", str(seed), "--phase", args.phase]
    for key in ("backbone", "base", "batch_size", "epochs", "max_train_hours", "lr", "min_lr", "warmup_epochs", "field_interval", "validation_count", "field_steps"):
        command += ["--" + key.replace("_", "-"), str(settings[key])]
    if args.smoke:
        command += ["--smoke", "--device", "cpu", "--base", "4"]
    return command, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["screen", "confirm"], default="screen")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--data", type=Path, default="/courses/CS6180.202710/shared/final_proj/em2d_task.h5")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    campaign = json.loads((HERE / "campaign.json").read_text())
    try:
        if args.task_index < 0:
            raise ValueError("Negative task index")
        command, output = build(args, campaign)
    except (ValueError, IndexError, StopIteration, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({"command": command, "output": str(output)}), flush=True)
    if args.dry_run:
        return
    if output.exists():
        parser.error("Use a new output or resume that run explicitly with train.py")
    if not args.smoke:
        digest = hashlib.sha256()
        with args.data.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != campaign["dataset_sha256"]:
            parser.error("Course dataset checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / (output.name + ".log")).open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"Training failed: {output.name}.log")
    complete = json.loads((output / "complete.json").read_text())
    if complete["status"] != "complete":
        raise SystemExit("Update budget incomplete; recover last.pt before comparing schedules")


if __name__ == "__main__":
    main()
