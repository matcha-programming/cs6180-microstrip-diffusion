"""Export validation scores and actual LR histories, including truncated runs."""
import argparse
import csv
import json
from pathlib import Path


def write(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores, learning_rates = [], []
    for path in sorted(args.root.rglob("validation.json")):
        cfg = json.loads((path.parent / "config.json").read_text())
        val = json.loads(path.read_text())
        if cfg.get("round") != 3 or cfg.get("smoke_test_only"):
            continue
        for name, checkpoint in val["checkpoints"].items():
            for step, metrics in checkpoint["steps"].items():
                scores.append({"run": str(path.parent), "phase": cfg["phase"], "schedule": cfg["schedule"],
                               "training_seed": cfg["seed"], "checkpoint": name,
                               "primary": name == "best-field", "epoch": checkpoint["epoch"], "steps": step,
                               "gpu": cfg["gpu"], "train_hours": val["total_train_seconds"] / 3600,
                               "equal_update_budget_complete": val["equal_update_budget_complete"],
                               "updates": val["updates_in_round3"], **metrics})
        for line in (path.parent / "history.jsonl").read_text().splitlines():
            row = json.loads(line)
            if "lr_last" not in row:
                continue
            learning_rates.append({"run": str(path.parent), "schedule": cfg["schedule"], "training_seed": cfg["seed"],
                                   "epoch": row["epoch"], "updates": row["updates_in_round3"],
                                   "lr_first": row["lr_first"], "lr_last": row["lr_last"], "lr_next": row["lr_next"],
                                   "field_mse": row["field"]["masked_mse_global"] if row["field"] else None})
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "validation.csv", scores)
    write(args.output / "lr-history.csv", learning_rates)
    print(json.dumps({"validation_rows": len(scores), "lr_rows": len(learning_rates)}))


if __name__ == "__main__":
    main()
