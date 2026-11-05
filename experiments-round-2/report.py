"""Build separate validation tables for training trials and ensemble-size studies."""
import argparse
import csv
import json
from pathlib import Path


def write_table(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    training, draws = [], []
    for path in sorted(args.root.rglob("validation.json")):
        run = path.parent
        if not (run / "complete.json").exists():
            continue
        cfg = json.loads((run / "config.json").read_text())
        val = json.loads(path.read_text())
        if cfg.get("smoke_test_only") or "round" not in cfg:
            continue
        for name, checkpoint in val["checkpoints"].items():
            for steps, scores in checkpoint["steps"].items():
                training.append({"run": str(run), "checkpoint": name, "steps": steps,
                                 "backbone": cfg["backbone"], "lr": cfg["learning_rate"], "seed": cfg["seed"],
                                 "loss_weighting": cfg.get("loss_weighting", "none"), "snr_gamma": cfg.get("snr_gamma"),
                                 "epoch": checkpoint["epoch"], "gpu": cfg["gpu"],
                                 "total_training_hours": val["historical_plus_round2_train_seconds"] / 3600,
                                 "new_training_hours": val["new_train_seconds"] / 3600,
                                 "stop_reason": val["stop_reason"], **scores})
    for path in sorted(args.root.rglob("metrics.json")):
        cfg_path = path.parent / "config.json"
        if not cfg_path.exists() or not (path.parent / "complete.json").exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        result = json.loads(path.read_text())
        if cfg.get("smoke_test_only") or result.get("experiment") != "R2-A" or result["split"] != "val":
            continue
        for k, scores in result["sizes"].items():
            draws.append({"run": str(path.parent), "backbone": cfg["backbone"], "steps": cfg["steps"],
                          "draws": int(k), "designs": scores["ensemble"]["designs"],
                          **{metric: scores["ensemble"][metric]["mean"] for metric in ("relative_l2", "edge_relative_l2", "gradient_relative_l2", "masked_mse")},
                          "mean_design_variance": scores["mean_design_variance"],
                          "estimated_serial_sampler_seconds": scores["estimated_serial_ensemble_seconds"], "gpu": cfg["gpu"]})
    write_table(args.output / "training-validation.csv", training)
    write_table(args.output / "draw-validation.csv", draws)
    print(json.dumps({"training_rows": len(training), "draw_rows": len(draws)}))


if __name__ == "__main__":
    main()
