"""Rank completed schedules on validation, then compare seed-averaged confirmations."""
import argparse
import json
import statistics
import math
from dataclasses import asdict
from pathlib import Path

from r3_support import ROOT, dump, sha
from schedules import NAMES, ScheduleSpec

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / "campaign.json"


def read_runs(paths):
    items, protocol, identities = [], None, set()
    initialization = {}
    campaign = json.loads(CAMPAIGN.read_text())
    for run in paths:
        cfg = json.loads((run / "config.json").read_text())
        val = json.loads((run / "validation.json").read_text())
        complete = json.loads((run / "complete.json").read_text())
        if cfg.get("round") != 3 or cfg["smoke_test_only"] or val["smoke_test_only"]:
            raise ValueError("Selection requires full-data round-three runs")
        if complete["status"] != "complete" or not val["equal_update_budget_complete"] or val["updates_in_round3"] != cfg["max_updates"]:
            raise ValueError("A schedule has not finished its common update budget; resume it first")
        expected_updates = math.ceil(7048 / campaign["training"]["batch_size"]) * campaign["training"]["epochs"]
        if cfg["max_updates"] != expected_updates or cfg["data_sha256"] != campaign["dataset_sha256"]:
            raise ValueError("Run differs from the frozen course-data/update-horizon design")
        warmup = 0 if cfg["schedule"] == "constant" else campaign["training"]["warmup_epochs"] * math.ceil(7048 / cfg["batch_size"])
        expected_spec = asdict(ScheduleSpec(cfg["schedule"], expected_updates, warmup, campaign["training"]["lr"], campaign["training"]["min_lr"]))
        if cfg["schedule_config"] != json.loads(json.dumps(expected_spec)):
            raise ValueError("Schedule parameters differ from the predeclared design")
        signature = {k: cfg[k] for k in ("base", "backbone", "batch_size", "learning_rate", "min_lr", "mixed_precision", "loss_weighting",
                    "max_updates", "data_sha256", "split_hashes", "validation_rows", "field_interval", "field_steps", "source_sha256", "gpu")}
        if signature["loss_weighting"] != "none":
            raise ValueError("Schedule comparisons keep the loss unweighted")
        if protocol is not None and signature != protocol:
            raise ValueError("Runs have different non-schedule settings or source versions")
        protocol = signature
        identity = (cfg["schedule"], cfg["seed"])
        if identity in identities:
            raise ValueError("Duplicate schedule/training seed")
        identities.add(identity)
        seed = str(cfg["seed"])
        if seed in initialization and initialization[seed] != cfg["initial_model_sha256"]:
            raise ValueError("Initial weights differ across schedules with the same training seed")
        initialization[seed] = cfg["initial_model_sha256"]
        checkpoint = val["checkpoints"]["best-field"]
        path = (run / "best-field.pt").resolve()
        if sha(path) != checkpoint["sha256"]:
            raise ValueError("Checkpoint changed after validation")
        if set(checkpoint["steps"]) != {"20", "50"} or any(s["designs"] != 881 for s in checkpoint["steps"].values()):
            raise ValueError("Primary selection requires full validation at 20 and 50 steps")
        items.append({"schedule": cfg["schedule"], "seed": cfg["seed"], "phase": cfg["phase"],
                      "run": str(run.resolve()), "checkpoint": str(path), "checkpoint_sha256": checkpoint["sha256"],
                      "epoch": checkpoint["epoch"], "steps": checkpoint["steps"], "config_sha256": sha(run / "config.json")})
    if not items:
        raise ValueError("No completed runs")
    return items, protocol


def screen(paths):
    items, protocol = read_runs(paths)
    campaign = json.loads(CAMPAIGN.read_text())
    if {i["schedule"] for i in items} != set(NAMES) or len(items) != len(NAMES) or any(i["seed"] != 6180 or i["phase"] != "screen" for i in items):
        raise ValueError("Screening requires all five schedules at training seed 6180")
    ranking = []
    for item in items:
        step = min((20, 50), key=lambda s: (item["steps"][str(s)]["masked_mse_global"], s))
        ranking.append({**item, "selected_steps": step, "validation_field_mse": item["steps"][str(step)]["masked_mse_global"]})
    ranking.sort(key=lambda r: (r["validation_field_mse"], NAMES.index(r["schedule"])))
    challenger = next(i for i in ranking if i["schedule"] != "constant")["schedule"]
    tasks = [{"schedule": name, "seed": seed} for name in ("constant", challenger)
             for seed in campaign["confirmation_seeds"] if seed != 6180]
    return {"round": 3, "selection_split": "val", "campaign_sha256": sha(CAMPAIGN),
            "protocol": protocol, "screen_ranking": ranking, "challenger": challenger,
            "confirmation_tasks": tasks,
            "note": "Confirm the best nonconstant challenger against constant even if constant won screening. No held-out results enter this plan."}


def confirm(paths, plan):
    if plan["campaign_sha256"] != sha(CAMPAIGN):
        raise ValueError("Campaign differs from the frozen plan")
    items, protocol = read_runs(paths)
    if protocol != plan["protocol"]:
        raise ValueError("Confirmation protocol differs from screening")
    seeds = json.loads(CAMPAIGN.read_text())["confirmation_seeds"]
    names = ("constant", plan["challenger"])
    expected = {(name, seed) for name in names for seed in seeds}
    if {(i["schedule"], i["seed"]) for i in items} != expected:
        raise ValueError("Need constant and the challenger, each at all three training seeds")
    for item in items:
        if item["seed"] == 6180:
            original = next(r for r in plan["screen_ranking"] if r["schedule"] == item["schedule"])
            if item["checkpoint_sha256"] != original["checkpoint_sha256"] or item["config_sha256"] != original["config_sha256"]:
                raise ValueError("A screening run changed after planning confirmation")
        elif item["phase"] != "confirm":
            raise ValueError("Additional seeds must be recorded as confirmation runs")
    summary = []
    for name in names:
        per_step = {}
        for steps in (20, 50):
            values = [i["steps"][str(steps)]["masked_mse_global"] for i in items if i["schedule"] == name]
            per_step[str(steps)] = {"mean": statistics.mean(values), "training_seed_std_ddof1": statistics.stdev(values)}
        chosen = min((20, 50), key=lambda s: (per_step[str(s)]["mean"], s))
        summary.append({"schedule": name, "steps": chosen, "validation_field_mse": per_step[str(chosen)]["mean"], "per_step": per_step})
    winner = min(summary, key=lambda r: (r["validation_field_mse"], NAMES.index(r["schedule"])))
    return {"round": 3, "selection_split": "val", "protocol": protocol, "comparison": summary,
            "winner": winner, "runs": items,
            "note": "Schedule and sampler selected by mean validation MSE across three independently trained models, never by the luckiest seed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["screen", "confirm"])
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.phase == "confirm" and not args.plan:
        parser.error("Confirmation selection requires --plan")
    result = screen(args.runs) if args.phase == "screen" else confirm(args.runs, json.loads(args.plan.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / ("confirmation-plan.json" if args.phase == "screen" else "confirmed-selection.json"), result)
    if args.phase == "confirm":
        winner = result["winner"]
        for item in result["runs"]:
            if item["schedule"] != winner["schedule"]:
                continue
            manifest = {"round": 3, "selection_split": "val", "smoke_test_only": False,
                        "data_sha256": result["protocol"]["data_sha256"], "split_hashes": result["protocol"]["split_hashes"],
                        "checkpoint": item["checkpoint"], "checkpoint_sha256": item["checkpoint_sha256"],
                        "steps": winner["steps"], "report_sizes": [1, 3], "seed_start": 6180,
                        "training_seed": item["seed"], "schedule": item["schedule"],
                        "confirmed_selection_sha256": sha(args.output / "confirmed-selection.json")}
            dump(args.output / f"final-seed-{item['seed']}.json", manifest)
    print("ROUND3_VALIDATION_SELECTION", args.output, flush=True)


if __name__ == "__main__":
    main()
