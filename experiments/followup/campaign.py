"""Submit the existing round-two protocol once; freeze validation before held-out scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = "/courses/CS6180.202710/shared/final_proj/em2d_task.h5"
RESUME = "/scratch/he.zit/cs6180-microstrip-experiments/paired-4r-20261103-v2/runs/11604740/diffusion/last.pt"
TASKS = [("draws", 0), ("draws", 1), ("continue", 0), ("lr", 0), ("lr", 1), ("lr", 2), ("min-snr", 0)]


def run(args, **kw):
    return subprocess.run([str(x) for x in args], cwd=ROOT, check=True, text=True, **kw)


def write(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def preflight():
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError("Package source mismatch: " + name)
    spec = json.loads((ROOT / "experiments-round-2/campaign.json").read_text())
    if digest(DATA) != spec["dataset_sha256"]:
        raise RuntimeError("Course dataset checksum mismatch")
    import torch
    if torch.version.cuda is None:
        raise RuntimeError("CUDA-enabled PyTorch is required")
    checkpoint = torch.load(RESUME, map_location="cpu", weights_only=False)
    if not {"model", "ema", "optimizer", "numpy_rng", "torch_rng"}.issubset(checkpoint):
        raise RuntimeError("Full continuation checkpoint unavailable")
    # Run the same migration checks used by the actual worker before any GPU allocation.
    sys.path.insert(0, str(ROOT / "experiments-round-2"))
    import train
    cfg = {"base": 36, "seed": 6180, "batch_size": 16, "backbone": "paired", "learning_rate": .0002,
           "mixed_precision": False, "loss_weighting": "none", "smoke_test_only": False,
           "data_sha256": spec["dataset_sha256"]}
    train.validate_resume(checkpoint, cfg)
    return {"source_commit": manifest["source_commit"], "dataset_sha256": spec["dataset_sha256"],
            "resume_sha256": digest(RESUME), "resume_epoch": checkpoint["epoch"],
            "resume_elapsed_seconds": checkpoint["elapsed"], "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda, "tasks": TASKS, "gpu_type": "v100-sxm2", "parallel_limit": 2}


def submit():
    journal = ROOT / "submission.json"
    if journal.exists():
        raise RuntimeError("Submission journal already exists; inspect it instead of duplicating jobs")
    record = preflight()
    # Exclusive creation before sbatch: unknown outcomes require inspection, never blind retries.
    with journal.open("x") as f:
        json.dump({**record, "state": "submitting"}, f, indent=2)
    array = run(["sbatch", "--parsable", "--array=0-6%2", HERE / "round2_worker.sbatch"], capture_output=True).stdout.strip().split(";")[0]
    if not array.isdigit():
        raise RuntimeError("Unexpected sbatch response; inspect the scheduler")
    record.update(state="array_submitted", array_job_id=array)
    write(journal, record)
    final = run(["sbatch", "--parsable", "--dependency=afterok:" + array, HERE / "round2_final.sbatch"], capture_output=True).stdout.strip().split(";")[0]
    if not final.isdigit():
        raise RuntimeError("Unexpected dependent sbatch response; inspect the scheduler")
    record.update(state="submitted", final_job_id=final)
    write(journal, record)
    print(json.dumps(record), flush=True)


def worker():
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    stage, index = TASKS[task]
    # The first array task may start before submit() has persisted sbatch's reply.
    out = ROOT / "runs" / ("round2-" + os.environ["SLURM_ARRAY_JOB_ID"])
    args = [sys.executable, ROOT / "experiments-round-2/run.py", "--stage", stage,
            "--task-index", index, "--data", DATA, "--output", out]
    if stage == "continue":
        args += ["--resume", RESUME]
    out.mkdir(parents=True, exist_ok=True)
    gpu = run(["nvidia-smi"], capture_output=True).stdout
    (out / f"gpu-task-{task}.txt").write_text(gpu)
    run(args)


def finalize():
    record = json.loads((ROOT / "submission.json").read_text())
    out = ROOT / "runs" / ("round2-" + record["array_job_id"])
    candidates = [out / n for n in ("R2-B-paired-long", "R2-C-lr-0.0001", "R2-C-lr-0.0002", "R2-C-lr-0.0004", "R2-D-min-snr-5")]
    selection = out / "selection.json"
    if selection.exists():
        raise RuntimeError("Final selection already exists; inspect before a deliberate recovery")
    run([sys.executable, ROOT / "experiments-round-2/freeze_selection.py", "--runs", *candidates,
         "--report-sizes", 1, 3, "--output", selection])
    chosen = json.loads(selection.read_text())
    for split in ("test", "test_extrap"):
        run([sys.executable, ROOT / "experiments-round-2/draw_study.py", "--data", DATA,
             "--checkpoint", chosen["checkpoint"], "--steps", chosen["steps"], "--sizes", 1, 3,
             "--limit", 0, "--split", split, "--confirm-test", "--selection", selection,
             "--output", out / ("final-" + split)])
    run([sys.executable, ROOT / "experiments-round-2/report.py", "--root", out, "--output", out / "report"])
    write(out / "campaign-complete.json", {"status": "complete", "selection_sha256": digest(selection),
                                           "array_job_id": record["array_job_id"], "final_job_id": record["final_job_id"]})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["preflight", "submit", "worker", "finalize"])
    action = p.parse_args().action
    if action == "preflight":
        print(json.dumps(preflight(), indent=2))
    else:
        {"submit": submit, "worker": worker, "finalize": finalize}[action]()
