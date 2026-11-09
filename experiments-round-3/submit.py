"""Compute-node preflight and journaled, one-time submission of round-three screening."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def submit_job(command):
    response = subprocess.run(command, cwd=HERE, check=True, capture_output=True, text=True).stdout.strip()
    job = response.split(";")[0]
    if not job.isdigit():
        raise RuntimeError("Unexpected sbatch response; inspect the journal and scheduler before retrying")
    return job


def verify():
    manifest = json.loads((ROOT / "source-manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("Uploaded source checksum mismatch: " + name)
    campaign = json.loads((HERE / "campaign.json").read_text())
    data = Path("/courses/CS6180.202710/shared/final_proj/em2d_task.h5")
    if digest(data) != campaign["dataset_sha256"]:
        raise RuntimeError("Course dataset checksum mismatch")
    import torch
    if torch.version.cuda is None:
        raise RuntimeError("CUDA-enabled PyTorch is required for subsequent GPU jobs")
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(HERE), "-p", "test_round3.py", "-v"], cwd=ROOT, check=True)
    return {"source_manifest_sha256": digest(ROOT / "source-manifest.json"),
            "source_commit": manifest["source_commit"], "source_snapshot": manifest["source_snapshot"],
            "dataset_sha256": campaign["dataset_sha256"], "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda, "python": sys.executable,
            "screening_tasks": [s["name"] for s in campaign["schedules"]],
            "parallel_limit": 2, "confirmation": "not submitted; review screening first"}


def submit():
    journal = ROOT / "submission.json"
    # Exclusive before any checks or sbatch. An uncertain outcome is never retried blindly.
    with journal.open("x") as stream:
        json.dump({"state": "preflighting", "preflight_job_id": os.getenv("SLURM_JOB_ID")}, stream)
    record = json.loads(journal.read_text())
    try:
        record.update(verify())
        out = ROOT / "runs" / "screen"
        out.mkdir(parents=True, exist_ok=True)
        (ROOT / "slurm").mkdir(exist_ok=True)
        record.update(state="submitting_screen", output_root=str(out))
        write(journal, record)
        exported = f"ALL,MICROSTRIP_PYTHON={sys.executable},ROUND3_PHASE=screen,ROUND3_OUTPUT={out}"
        array = submit_job(["sbatch", "--parsable", "--array=0-4%2", "--export=" + exported,
                            "--output=" + str(ROOT / "slurm/screen-%A_%a.log"), str(HERE / "run.sbatch")])
        record.update(state="screen_submitted", array_job_id=array)
        write(journal, record)
        record["state"] = "submitting_report"
        write(journal, record)
        report = submit_job(["sbatch", "--parsable", "--dependency=afterok:" + array,
                             "--export=ALL,ROUND3_CAMPAIGN_ROOT=" + str(ROOT),
                             "--chdir=" + str(ROOT), "--output=" + str(ROOT / "slurm/report-%j.log"),
                             str(HERE / "report_cpu.sbatch")])
        record.update(state="submitted", report_job_id=report)
        write(journal, record)
        print(json.dumps(record), flush=True)
    except Exception as error:
        record.update(state="needs_inspection", error=str(error))
        write(journal, record)
        raise


def report():
    journal = json.loads((ROOT / "submission.json").read_text())
    out = Path(journal["output_root"])
    campaign = json.loads((HERE / "campaign.json").read_text())
    runs = [out / f"{s['name']}-seed6180" for s in campaign["schedules"]]
    subprocess.run([sys.executable, str(HERE / "report.py"), "--root", str(out), "--output", str(ROOT / "report")], check=True)
    subprocess.run([sys.executable, str(HERE / "choose.py"), "screen", "--runs", *map(str, runs),
                    "--output", str(ROOT / "screen-decision")], check=True)
    write(ROOT / "screen-complete.json", {"status": "complete", "array_job_id": journal["array_job_id"],
        "confirmation_plan": str(ROOT / "screen-decision/confirmation-plan.json"), "confirmation_submitted": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["submit", "report"])
    args = parser.parse_args()
    {"submit": submit, "report": report}[args.action]()
