# Recover the incomplete Round 3 plateau schedule

Status as of November 10, 2026: recovery is prepared but has not been submitted. SSH access is restored and verified. The current [results and scheduler records](../../results/round3/11716516/) show that plateau remains at 79,062 of 220,500 updates; its Slurm task ended with `NODE_FAIL` and the dependent report job was cancelled.

Recovery procedure:

1. Inspect `squeue -u he.zit` and `sacct -j 11716516,11716517`. Do not submit if any original plateau task or continuation is pending/running. Inspect the latest history, complete.json and last.pt; do not assume the November 9 snapshot is still current.
2. Upload this recovery folder into the original campaign's `experiments/round3_recovery/`. Do not edit the frozen round-three training sources: they are part of the checkpoint's resume signature.
3. From the original campaign root, submit exactly one `sbatch experiments/round3_recovery/resume.sbatch`. Record its job ID immediately. The worker also rejects a concurrent recovery and preserves the previous metadata and all three checkpoints under `recovery/` before invoking the unchanged trainer.
4. The worker allows five additional training hours within an eight-hour allocation, retaining the 500-epoch horizon, LR schedule and exact full recovery state. If the segment stops short of 220,500 updates, inspect its new checkpoint and scheduler state before authorizing/submitting another segment. A time cap is never convergence.
5. After `equal_update_budget_complete: true`, verify all five runs, then run the existing CPU report/selection workflow. The original afterok dependency may be stuck on the failed original array; inspect it rather than assuming it will recover automatically. Do not launch multi-seed confirmation or held-out scoring without reviewing the frozen screening plan.

The script checks source hashes against the existing source manifest and refuses mismatched checkpoint/config/data/update counters. It resumes in the original output directory, preserving optimizer, EMA, scheduler and random state through the existing trainer. Existing four completed groups are untouched.

[Official SSH instructions](https://rc.northeastern.edu/2026/10/23/ssh-instructions-for-explorer/).
