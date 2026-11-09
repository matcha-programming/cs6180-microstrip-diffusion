# Round-three screening launch

Submitted on November 7, 2026. This is a launch receipt, not completed experiment results.

| Job | Purpose | Verified status at handoff |
| --- | --- | --- |
| 11716507 | CPU source/data preflight and eight tests | COMPLETED, exit 0 |
| 11716516_0 | Constant LR | RUNNING on d1004 |
| 11716516_1 | Warmup + constant LR | RUNNING on d1004 |
| 11716516_2 | Warmup + cosine LR | Queued behind the two-task concurrency limit |
| 11716516_3 | Warmup + multistep LR | Queued behind the two-task concurrency limit |
| 11716516_4 | Warmup + plateau LR | Queued behind the two-task concurrency limit |
| 11716517 | Validation CSVs and frozen confirmation plan | Waiting for successful screening |

Both active workers record Tesla V100-SXM2-32GB GPUs, 5,602,322 parameters and a 220,500-update target. Screening uses fresh seed 6180 models; confirmation and held-out evaluation were not submitted.

Remote campaign root:

```text
/scratch/he.zit/cs6180-microstrip-experiments/round3-schedules-20261107-ec9dcf5e41-b5b556
```

Training outputs are under `runs/screen/`, scheduler logs under `slurm/`, and the dependent CPU job will write `report/`, `screen-decision/confirmation-plan.json`, and `screen-complete.json` after all five training tasks finish successfully. A time-capped incomplete run requires recovery before selection.

The original SSH submission response was lost and the scheduler registration appeared late. An additional CPU preflight, 11716512, was stopped by exclusive creation of the existing submission journal before it could submit GPU work. Exactly one GPU array was submitted. That failed preflight is an orchestration duplicate, not a model-training failure.

`submission.json` is the copied cluster journal. `source-manifest.json` records the 38 uploaded file hashes and working-tree snapshot. `deployment.json` records archive verification, observed states and the submission recovery. The original source/results and the user's comparison workbook were not changed by deployment.

Check current status:

```sh
ssh explorer 'squeue -j 11716516,11716517'
```
