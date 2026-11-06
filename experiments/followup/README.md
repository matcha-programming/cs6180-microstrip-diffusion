# Follow-up experiments

This folder audits the proposed physical similarity coordinates and launches the existing [Round 2 protocol](../../experiments-round-2/README.md). It does not replace that protocol or change the executed model architecture.

## Completed checks

- [Fixed-split support audit](../../results/followup/support-20261104/README.md): metadata only; original splits unchanged.
- [Validation-only kNN controls](../../results/followup/knn-validation-20261104/README.md): six predeclared candidates, fitted using training designs only.
- The existing 13 Round 2 integration tests pass, including actual paired initialization, optimizer continuation guards, design-keyed Gaussian noise, masked losses, the DDIM oracle, and unbiased draw variance. Seven additional checks cover geometric support, exact-neighbor weighting, and duplicate-submission protection after scheduler failures.

## GPU campaign

`campaign.py submit` verifies packaged source hashes, the course dataset, the full historical optimizer checkpoint, and the existing continuation migration checks before submitting anything. It refuses an existing submission journal. A Slurm array runs the two historical draw studies, continuation, three fresh learning-rate controls, and the optional Min-SNR trial. At most two V100 GPUs run concurrently. No DeepONet jobs are submitted because a teammate is already investigating that branch of the comparison.

The eight-hour scheduler limit is a safety bound for each task. Scientific training limits remain the predeclared limits in `experiments-round-2/campaign.json`: continuation to epoch 500 or six total hours including the historical two; fresh trials to epoch 200 or three hours each. Full validation confirmation is additional and recorded separately. The new training budget is at most about sixteen GPU-hours across five training trials, plus sampling and validation. A time-capped run is not labeled converged.

A dependent GPU job runs only after every array task succeeds. It freezes the checkpoint and DDIM steps using all 881 validation designs, then evaluates the selected configuration on the unchanged test and `test_extrap` splits with one and three draws. It never changes choices based on those held-out errors. The report includes all learning-rate and loss-weighting trials, including unsuccessful ones. Independent training seeds remain necessary before a claim about either method category.

Submit from the extracted campaign root on Explorer. The preflight imports CUDA-enabled PyTorch and loads a full optimizer checkpoint, so it runs on a scheduled CPU node with 16 GB RAM rather than on the memory-limited login node:

```sh
sbatch experiments/followup/submit_cpu.sbatch
```

`source-manifest.json` is generated during packaging from the committed source and is required on HPC. `submission.json` records actual scheduler IDs; do not infer successful submission from these instructions. A partial or unknown submission outcome requires inspecting that journal and `squeue` before recovery. The original runs and optimizer checkpoints are read only.

## Local reproduction

```sh
python experiments/followup/audit_support.py --data data/em2d_task.h5 --output runs/support
python experiments/followup/knn_control.py --data data/em2d_task.h5 --output runs/knn-validation
python -m unittest discover -s experiments/followup -p 'test_*.py' -v
```

The kNN implementation reads field targets for training and validation only. Its global masked validation MSE selects between original inputs and proposed similarity coordinates, and between 1, 5 and 15 neighbors. These are validation results; no held-out kNN score is reported or implied.

The [next-step plan](../../docs/next-steps-20261104.md) covers solver validation and broader geometries after this launch.

`physics-pilot.json` freezes eight metadata-selected training/validation designs and 32 solver configurations. It is a prepared pilot, not a simulation result. Palace was not found on Explorer's default PATH or in `module -t avail palace` during launch preparation.

## Additional R2-A/C/D held-out reporting

`evaluate_holdout.py` evaluates the saved R2-A-original, R2-A-paired, three R2-C learning-rate trials, and R2-D checkpoint on all 881 test and 979 extrapolation designs. It does not train or change the original campaign. R2-C/D choices are frozen independently from each trial's saved full-validation scores. R2-A keeps its predeclared historical checkpoints and 20/50-step samplers, and records that this is not a new full-validation checkpoint search.

All six cases report K=1 and K=3 with the original sample-ID-keyed seeds starting at 6180. R2-A additionally reports K=2/4/8/16. The existing globally selected R2-B reports supply the reference. Each method's sampler steps are fixed before these additional test scores are accessed. Do not use the resulting held-out scores for retuning.

The three commands below belong in Slurm jobs, using a CUDA-enabled environment. `prepare` runs on a CPU allocation, `worker` on one V100-SXM2 per task, and `pack` on a CPU allocation after all six workers succeed:

```sh
python evaluate_holdout.py prepare --configuration /path/to/configuration.json
python evaluate_holdout.py worker --output-root /path/to/evaluation --task-index 0
python evaluate_holdout.py pack --output-root /path/to/evaluation
```

The configuration records the original source root, campaign root, dataset path, output root, and reviewed source hashes. Preparation verifies those hashes, the dataset and checkpoints, and the existing R2-B reference before writing any frozen selections. The worker enforces the frozen settings through the unchanged round-two sampler's held-out guards. The output contains `plan.json`, one selection per case, both split reports, per-design metrics, fixed examples, logs, checksums, and a final ZIP archive. Original campaign files are read-only.

`collect_holdout.py --journal /path/to/submission.json` is a one-shot local monitor for an already submitted campaign. It reads scheduler status and a compact, compute-generated summary through the established Explorer session, verifies the summary checksums, imports the metrics/configurations/manifests into `results/round2/11638349/heldout-TIMESTAMP/`, and refreshes the comparison. `import-scope.json` identifies the full cluster archive and makes clear that per-design arrays are not part of this compact local import. The collector exits when this batch completes; it does not install a recurring schedule.

Lightweight orchestration checks do not require PyTorch:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s experiments/followup -p test_holdout.py -v
```
