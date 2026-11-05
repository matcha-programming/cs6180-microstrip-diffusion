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

Run from the extracted campaign root on Explorer:

```sh
/scratch/he.zit/envs/meep-gpu/bin/python experiments/followup/campaign.py preflight
/scratch/he.zit/envs/meep-gpu/bin/python experiments/followup/campaign.py submit
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
