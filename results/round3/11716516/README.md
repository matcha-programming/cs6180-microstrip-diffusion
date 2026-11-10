# Round 3 learning-rate screening: observed validation results

Campaign launched November 7, 2026; validation files inspected through authenticated Explorer Open OnDemand on November 9, 2026. **Four schedules completed their common update budget. Warmup + plateau remains incomplete; recovery is authorized but has not been submitted because SSH access is blocked.**

## Completed schedules

Primary `best-field` checkpoint, DDIM-20, full 881-design validation split. Relative L2 is the mean per-design error averaged over single draws with seeds 6180–6182; it is not the error of an averaged prediction. MSE below is the valid-element global MSE. These are validation scores, not held-out test scores.

| Schedule | Updates | Best-field epoch | Validation relative L2 | Global masked MSE | Training hours |
|---|---:|---:|---:|---:|---:|
| Constant | 220,500 | 460 | 5.263% | 6.022856e-5 | 3.398 |
| Warmup + constant | 220,500 | 480 | 5.145% | 5.791446e-5 | 3.414 |
| Warmup + cosine | 220,500 | 460 | **4.874%** | **5.747883e-5** | 3.398 |
| Warmup + multistep | 220,500 | 460 | 4.934% | 5.782798e-5 | 3.408 |

Cosine has the lowest primary validation MSE among these four runs, about 4.57% below constant. This is a provisional observation from one training seed. The predefined five-schedule screening and subsequent multi-seed confirmation are not complete. Reaching the update budget does not establish convergence.

[All 20 checkpoint/sampler combinations](validation.csv) retain DDIM-20 and DDIM-50, best-field and best-noise, edge and gradient errors, stopping reasons and actual budgets. DDIM-20 has lower global validation MSE than DDIM-50 for all four completed primary checkpoints. Noise-checkpoint results are diagnostic; they do not replace the predeclared best-field selection rule.

## Incomplete plateau schedule

The observed warmup + plateau run stopped for `training_budget`, with 79,062 of 220,500 updates (35.86%). Its best-field epoch 160 gives 6.739% validation relative L2 and 8.077325e-5 global MSE at DDIM-20. Its best-noise checkpoint is epoch 178. **Do not rank this shortened run against the completed schedules.** The word “plateau” here names its learning-rate schedule, not a measured convergence result.

Recovery must use the existing `last.pt` with model, EMA, AdamW, scheduler, RNG, epoch permutation and partial-epoch cursor. Preserve the original 220,500-update horizon and source. Extend only the training-time allowance; do not restart training or increase the epoch horizon. The existing trainer checks resume compatibility. [Recovery instructions](../../../experiments/round3_recovery/README.md).

## Evidence and limits

- [Launch receipt, original source manifest and submission journal](../launch-20261107-ec9dcf5e41-b5b556/) identify GPU array 11716516 and dependent report job 11716517. The recorded training protocol uses a paired PositionUNet, seed 6180, FP32, batch 16, peak LR 1e-4, minimum LR 1e-6, and a 500-epoch/220,500-update horizon. Launch logs reported V100-SXM2-32GB.
- The five `validation.json` files here are parsed snapshots transcribed from the rendered HPC JSON, preserving all observed fields and numerical values. Whitespace differs from the server files. [Provenance](provenance.json) records the source URLs and scope. Checksums cover these published snapshots, not independently downloaded HPC bytes.
- Each file contains four metric records, each with `designs: 881`, finite nonnegative metrics, checkpoint epochs and reported SHA256 values. Counts and fields were checked; full prediction reproduction, checkpoint-byte hashes, per-design IDs, masks and aggregation were **not independently revalidated** for this publication.
- Best checkpoints, full histories, configurations and optimizer recovery files remain on HPC. The browser download attempt did not produce local files. Their contents are not represented as downloaded or audited here. No datasets, caches or optimizer checkpoints are published.
- Recovery access checks found the official generated key owned by root with mode 600. SSH was denied; the alternative OOD Job Composer also failed initialization with `Disk quota exceeded` in the home directory. No recovery job was submitted and no files were deleted.
- The campaign root did not show `report/`, `screen-decision/` or `screen-complete.json` at inspection. The original report dependency must be checked after recovery; no successful final report is claimed. Confirmation and held-out evaluation were not submitted in the launch record, and no later results were found in this campaign directory.
- Do not infer calibrated physical uncertainty from diffusion draw variation. These are stored-target validation errors; no new Palace simulations or physical-truth claims are added.

HPC output root:

```text
/scratch/he.zit/cs6180-microstrip-experiments/round3-schedules-20261107-ec9dcf5e41-b5b556/runs/screen
```
