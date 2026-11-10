# Round 3 learning-rate screening results

Verified against the original HPC files on November 10, 2026. Four schedules completed 220,500 updates. Warmup + plateau remains incomplete at 79,062 updates and is excluded from the equal-budget comparison. These are screening validation results from one training seed; confirmation and held-out evaluation are pending.

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

The saved warmup + plateau run stopped for `training_budget`, with 79,062 of 220,500 updates (35.86%). Its best-field epoch 160 gives 6.739% validation relative L2 and 8.077325e-5 global MSE at DDIM-20. Its best-noise checkpoint is epoch 178. **Do not rank this shortened run against the completed schedules.** The word “plateau” here names its learning-rate schedule, not a measured convergence result.

Recovery must use the existing `last.pt` with model, EMA, AdamW, scheduler, RNG, epoch permutation and partial-epoch cursor. Preserve the original 220,500-update horizon and source. Extend only the training-time allowance; do not restart training or increase the epoch horizon. The existing trainer checks resume compatibility. [Recovery instructions](../../../experiments/round3_recovery/README.md).

## Scheduler status and timing

[Slurm accounting](slurm/accounting.txt) records tasks `11716516_0` through `_3` as `COMPLETED`, plateau task `_4` as `NODE_FAIL`, and dependent report job `11716517` as `CANCELLED`. No jobs were queued or running for this account at download.

The plateau training artifact reports `training_budget` and 6.634 elapsed training hours, while Slurm records 2:03:09 for the failed task. These records do not agree on elapsed time; neither is treated as verified active GPU time for that run. Its history ends in a partial epoch 180. The successful runs each report about 3.4 training hours and 3.5 hours of scheduler wall time.

SSH access was restored on November 10. No plateau continuation or new training was submitted as part of this results update. The five-schedule selection remains pending until plateau completes its original update budget.

## Files and verification

- [Validation table](validation.csv): all 20 checkpoint/sampler settings, including the incomplete plateau run.
- [Learning-rate history](lr-history.csv): 2,179 complete-epoch rows. Original histories also retain initial validation and the plateau partial-epoch record.
- Each schedule directory contains the original `config.json`, `validation.json`, `complete.json`, `history.jsonl`, and any additional recorded field-check history. Matching training logs sit beside these directories.
- [Slurm logs](slurm/) and [original submission/source records](provenance/) retain execution evidence.
- [Verification record](provenance.json) maps published paths to their HPC sources. All 45 downloaded files match SHA256 values computed on HPC. The five validation JSON objects match the previously published November 9 snapshots; this update replaces those transcriptions with the original bytes.
- All 10 evaluated checkpoint hashes match their validation records. Hashes for those checkpoints and the five `last.pt` files are in [the HPC manifest](provenance/hpc-verification.json). Checkpoint and optimizer files remain on HPC.
- All runs share their initial model hash, dataset hash, GPU model and 19 recorded source-file hashes. Source hashes match the repository. Reported update counts agree with histories and completion records; best-field checkpoint epochs agree with the lowest recorded selection-panel MSE. All 20 metric records contain 881 designs and finite, nonnegative values.

The CSV tables were regenerated locally with `experiments-round-3/report.py`; run paths were shortened to portable directory names. [Checksums](sha256sums.txt) cover every published file in this result directory except the checksum list itself.

These checks establish file integrity and consistency. Predictions, per-design aggregation and physical accuracy were not independently recomputed. Diffusion draw variation is not evidence of calibrated physical uncertainty. No new Palace simulations were run.

HPC output root:

```text
/scratch/he.zit/cs6180-microstrip-experiments/round3-schedules-20261107-ec9dcf5e41-b5b556/runs/screen
```
