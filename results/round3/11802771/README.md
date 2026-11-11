# Round 3 completed learning-rate screen

All five schedules completed 220,500 updates at training seed 6180. Warmup + plateau resumed at update 79,062 using its existing full checkpoint and finished at epoch 500. The [original screening snapshot](../11716516/) records the state before continuation.

| Schedule | Best-field epoch | DDIM steps | Validation relative L2 | Global masked MSE |
|---|---:|---:|---:|---:|
| Warmup + cosine | 460 | 20 | **4.874%** | **5.747883e-5** |
| Warmup + multistep | 460 | 20 | 4.934% | 5.782798e-5 |
| Warmup + plateau | 480 | 20 | 5.142% | 5.788888e-5 |
| Warmup + constant | 480 | 20 | 5.145% | 5.791446e-5 |
| Constant | 460 | 20 | 5.263% | 6.022856e-5 |

Each row uses the best-field checkpoint and the full 881-design validation split. Relative L2 averages per-design errors across three single draws with sampling seeds 6180–6182. The screening rule compares global masked MSE at DDIM-20 and DDIM-50 and uses the lower value for each schedule. All five selected DDIM-20.

Cosine's global validation MSE was 4.57% below constant. The [frozen confirmation plan](confirmation-plan.json) selects cosine and constant for training seeds 6181 and 6182, alongside their existing seed 6180 runs. The CPU report job records confirmation as pending. The table reports screening validation at one training seed.

Slurm recorded the [continuation](plateau-resume-11802771.log) as completed in 2:16:22 and the [CPU report](report-11807152.log) in 16 seconds. The [launch record](resume-launch.json) identifies the backup of the original plateau artifacts. The resumed segment recorded 2.16 training hours and 5.62 minutes for full validation. The trainer's cumulative elapsed time includes the original checkpoint's counter, which differs from the original job's recorded wall time. Use the [Slurm accounting](accounting.txt) for job durations.

The [validation CSV](validation.csv) includes all 20 checkpoint and sampler combinations. The [learning-rate history](lr-history.csv) includes 2,500 completed epochs across the five runs. The final plateau [history](history.jsonl) preserves the original partial-epoch record before appending the continuation.

The [artifact checksums](hpc-artifact-sha256sums.txt) match the files transferred over SSH. The [HPC run manifest](hpc-run-sha256sums.txt) records all five run histories and checkpoint hashes. The 10 evaluated checkpoint hashes agree with their validation records. Both CSVs were regenerated from the verified inputs and compared with the HPC output after normalizing run paths. The [verification record](provenance.json) covers file integrity and internal consistency. Evaluation metrics come from the HPC runs.
