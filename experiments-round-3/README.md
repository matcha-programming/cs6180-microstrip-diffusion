# Round 3: learning-rate schedules for direct conditional diffusion

Status: screening ran on HPC. Four schedules completed 220,500 updates; plateau remains incomplete. See the [verified results, histories and execution records](../results/round3/11716516/). See the [project plan](../docs/project-plan.md). These are fresh training experiments using the executed position-conditioned direct DDPM. It generates the complete field; it is not residual diffusion and does not load a regression predictor.

## Evidence used for the design

Round-two full-validation results, with checkpoint/sampler choices from validation only:

| Fresh constant LR | Best full-validation global field MSE | Epoch budget |
| --- | ---: | ---: |
| 0.0001 | 0.000074680 | 200 |
| 0.0002 | 0.000095902 | 200 |
| 0.0004 | 0.000144530 | 200 |

The longer unweighted LR 0.0002 continuation reached about 0.000061179 at epoch 500. Min-SNR at LR 0.0002 reached about 0.000061706 at epoch 200. Source: [round-two validation artifacts](../results/round2/11638349/). The continuation has a different initialization/training history from the fresh runs. These observations motivate longer training, a peak LR of 0.0001, and a separate schedule experiment. They do not demonstrate an optimal peak rate or schedule.

The primary experiment holds the loss unweighted to isolate scheduling. Min-SNR, input changes, more data, higher resolution, model capacity and Fourier work are excluded. Existing held-out scores are not used to choose the schedule matrix. The original unpaired direct DDPM and the position-conditioned direct DDPM are different controls; this round uses the paired backbone throughout.

## Five schedule controls

Common peak LR: **0.0001**. Decay floor: **0.000001**. Warmup starts at **0.00001** and reaches the peak over the first **five epoch equivalents / 2,205 successful updates**, where applicable.

| ID | Schedule | After warmup | Question |
| --- | --- | --- | --- |
| R3-S0 | Constant, no warmup | LR 0.0001 throughout | Fresh long-budget reference |
| R3-S1 | Linear warmup + constant | Hold LR 0.0001 | Does warmup itself help? |
| R3-S2 | Linear warmup + cosine | Smooth decay to 0.000001 at the final update, no restarts | Does gradual late refinement help? |
| R3-S3 | Linear warmup + multistep | Multiply LR by 0.1 after 60% and 85% of updates | Are discrete reductions effective? |
| R3-S4 | Linear warmup + plateau | Halve LR after three consecutive field checks without a 0.1% relative improvement; floor 0.000001 | Does validation-driven decay help? |

S0 versus S1 isolates warmup. S1 versus S2/S3/S4 isolates the subsequent decay rule. Every run shares peak LR, architecture, data, initialization seed and random-number streams. Cosine and multistep schedules are indexed by successful optimizer updates rather than scheduler wall time or consumed minibatches. The plateau rule observes generated field MSE after warmup, never training loss, noise validation MSE or test results.

`schedules.py` defines the exact indexing. LR for update 1 is the warmup start; update 2,205 reaches the peak. The first postwarmup cosine update also uses the peak; update 220,500 uses the floor. Multistep changes occur at updates **132,301** and **187,426**, just after epoch equivalents 300 and 425. Skipped AMP updates do not advance the schedule, although the declared campaign uses FP32.

The plateau implementation follows the field-metric/patience convention of [ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html): patience 2 means the third bad observation triggers reduction. The cosine shape follows [cosine annealing without restarts](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html). These are experimental choices, not guarantees for this task.

## Common training protocol

- Original fixed splits: 7,048 train, 881 validation, 881 test and 979 extrapolation. The target remains two channels on 64 x 128.
- Executed paired U-Net, base width 36, **5,602,322 parameters**, fixed row/column index planes and the same four design inputs.
- AdamW, betas 0.9/0.999, weight decay zero, batch size 16, gradient clipping 1, FP32, EMA 0.999.
- Masked epsilon MSE without Min-SNR weighting; original linear 1,000-step noise schedule 0.0001..0.02. Training-only field standardization, component clipping and mask handling match round two.
- **220,500 successful optimizer updates**, equivalent to 500 complete epochs at 441 updates/epoch. Early stopping is disabled. A low LR late in training must have time to act.
- Six-hour training/periodic-validation safety cap per run, eight-hour Slurm allowance including final validation. A capped run is marked incomplete and excluded from primary schedule selection until its update budget is finished.
- Fresh models for every schedule. Do not import round-two weights or reset/change a cosine horizon during recovery. Native recovery restores optimizer, EMA, scheduler, permutation/cursor and RNG together.

All five schedules use model seed **6180** during screening. The common initial state hash is checked when comparing runs. Data corruption and shuffle streams are common across schedules; deterministic validation uses sample-ID-keyed Gaussian tensors from the existing round-two implementation.

At initialization and every 20 completed epochs, evaluate the same fixed 128-design validation panel with DDIM-50 and sampling seed 6180. The lowest globally accumulated masked field MSE selects `best-field.pt`. Save `best-noise.pt` separately as a diagnostic, but do not let noise-selected checkpoints win the primary schedule comparison. All schedules receive the same checkpoint opportunities. The initial validation score does not update the plateau controller.

After training, score retained checkpoints on all 881 validation designs with DDIM-20 and DDIM-50, using three sampling seeds 6180/6181/6182. Full-validation MSE is the average of single-draw squared-error scores, not MSE of an averaged field. Relative L2, edge and gradient errors use the unchanged metric functions. Field checks, noise validation and checkpointing are charged to the training budget; data loading and final full-validation confirmation are recorded separately.

Partial-epoch interruptions do not create extra validation/plateau observations. Time-limited runs retain their exact minibatch cursor. Schedules cannot be stretched or compressed to whatever number of updates fits in a time cap. If the cap prevents completion, revise/recover the common operational budget explicitly and retain the actual GPU-hours; do not label the shorter run an equal-update comparison.

## Screening, confirmation and selection

1. Run all five schedules at seed 6180. Rank each primary field-selected checkpoint by its best full-validation MSE among the two declared DDIM step counts.
2. Freeze a confirmation plan for the constant control and the strongest nonconstant challenger. Keep the constant control even if it won screening. Repeat both at seeds 6181 and 6182: four additional fresh runs, reusing the two seed-6180 runs.
3. Confirm using all three model-training seeds per method. For each method, choose one common DDIM step count using mean validation MSE across training seeds. Select the winning schedule by this mean, with the declared constant-first tie rule. Report training-seed standard deviation separately from sampling variability. Do not pick the single luckiest trained model.
4. Freeze one final manifest for each of the winning schedule's three trained models. Evaluate each on both held-out splits using one and three draws. Retain all three individual scores so the mean single-draw accuracy can be compared with historical reports; ensemble-of-three accuracy has its own cost.

`choose.py` rejects incomplete budgets, smoke data, changed checkpoints, mismatched hardware/protocols and duplicate schedule/seed identities. The original screen checkpoint/config hashes must still match the frozen confirmation plan. All results, including losing schedules, remain in the report. Selection is based on validation; the existing repeatedly reported test sets are descriptive historical benchmarks. New Palace cases would be needed for a genuinely untouched external confirmation set.

## Runtime and files

The round-two fresh runs took about 1.36 V100 training hours for 200 epochs. A rough extrapolation is **3.4 V100 hours per 500-epoch run**, subject to validation and checkpoint overhead. Screening is approximately 17 training GPU-hours, with a maximum declared allowance of 30. Four confirmation runs add approximately 14, with a maximum allowance of 24. Full-validation and held-out sampling are additional. These estimates are not measured round-three runtimes. Slurm arrays limit concurrency to two V100s in the commands below.

`campaign.json` fixes the design. `train.py` versions the round-two loop with explicit scheduler/recovery state; round-two source is not modified. `r3_support.py` reuses its numerical helpers and records both rounds' code hashes. `run.py` materializes exact commands, `choose.py` freezes stage decisions, `report.py` exports validation/LR histories, and `evaluate.py` invokes the unchanged held-out sampler with round-three provenance.

Keep both `experiments-round-2/` and `experiments/remaining/` alongside this folder on HPC. Python dependencies are in `requirements.txt`. Use the same V100-SXM2 GPU model and FP32 policy across the primary comparison. Do not modify source while jobs are running: hashes are checked when resuming and comparing.

## Commands

For a verified uploaded source bundle containing `source-manifest.json`, the cluster deployment entrypoint is `submit_cpu.sbatch`, submitted from the bundle root. It runs dataset/source checks and the CPU tests on an allocated node, then journals a one-time five-task GPU array with concurrency two. A dependent CPU job builds validation tables and a frozen confirmation plan. It does not submit confirmation or held-out evaluations. `submission.json` records the actual job IDs and refuses duplicate submission; inspect the journal and scheduler if a submission response is uncertain.

The deployment scripts use the previously verified Explorer interpreter `/scratch/he.zit/envs/meep-gpu/bin/python`. The manual commands below allow selecting another environment explicitly.

From `experiments-round-3/`, preview all five commands by changing the task index from 0 through 4:

```sh
python run.py --phase screen --task-index 0 --output ../runs/round3 --dry-run
```

Submit screening with a selected CUDA environment and shared campaign root:

```sh
export MICROSTRIP_PYTHON=/path/to/cuda-environment/bin/python
export ROUND3_OUTPUT=/path/to/round3
sbatch --array=0-4%2 --export=ALL run.sbatch
```

After all five finish, use `choose.py screen --runs` with the five completed directories:

```sh
python choose.py screen --runs \
  /path/to/round3/R3-S0-constant-seed6180 \
  /path/to/round3/R3-S1-warmup-constant-seed6180 \
  /path/to/round3/R3-S2-warmup-cosine-seed6180 \
  /path/to/round3/R3-S3-warmup-multistep-seed6180 \
  /path/to/round3/R3-S4-warmup-plateau-seed6180 \
  --output /path/to/round3/screen-decision
export ROUND3_PLAN=/path/to/round3/screen-decision/confirmation-plan.json
sbatch --array=0-3%2 --export=ALL,ROUND3_PHASE=confirm,ROUND3_PLAN run.sbatch
```

The plan lists which challenger and four task/seed combinations were selected. No jobs are submitted by `choose.py` or a dry run. Confirm using exactly six run directories: constant/challenger at seed 6180, plus their four new confirmation runs:

```sh
python choose.py confirm --plan /path/to/confirmation-plan.json \
  --runs /six/completed/run/directories --output /path/to/round3/confirmed
```

Replace the placeholder with six separate directory arguments. This writes `confirmed-selection.json` and `final-seed-6180.json`, `final-seed-6181.json`, `final-seed-6182.json` for the selected schedule. In scheduled GPU allocations, evaluate each seed manifest separately:

```sh
python evaluate.py --selection /path/to/round3/confirmed/final-seed-6180.json \
  --split test --output /path/to/round3/final/seed6180-test
python evaluate.py --selection /path/to/round3/confirmed/final-seed-6180.json \
  --split test_extrap --output /path/to/round3/final/seed6180-extrap
python report.py --root /path/to/round3 --output /path/to/round3/report
```

Repeat final evaluation for seeds 6181 and 6182. Full checkpoints remain outside Git under the ignored root `runs/` or a cluster workspace. Publish compact final artifacts with checksums and source/config provenance under `results/round3/` later.

Recovery uses the original schedule and horizon:

```sh
python train.py --schedule warmup-cosine --data /path/to/em2d_task.h5 \
  --output /path/to/R3-S2-warmup-cosine-seed6180 \
  --resume /path/to/R3-S2-warmup-cosine-seed6180/last.pt
```

Supply the original nondefault settings if any. Round-two checkpoints and weights-only EMA files are rejected for round-three recovery. A separately labeled future fine-tuning experiment is needed to compare schedules on an existing checkpoint.

## Local verification

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s . -p test_round3.py -v
```

Checks cover LR endpoints and update boundaries, plateau patience/warmup, scheduler-state recovery, five-schedule CPU training with equal updates, bitwise model/RNG replay after a partial-epoch interruption, validation-based screening and seed-averaged confirmation, and rejection of incomplete runs. Synthetic CPU outputs are explicitly marked smoke-only and cannot enter scientific selection. Measured screening results are linked above; convergence and multi-seed confirmation remain open.
