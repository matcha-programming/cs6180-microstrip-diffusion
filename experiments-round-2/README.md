# Round 2: diffusion training and sampling

Runnable next experiments for the [project plan](../docs/project-plan.md). This round contains no Fourier search, regression retraining, new data generation, or change to the executed U-Net architecture. It reuses the original model and metric source while keeping round-one runs unchanged.

## What has already been done

| Run | Direct diffusion | Relevant result |
| --- | --- | --- |
| 11603443 | 5,601,674 parameters; no position planes; V100/AMP; 200 epochs | DDIM-20: 6.32% test / 4.48% extrapolation single-draw mean error |
| 11604740 | 5,602,322 parameters; fixed position planes; P100/FP32; selected epoch 158 | DDIM-50: 7.56% / 5.81%; training hit two hours during epoch 159 |

The paired deterministic U-Net scored 2.87% / 1.15%. The old diffusion pipeline selected checkpoints using noise MSE, then selected sampler steps using generated fields. Existing three-draw ensembles, per-draw design scores, and four examples per split are available, but per-design unbiased pixel variance is missing. Recovery files with optimizer state remain on HPC; published checkpoints contain selected EMA weights only.

## Predeclared tasks

`campaign.json` is the machine-readable source for `run.py`.

| Task | Action | Maximum training budget |
| --- | --- | --- |
| R2-A-original | Existing original direct checkpoint; DDIM-20; 1/2/4/8/16 draws | No training |
| R2-A-paired | Existing paired checkpoint; DDIM-50; 1/2/4/8/16 draws | No training |
| R2-B-paired-long | Continue the paired last.pt at LR 0.0002 | Epoch 500 or six hours **including historical training time** |
| R2-C-lr-0.0001 | Fresh paired U-Net | 200 epochs or three hours |
| R2-C-lr-0.0002 | Fresh paired U-Net | 200 epochs or three hours |
| R2-C-lr-0.0004 | Fresh paired U-Net | 200 epochs or three hours |
| R2-D-min-snr-5 (optional) | Fresh paired U-Net; Min-SNR gamma 5; LR 0.0002; control is R2-C-lr-0.0002 | 200 epochs or three hours |

Fresh LR trials have the same seed 6180 and actual initial weights. They do not warm-start from the continued model. The LR 0.0002 fresh trial is essential: the continued model has a different optimization history and cannot substitute for it. Actual GPU-hours and optimizer updates are retained. The array can use up to nine training hours across three GPUs; continuation adds at most about four training hours beyond its recorded two. Add separate full-validation and sampling time. The scheduler's eight-hour request is a wall-time allowance, not a promise that every task consumes eight GPU-hours.

The unchanged paired configuration is base width 36, batch 16, FP32, AdamW (betas 0.9/0.999, no weight decay), gradient clipping 1, EMA 0.999, masked epsilon MSE, 1,000 linear beta steps 0.0001..0.02, component clipping [-1,1], and mask zero at each DDIM step. Design conditioning and fixed row/column planes match the executed 4R pair. No frozen regression checkpoint is loaded by these commands.

### Optional Min-SNR ablation

R2-D changes only timestep loss weights relative to the fresh LR 0.0002 control. Min-SNR is a candidate, not an established improvement on these fields. Its rationale is to balance optimization across noise levels; the [paper](https://arxiv.org/abs/2303.09556) reports improvements on image tasks. Use the epsilon-prediction formula from the [authors' code](https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/main/guided_diffusion/gaussian_diffusion.py):

```text
SNR(t) = alpha_bar(t) / (1 - alpha_bar(t))
w(t) = min(SNR(t), 5) / SNR(t)
loss = sum(valid * w(t) * (predicted_noise - noise)^2) / (2 * sum(valid))
```

The valid-element denominator is unchanged; weights are not renormalized. This epsilon-specific formula differs from the weight for predicting the clean field. It reduces high-SNR timestep contributions while preserving low-SNR weights at one. Neither the noise schedule nor timestep sampling changes. Gamma 5 is one predeclared starting value, not a tuned optimum.

The weighted objective, ordinary noise MSE and average timestep weight are logged separately; weighted and unweighted objective values cannot establish an accuracy advantage. Checkpoints and sampler selection still use the same generated-field validation protocol. `best-noise.pt` continues to track ordinary, unweighted validation noise MSE. CSV reports and the frozen selection manifest label weighting and gamma explicitly.

R2-D adds at most three training GPU-hours plus final validation. Run it as a fresh model with the same seed and initialization as R2-C-lr-0.0002. Recovery rejects changes to weighting or gamma. A positive finding merits independent training seeds and a later learning-rate interaction study; a negative finding at this one gamma/LR does not rule out the method generally.

## Setup and preview

Use Python with CUDA PyTorch, NumPy, SciPy, and h5py; see `requirements.txt`. CUDA runs refuse a CPU fallback. The same code supports explicitly marked synthetic CPU smoke tests. Keep this folder at the repository root so it can import the existing model/metric implementation.

From `experiments-round-2/`:

```sh
python run.py --stage draws --task-index 0 --output ../runs/round2 --dry-run
python run.py --stage lr --task-index 0 --output ../runs/round2 --dry-run
python run.py --stage min-snr --output ../runs/round2 --dry-run
python run.py --stage continue --resume /path/to/diffusion/last.pt --output ../runs/round2 --dry-run
```

The campaign launcher verifies the course HDF5 checksum. Standalone workers record their dataset hash, split hashes, source hashes, versions, hardware, seeds, and selected rows. All numerical tuning defaults to validation; the CPU-resident data loader does not load test targets for training or validation studies. Only each minibatch is moved to the GPU.

## Submit on Explorer

Select an environment rather than relying on another user's hard-coded interpreter path:

```sh
export MICROSTRIP_PYTHON=/path/to/cuda-environment/bin/python
sbatch --array=0-1 --export=ALL,ROUND2_STAGE=draws run.sbatch
sbatch --array=0-2 --export=ALL,ROUND2_STAGE=lr run.sbatch
```

Optional weighting experiment, using the same interpreter and campaign root convention:

```sh
sbatch --export=ALL,ROUND2_STAGE=min-snr run.sbatch
```

The verified source-run provenance places recovery under:

```text
/scratch/he.zit/cs6180-microstrip-experiments/paired-4r-20261103-v2/runs/11604740/diffusion/last.pt
```

Check that the recovery file exists in the accessible cluster workspace. Then:

```sh
export ROUND2_RESUME=/path/to/11604740/diffusion/last.pt
sbatch --export=ALL,ROUND2_STAGE=continue,ROUND2_RESUME run.sbatch
```

Default outputs are `../runs/round2-ARRAY_OR_JOB_ID/TASK_NAME/`. `ROUND2_OUTPUT` can override the campaign root; distinct task names permit draws, LR, and continuation to share that root. Each task writes its own log and launch status. `ROUND2_DATA` can override the cluster data path, but its content must match the course checksum. GPU and CPU details are saved per task. Use the same GPU model for direct speed comparisons; record different hardware as a limitation.

`best-diffusion.pt` cannot replace `last.pt`: continuation validates optimizer, EMA, RNG, normalization, architecture, precision, learning rate, source, and dataset. It refuses a silent weights-only restart. If HPC recovery is unavailable, a separately labeled fresh long run can be started with `train.py --epochs 500 --max-train-hours 6`; report its actual initialization and history.

Legacy last.pt did not save a partial-epoch permutation/cursor. Migration preserves model, EMA, optimizer, scaler and RNG, begins the next epoch, and records that the final epoch tail cannot be replayed exactly. No exact historical update count is claimed. Native recovery stores permutation/cursor and supports a mid-epoch restart:

```sh
python train.py --data /path/to/em2d_task.h5 --output /path/to/round2/R2-B-paired-long \
  --resume /path/to/round2/R2-B-paired-long/last.pt --epochs 500 --max-train-hours 6
```

Keep all other options identical to the source round-two run. Resume rejects protocol/data/source changes and never overwrites round-one files.

## Validation and final selection

Every training run evaluates the starting EMA and a fixed validation panel of 128 designs every 20 epochs, using DDIM-50 and draw seed 6180. The panel is chosen from the fixed validation split using seed 9137, without using errors. `best-field.pt` minimizes summed field squared error divided by valid element count, matching 4R's validation MSE aggregation. `best-noise.pt` independently retains the lowest fixed-noise validation MSE. Five field checks without improvement trigger early stopping; maximum epoch/time limits also apply. Training time includes periodic validation/checkpointing and inherited elapsed time, but excludes setup/data loading and the separately timed final full-validation pass.

Both checkpoints are confirmed on all 881 validation designs at 20 and 50 DDIM steps using seeds 6180/6181/6182. Scores average individual-draw errors, rather than scoring an ensemble unless explicitly labeled. `validation.json` records both selection routes and all four checkpoint/sampler combinations. A field-selected checkpoint is the best among evaluated states, not necessarily every trained epoch.

Use the completed candidate directories to freeze the lowest full-validation field MSE, with fewer steps breaking ties. Set final reporting draw counts now (default 1 and 3):

```sh
python freeze_selection.py \
  --runs /path/to/R2-B-paired-long /path/to/R2-C-lr-0.0001 /path/to/R2-C-lr-0.0002 /path/to/R2-C-lr-0.0004 \
  --report-sizes 1 3 --output /path/to/round2/selection.json
```

This considers the predefined field/noise checkpoints and sampler settings, rejects smoke or mismatched validation protocols, and records checkpoint hashes. It never accesses test scores. After freezing, supply the selected checkpoint path and step count printed in that manifest:

If optional R2-D is run, include its completed directory in `--runs` before freezing, and retain its comparison with the unweighted LR 0.0002 control regardless of the outcome.

```sh
python draw_study.py --data /path/to/em2d_task.h5 --checkpoint /selected/checkpoint.pt \
  --steps 50 --sizes 1 3 --limit 0 --split test --confirm-test \
  --selection /path/to/round2/selection.json --output /path/to/round2/final-test
python draw_study.py --data /path/to/em2d_task.h5 --checkpoint /selected/checkpoint.pt \
  --steps 50 --sizes 1 3 --limit 0 --split test_extrap --confirm-test \
  --selection /path/to/round2/selection.json --output /path/to/round2/final-extrap
```

Replace `50` with `20` if selected. The commands verify checkpoint, dataset, sampler, full split, and reporting sizes against the frozen manifest. Single-draw rows at K=1 use seed 6180; the three-draw report additionally contains three individual scores and their average, permitting the same single-draw aggregation as round one. Existing test results remain historical comparisons; do not use new test scores to change the configuration.

## Draw variance and timing

R2-A uses the same validation panel and nested draws, with online Welford mean/variance accumulation rather than storing every full field for every design. Each K has:

- Per-design ensemble errors and individual-draw errors using the original metric functions.
- Unbiased variance (denominator K-1) averaged over valid pixels, each field channel, and the Euclidean three-pixel trace-edge band. K=1 variance is null.
- Estimated variance of the ensemble mean, variance/K, conditional on independent draws.
- Normalized design inputs and row/sample IDs for regime/error-versus-variance analysis.
- All draws and final pixel-variance maps for the first four selected examples.

These measure sampling variability from one trained model; they are not calibrated physical uncertainty or variation across training seeds. Spatial masks and nonuniform coordinate derivatives retain the round-one conventions.

Gaussian tensors are keyed by sample ID and draw seed and are invariant to minibatch grouping or row ordering. This changes the exact tensors from round one's batch-dependent RNG stream, so R2-A is a reproducible re-evaluation, not an exact replay of published seeds. Fixed Gaussian starts remain part of DDIM eta=0; there is no zero-noise or regression warm-start ablation in this batch.

Batch-one sampler timing uses two warmups, five repetitions and CUDA synchronization, with initial Gaussian input cached on the device. It excludes noise creation, host transfers, field scoring and I/O. K-draw cost is labeled an estimate of serial sampler cost, K times one draw. The complete study also records actual evaluation wall time including noise construction and metric computation. Do not compare these timing definitions unqualified with old reports or different GPUs.

Build separate machine-readable training and ensemble validation tables:

```sh
python report.py --root /path/to/round2 --output /path/to/round2/report
```

The tables exclude smoke and held-out sampling results and identify loss weighting/gamma for each training trial. Selected candidates should later be replicated across independent model-training seeds before making stronger category-level claims. Unchanged PCA-GP and 4R provide accuracy reference points; match hardware when comparing inference costs. Larger data, higher resolution, alternative initial states and capacity changes are later decisions.

## Local checks

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s . -p test_round2.py -v
```

Tests create a small synthetic HDF5 fixture in a temporary directory, perform CPU training, native resume, sampling and variance reporting, verify masking and target access, test the DDIM oracle, and reject smoke selection/unguarded test access. These outputs are not scientific experiment results. Full GPU training has to run on the course cluster; no jobs are submitted by local tests or dry runs.

## Files

`support.py` loads authorized targets into CPU memory and provides compatible models/sampling/metrics and masked loss weighting. `draw_study.py` implements R2-A and frozen held-out reporting. `train.py` implements R2-B/C and optional R2-D. `run.py`, `campaign.json` and `run.sbatch` launch declared tasks. `freeze_selection.py` locks validation choices. `report.py` creates CSV comparisons. `test_round2.py` provides focused correctness and integration checks. Keep large recovery files and generated outputs outside Git; publish final artifacts under `results/round2/` with provenance and checksums.
