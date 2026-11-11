# Feedback experiments: Palace data and controlled training screen

This campaign addresses the [November 7 meeting feedback](../../docs/Meeting%20Minutes). It starts from main `0f634de359bb1283dc90131e355b07807917d914` and imports `palace_regen/` unchanged from `palace-rebuild` commit `f8d7df46d8201ef995b4d2b8efb985cb10d0af3e`.

The five Round 3 learning-rate runs are complete. They are not rerun. The existing best cosine checkpoint is reused for the validation audit and sampling-step study.

## Experiments

| Feedback | Experiment | Control |
|---|---|---|
| More data before higher resolution | 256 new Palace simulations at 64×128; 24 training reference designs; 8 refined meshes | Separate synthetic dataset and a predeclared compatibility gate |
| Smaller network | Base width 24 | Base width 36 |
| L1 loss | Masked L1 noise prediction; separately, masked L1 field regression | Corresponding L2 objectives |
| Batch size | 32 | 16; same 100 passes through the training data |
| Suspiciously smooth validation | Target-independence, batch/order invariance, independent aggregation, zero and shuffled prediction controls | Existing cosine checkpoint, unchanged validation split |
| Flow matching | Straight Gaussian-to-field path, velocity MSE, Euler sampling | Direct DDPM with the same backbone |
| More sampling steps | 20, 50, 100 network evaluations | Same frozen checkpoint and per-design random draws |
| Learning-rate decay | Warmup + cosine for this screen | Completed Round 3 schedule comparison supplies prior evidence |

`campaign.json` freezes seven training variants. All start fresh at seed 6180, use AdamW at peak LR 1e-4, five warmup epochs, cosine decay to 1e-6, FP32, no weight decay, gradient clipping at 1, training-only channel statistics, and the original train/validation split. EMA decay is adjusted by batch examples (`0.999**(batch_examples/16)`) to keep its smoothing comparable across batch sizes.

The first screen uses 100 epochs, not the 500-epoch Round 3 budget. The batch-32 variant has fewer optimizer updates at equal example exposure. Compare results within this screen; do not describe a short-screen score as a full-budget improvement or convergence. Batch size is a hypothesis, not a guaranteed overfitting remedy. Training losses across epsilon, velocity and field objectives are not directly comparable.

Checkpoint selection uses global masked field MSE on the same fixed 128-design validation panel every 20 epochs at 20 sampling steps (one forward pass for regression). The selected checkpoint is evaluated on all 881 validation designs. Diffusion and flow use sampling seeds 6180–6182; scores average single-draw errors, not an ensemble of predictions. The prespecified primary comparison is validation global masked MSE at 20 network evaluations; relative L2, edge/gradient errors and 50/100-step scores are secondary. A later multi-seed/full-budget confirmation is required before a final held-out comparison. This campaign does not load held-out targets.

Flow matching uses `x_t=(1-t)z+t*y` with target velocity `y-z`, where `y` is the standardized field and `z` is Gaussian noise. The sampler integrates from t=0 to 1 with exactly N Euler evaluations. Both generative methods restore zero in masked pixels and return fields bounded to [-1,1]. Diffusion bounds each reconstructed clean estimate; flow bounds its final field. This is a baseline implementation, not a claim that this integrator is optimal. Reference: [Lipman et al., Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).

## Validation audit

`audit.py` tests an actual saved Round 3 cosine checkpoint. It changes target tensors while keeping designs fixed, changes batch sizes and row order, checks zero and shuffled-design predictions, and recomputes global masked MSE in NumPy float64 against the previous evaluation helper. It then saves per-design results for a full 20/50/100-step sweep. Smooth noise loss alone is not evidence of a bug: the old diagnostic uses fixed corruption and EMA weights. Its RNG stream depends on batching, so that noise diagnostic must not be compared across batch-size variants. Field sampling uses sample-ID-keyed noise.

`test_feedback.py` checks masking, objective values/gradients, straight-flow endpoints, the Euler solution for constant velocity, target independence and deterministic noise. GPU smoke jobs exercise every training variant. `capacity_check.py` tests the actual largest batch (32), since smoke tests use batch size 2.

## Palace pilot

The official Palace v0.19.0 source is built on CPU nodes with MPI, GSLIB and BoomerAMG. The source SHA and environment are recorded in scratch. See the [official build instructions](https://awslabs.github.io/palace/stable/install/). No administrative installation or home-directory cache is used.

The reconstruction is inferred from the course data; staff generation sources were not present in `/courses/CS6180.202710/shared/final_proj` when checked. That directory contains `em2d_task.h5`, `pca_gp.py`, `project.html`, and `project.pdf`. Do not call this the original staff generator.

New inputs use a seeded Latin hypercube over h, log(w/h), t and epsilon_r. The h range stays below the course extrapolation threshold; other bounds follow the course attributes. Exact existing parameter tuples are rejected. The normalized z grid is copied from training metadata. The geometric pixel-mask rule must match every one of the 7,048 training masks before new cases are prepared. The main mesh spacing is t; eight reference meshes use 0.75t. All use order-1 electrostatics, the branch's 1V trace and grounded substrate boundary, potential interpolation, chord differences, and valid-pixel peak normalization. E and the stored scale s are converted from gradients in mm to V/m.

The gate uses raw valid-pixel errors with no target-fitted scale and no removal of spike pixels:

- Mean reference relative L2 ≤ 5%.
- Reference 95th-percentile relative L2 ≤ 10%.
- Maximum coarse/refined relative L2 ≤ 2%.

These are conservative experiment-design thresholds, not established physical tolerances. All references come from the training split, stratified by t/h. A failed gate still yields a labeled, separate synthetic dataset, but it blocks course-data augmentation. Passing permits a separate original-only versus augmented training comparison; it does not prove identical staff provenance. Keep raw meshes, configs, logs and fields in scratch; publish manifests and summaries in GitHub. Mesh refinement estimates discretization sensitivity, not full physical accuracy.

## Running on HPC

Campaign root: `/scratch/he.zit/cs6180-microstrip-experiments/feedback-20261111`.

1. Build Palace (`build_palace.sbatch`) and prepare/audit parameter metadata (`palace_prepare.sbatch`) on CPUs.
2. Run `preflight.sbatch`, `capacity.sbatch`, and `audit.sbatch` on V100 GPUs; inspect results before training submission.
3. Submit `gpu.sbatch` as array `0-6%2` only after preflight and capacity pass. Each task has a six-hour allocation and a four-hour training guard.
4. After the Palace build and preparation pass, run one generation shard first. If its complete samples pass structural checks, submit the remaining shards with at most four in parallel. Pack only after every case has a verified completion record.
5. Review the compatibility gate before any augmentation. Review the training screen before expensive confirmation or held-out scoring.

Jobs have distinct output directories. Never resubmit an uncertain submission without checking Slurm and the recorded job IDs. Completed cases are checksum-verified during packing; failed cases retain their diagnostic logs. Training saves model, EMA, optimizer, scheduler and RNG state at epoch boundaries. Automatic resume is not implemented for this new trainer; inspect a stopped run before adding recovery. A timeout is reported as incomplete.
