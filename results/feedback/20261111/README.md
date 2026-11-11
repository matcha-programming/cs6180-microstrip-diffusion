# Feedback campaign: validation audit and new experiments

All seven 100-epoch training runs, the validation audit, and the sampler comparison are complete. Regression leads this validation screen; flow matching outperforms the diffusion control at the same 20 network evaluations. Palace inputs are prepared, but simulation output is still pending a solver build repair. The [experiment protocol](../../../experiments/feedback_round4/) maps each meeting suggestion to a controlled test.

## Training screen

Slurm array **11811956** completed all seven runs; report job **11811957** completed afterward. Each run used training seed 6180 and 100 passes through the original training split. Checkpoint selection used global masked field MSE on a fixed 128-design validation panel. The selected checkpoints were then scored on all 881 validation designs.

| Variant | Global masked MSE (primary) | Relative L2 | Network evaluations | Selected epoch |
|---|---:|---:|---:|---:|
| regression-l2 | 2.683217e-05 | 2.784% | 1 | 100 |
| regression-l1 | 2.818730e-05 | 2.728% | 1 | 100 |
| flow-matching | 4.728336e-05 | 4.611% | 20 | 80 |
| diffusion-l1 | 6.415357e-05 | 5.882% | 20 | 100 |
| diffusion-control | 6.913678e-05 | 6.063% | 20 | 100 |
| diffusion-batch32 | 8.195836e-05 | 6.823% | 20 | 100 |
| diffusion-small | 8.492290e-05 | 7.064% | 20 | 100 |

Rows are ordered by the prespecified global MSE metric. Regression-L2 leads on that metric, while regression-L1 has the lowest mean relative L2. Flow and diffusion scores average three single-draw errors (sampling seeds 6180–6182); they do not average predicted fields. Regression uses one forward pass.

![Validation comparison](validation-comparison.png)

The smaller diffusion model and batch-32 setting did not improve this screen. Batch 32 receives 22,100 optimizer updates versus 44,100 for batch 16, so this tests equal example exposure rather than equal update count. Diffusion-L1 improves field MSE modestly, but its gradient relative L2 rises to 0.630 from the control's 0.479. It is not an improvement across all metrics.

![Validation panel history](validation-panel-history.png)

These are single-training-seed, 100-epoch results. They do not establish convergence, performance at the 500-epoch Round 3 budget, or held-out generalization. Six variants selected their final epoch. A full-budget, multi-seed comparison of the diffusion control, flow matching, and regression is the next confirmation candidate; it has not been submitted. New synthetic data must first pass the separate compatibility gate before an augmentation experiment.

See the [summary](screen-report/summary.json), [20/50/100-step results](screen-report/validation.csv), [run histories and per-design errors](screen/), and [figure source](build_figures.py). Forty-seven downloaded evidence files matched their HPC SHA256 hashes. Checkpoint hashes were checked on HPC, and the per-design averages were recomputed locally; predictions were not recomputed locally. See [transfer verification](screen-transfer-verification.json), [checkpoint and completion checks](screen-verification.json), and [file hashes](screen-evidence-sha256.json).

## Validation audit

Job **11811898** evaluated the completed Round 3 cosine checkpoint (`20cf28fcba91cec7626793139e2342c92326523ec491d992393af53cf4ad2430`). All 881 validation designs were scored at each sampling budget using seeds 6180–6182. These are averages of single-draw errors, not an ensemble of fields.

| DDIM network evaluations | Relative L2 | Global masked MSE | Evaluation wall time |
|---:|---:|---:|---:|
| 20 | 4.874% | 5.747883e-5 | 51.6 s |
| 50 | 5.036% | 6.145003e-5 | 118.3 s |
| 100 | 5.094% | 6.282015e-5 | 229.5 s |

The 20- and 50-step scores reproduce the previous evaluation. Increasing to 100 steps did not improve this checkpoint. Timings include evaluation and metric work on the allocated GPU; they are not a dedicated single-design latency benchmark.

On a 16-design validation panel:

- Replacing target tensors left sampled predictions unchanged.
- Changing inference batch size from 16 to 4 changed the largest output element by only 4.77e-7.
- Reversing row order preserved predictions within the stated numerical tolerance.
- Independent NumPy global masked MSE was 5.7439651003e-5, versus 5.7439651622e-5 from the prior helper.
- A zero predictor had relative L2 of 1.0. Shuffling predictions across designs increased mean relative L2 from 4.51% to 40.40%.

These checks found no target dependence or metric-aggregation error in the audited path. They do not prove every possible validation issue is absent. The prior noise-loss diagnostic uses fixed randomness and EMA weights, which can make its curve smooth; its RNG stream also depends on batching, so it is unsuitable for comparing noise loss across batch-size settings. Field sampling uses sample-ID-keyed randomness.

See [checks](validation-audit/checks.json), [metrics](validation-audit/validation.json), and per-design files in [validation-audit](validation-audit/).

## Preflight and data preparation

GPU preflight **11811854** passed six numerical tests and all seven end-to-end smoke runs. These smoke scores are not performance results. Capacity job **11811929** completed a base-36, batch-32 forward/backward/optimizer step with peak allocated GPU memory of 2,762,168,320 bytes.

Palace metadata preparation **11811896** matched the proposed mask rule on all 7,048 training designs, with zero mismatches. It prepared 256 new Latin-hypercube parameter combinations, 24 training-only reference simulations, and eight refined reference simulations. The [case manifest](palace/cases.json) and [mask audit](palace/mask-audit.json) are available; these records describe prepared inputs, not completed simulation outputs. The initial build (**11811825**) failed at MFEM's LAPACK linkage. Recovery **11811978** fixed that dependency but exposed Palace's required eigensolver. Build **11818445** added ARPACK and then stopped because ParMETIS was absent. Recovery **11818544** installed the ParMETIS version pinned by Palace v0.19, but compilation then failed because GCC 11.5 could not compile its C++20 `constexpr std::string` functions. Job **11818592** installs a private GCC 13 compiler in scratch and builds the Palace application in a separate directory, reusing the completed libraries. Build and runtime checks are pending. The earlier pilot job **11812063** was cancelled without running; it produced no simulated fields. Simulation completion and the compatibility gate will be recorded separately.

All heavy output stays under `/scratch/he.zit/cs6180-microstrip-experiments/feedback-20261111`. Synthetic data will not be mixed into course training unless its prespecified raw-error and mesh-refinement checks pass. The course train/validation/test splits remain intact.

The [submission record](screen-submission.json) records the GPU array and its preconditions. Thirteen downloaded evidence files matched SHA256 values calculated on HPC; see [verification](verification.json) and the original [hash mapping](evidence-sha256.json). The [source manifest](microstrip-feedback-manifest.json) identifies the submitted experiment source snapshot. Build-recovery scripts added after the initial submission are tracked separately in the experiment folder.
