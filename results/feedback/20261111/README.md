# Feedback campaign: validation audit and new experiments

The validation audit and sampler comparison are complete. Seven fresh training variants are running or queued under Slurm array **11811956**; no training-screen results are claimed here yet. Report job **11811957** follows the array. The [experiment protocol](../../../experiments/feedback_round4/) maps each meeting suggestion to a controlled test.

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

Palace metadata preparation **11811896** matched the proposed mask rule on all 7,048 training designs, with zero mismatches. It prepared 256 new Latin-hypercube parameter combinations, 24 training-only reference simulations, and eight refined reference simulations. The [case manifest](palace/cases.json) and [mask audit](palace/mask-audit.json) are available; these records describe prepared inputs, not completed simulation outputs. Palace build **11811825** stopped at an MFEM example link error; recovery **11811978** enables explicit LAPACK linkage and reuses the compiled dependencies. Simulation completion and the compatibility gate will be recorded separately.

All heavy output stays under `/scratch/he.zit/cs6180-microstrip-experiments/feedback-20261111`. Synthetic data will not be mixed into course training unless its prespecified raw-error and mesh-refinement checks pass. The course train/validation/test splits remain intact.

The [submission record](screen-submission.json) records the GPU array and its preconditions. Thirteen downloaded evidence files matched SHA256 values calculated on HPC; see [verification](verification.json) and the original [hash mapping](evidence-sha256.json). The [source manifest](microstrip-feedback-manifest.json) identifies the submitted experiment source snapshot. Build-recovery scripts added after the initial submission are tracked separately in the experiment folder.
