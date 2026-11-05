# Project scope and experiment plan

## Question

Can conditional diffusion recover the sharp electric-field lobes near a copper trace more accurately than regression, at an acceptable compute cost? Each design has one target field; sample diversity is not a goal in itself. The comparison will measure both accuracy and cost.

## Experiments

1. Inspect the fixed splits, masks, coordinate grids, and example targets. Run the staff PCA (32 components) + Gaussian-process baseline; fit PCA and GP using training data only.
2. Train DeepONet: branch MLP maps four inputs to two sets of p coefficients (start with p=64); trunk MLP maps each pixel's coordinates to two sets of p basis values. Each channel is their dot product. Use masked MSE.
3. Compare the plain trunk with a Fourier-feature trunk: fixed random B of shape (128, 2), using sin/cos of 2*pi*B*coordinate. Tune frequency scale on validation data.
4. Train direct conditional DDPM with a small U-Net (proposal suggests 5-20M parameters), design embedding added to the time embedding, approximately 1,000 noise steps, and masked noise-prediction MSE. Only the two field channels are noised. Pair this with [Experiment 4R: deterministic U-Net regression](../experiments/unet_regression/README.md), using the same backbone and conditioning to test the benefit of diffusion beyond convolutional regression. The paired design adds identical fixed row/column position planes to both models; record this addition to the original architecture sketch.
5. Evaluate DDIM sampling at 20-50 steps. Reset masked field pixels to zero after every step.
6. If time allows, train residual diffusion on u minus a frozen DeepONet prediction. Supply both prediction channels as conditioning. Optional ablations add coordinates or a mask/distance channel. Give comparable geometry inputs to regression controls and report any input differences.

The initial architectures follow the course brief. Any generative-model variant will use the same regression comparator.

## Round 1 findings and remaining questions

The first campaign is complete. The fixed test / extrapolation mean relative L2 errors are:

| Model | Test | Extrapolation | Important qualification |
| --- | ---: | ---: | --- |
| PCA-32 + GP | 3.67% | 2.51% | CPU baseline; training-only fitting |
| Plain DeepONet | 17.75% | 18.39% | One training seed |
| Fourier DeepONet | 4.93% | 3.81% | Historical comparator; no new Fourier work in round 2 |
| Original direct DDIM-20 | 6.32% | 4.48% | V100, AMP, 200 epochs, no fixed position planes |
| Deterministic U-Net 4R | 2.87% | 1.15% | P100, FP32, selected epoch 37; stopped after epoch 67 |
| Matched position-conditioned DDIM-50 | 7.56% | 5.81% | P100, FP32, selected epoch 158; two-hour cap during epoch 159 |
| Residual DDIM-50 | 7.99% | 6.50% | Frozen Fourier comparator; V100, AMP, 200 epochs |

Diffusion figures above average individual-draw errors across three sampling seeds; those are not three independent model-training seeds. Three-draw means improve accuracy but do not close the overall-error gap with 4R. All final field metrics use the same implementation. See [original diffusion results](../results/remaining/11603443/README.md) and the [matched pair](../results/unet_regression/11604740/README.md).

Investigation of the executed code identified four issues for the next campaign:

- Diffusion selected EMA checkpoints by validation noise MSE. Generated validation field accuracy was used only to select 20 versus 50 DDIM steps after training.
- The matched model stopped while noise validation was still improving. Its comparison to the original direct run also changes precision, hardware, input planes, initialization, and completed updates.
- Draw variability was summarized across each split; full draws were retained only for four examples. There is no saved unbiased pixel-variance summary for every design.
- The old direct trainer loads a frozen regression checkpoint even when it is unused. Round 2 removes that dependency and transfers only minibatches to the GPU.

## Round 2: diffusion optimization and sampling

Approved scope: the recommended next batch, with Fourier experiments removed, plus an optional controlled Min-SNR ablation. Implementation, machine-readable task definitions, launch commands, and checks are in [experiments-round-2](../experiments-round-2/README.md). Round 1 artifacts and source stay unchanged.

| ID | Experiment | Fixed settings and changes | Decision/output |
| --- | --- | --- | --- |
| R2-A | Draw-count and variance study on each existing direct checkpoint | Original DDIM-20 and paired DDIM-50 separately; fixed 128-design validation panel; nested 1/2/4/8/16 draws; no training | Ensemble and individual-draw errors, unbiased per-design/channel/edge variance, fixed draw examples, cost curve |
| R2-B | Longer paired direct DDPM training | Continue the paired optimizer/EMA/RNG checkpoint at LR 0.0002; base 36, batch 16, FP32, original mask and 1,000-step noise schedule; maximum epoch 500 / six total training hours including the previous run | Field-selected versus noise-selected EMA checkpoints; improvement versus the recorded starting state |
| R2-C | Learning-rate screen | Three fresh paired models at LR 0.0001 / 0.0002 / 0.0004, identical initialization seed 6180, base 36, batch 16, FP32; 200 epochs / three hours per candidate | Full-validation comparison at a common declared budget; record actual updates and stopping reasons |
| R2-D (optional) | Min-SNR noise-loss weighting | Fresh paired model, gamma 5, LR 0.0002, same initialization and all other settings/budgets as R2-C-lr-0.0002 | Isolate the effect of weighting; compare generated-field quality and convergence versus updates and GPU-hours |

R2-B requires the full `diffusion/last.pt` remaining on HPC. Published `best-diffusion.pt` contains selected weights and cannot recover optimizer or RNG state. If recovery is unavailable, run a separately labeled fresh long-budget experiment rather than reporting a weights-only restart as continuation. Legacy recovery has no partial-epoch order/cursor: resume at the next epoch, document the unrecoverable tail, and do not invent an exact historical update count. New round-two recovery includes the minibatch order/cursor.

All round-two training retains the actual 5,602,322-parameter paired backbone, EMA 0.999, AdamW with zero weight decay, gradient clipping 1, and the existing training-only channel transform. R2-C isolates learning rate; it does not add an LR scheduler, larger model, new conditioning, or different loss. All training uses the original 7,048 designs.

R2-D is a candidate, not an established improvement for this task. For our epsilon-prediction objective, use `SNR(t) = alpha_bar(t)/(1-alpha_bar(t))` and weight each design's valid-pixel noise squared errors by `min(SNR(t), gamma)/SNR(t)`, with gamma 5. Preserve the original valid-element denominator rather than dividing by the sum of weights. This is the epsilon-specific weighting from the [authors' implementation](https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/main/guided_diffusion/gaussian_diffusion.py), adapted to the copper mask; it is not the x0-prediction weight. Keep timestep sampling, noise schedule, EMA, architecture, sampler and field-selection protocol unchanged. R2-D starts fresh, never by changing the objective of R2-B's resumed run.

Use the existing unweighted LR 0.0002 fresh trial as its control. Log weighted training objective, unweighted training/validation noise MSE and mean timestep weight separately. A lower weighted loss is not evidence of improved prediction. Decide benefit using validation field, edge and gradient errors and cost, then replicate any promising effect with independent training seeds. This adds up to three optional training GPU-hours plus full-validation time. Do not infer that it works across learning rates from this single controlled trial.

At initialization and every 20 epochs, evaluate the EMA field on a deterministic 128-design validation panel, DDIM-50, fixed draw seed 6180. Select `best-field.pt` by globally accumulated valid-element field MSE, matching 4R's aggregation. Keep `best-noise.pt` independently for the checkpoint-selection diagnostic. Stop after five field evaluations without improvement, or the epoch/time limit. Preserve the starting EMA as an eligible reference. Budget counters include training, periodic validation and recovery work, including historical elapsed time for continuation. Full validation confirmation and final test evaluation are timed separately.

After training, evaluate both retained checkpoints on all 881 validation designs using 20 and 50 DDIM steps and three fixed sampling seeds. Freeze the checkpoint and sampler with the lowest mean single-draw validation field MSE, breaking ties toward fewer steps. `freeze_selection.py` records all candidates and checkpoint/data hashes. Fix reporting draw counts before opening the held-out evaluations; the default final report is one draw and a three-draw ensemble, with individual-draw scores retained. The original split and the extrapolation split require an explicit frozen-selection manifest and separate commands. Existing test results are historical evidence, not a tuning objective.

Round-two sampling uses Gaussian noise keyed by sample ID and draw seed, independent of minibatch size and row order. This preserves distribution and reproducibility but changes the exact noise tensors from round one's batch-dependent stream. Re-evaluate both historical checkpoints with the round-two sampler instead of expecting identical numerical replay of published draw scores. Variance uses the unbiased K-1 denominator; K=1 variance is undefined and saved as null. Measure sampling spread separately from error variation across designs or training seeds.

R2-A and the three R2-C trials can be submitted independently. R2-B is independent once the HPC recovery path is verified. Before submitting full jobs, run the synthetic CPU integration checks and verify actual GPU model, precision, memory and runtime. Only compare latency numbers measured on the same GPU and protocol. Preserve actual GPU-hours and update counts; equal epoch limits and equal compute budgets answer different questions.

The deliverable is a validation-based accuracy/cost comparison, per-design variance diagnostics, field-versus-noise checkpoint comparison, and final frozen test/extrapolation reports against the existing PCA+GP and 4R references. Replicate a selected configuration with independent training seeds before strengthening a category-wide conclusion.

### Later decision points

The following discussed ideas are deferred, not part of this launch batch: expanded Palace training sets, higher-resolution targets, zero/structured noise starts, partial denoising of 4R predictions, residual diffusion around 4R, and capacity/conditioning changes. Min-SNR weighting is now covered by optional R2-D. Data generation first requires solver-equivalence and mesh-convergence pilots. No new spatial or DeepONet Fourier experiments are included. Progress to later work only when the round-two validation findings identify a useful direction.

## Evaluation protocol

- Tune on validation only; report test and extrapolation test separately.
- Exclude masked pixels from every loss and metric, including denominators.
- Report relative L2 over valid pixels, relative L2 within three pixels of the mask, spatial-gradient error, and time per prediction.
- Before implementation, agree on edge-band distance convention, gradient norm, finite-difference stencil, and aggregation across designs. Avoid gradient stencils crossing masked pixels; account for the nonuniform coordinate grid when reporting physical-coordinate derivatives. Document these conventions alongside the metric implementation.
- For diffusion, report variation across multiple samples of the same design, with seeds and sample count. Distinguish single-sample accuracy from ensemble-average accuracy and its extra cost.
- Record parameter counts, GPU model, GPU-hours, training settings, inference batch size, sampling steps, and timing method. Synchronize GPU timing and distinguish model inference from data loading.
- Show truth, prediction, and error for several fixed test examples, including trace-edge zooms. Avoid selecting only favorable examples.

## Deliverables

Reproducible baseline and training commands; saved configuration and seeds; comparison table; edge-detail figures; and an evidence-based answer to the accuracy-versus-cost question.
