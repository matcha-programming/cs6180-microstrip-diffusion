# Project scope and experiment plan

## Question

Can conditional diffusion recover the sharp electric-field lobes near a copper trace more accurately than regression, at an acceptable compute cost? Each design has one target field; sample diversity is not a goal in itself. A measured negative result is useful.

## Proposed sequence

1. Inspect the fixed splits, masks, coordinate grids, and example targets. Run the staff PCA (32 components) + Gaussian-process baseline; fit PCA and GP using training data only.
2. Train DeepONet: branch MLP maps four inputs to two sets of p coefficients (start with p=64); trunk MLP maps each pixel's coordinates to two sets of p basis values. Each channel is their dot product. Use masked MSE.
3. Compare the plain trunk with a Fourier-feature trunk: fixed random B of shape (128, 2), using sin/cos of 2*pi*B*coordinate. Tune frequency scale on validation data.
4. Train direct conditional DDPM with a small U-Net (proposal suggests 5-20M parameters), design embedding added to the time embedding, approximately 1,000 noise steps, and masked noise-prediction MSE. Only the two field channels are noised.
5. Evaluate DDIM sampling at 20-50 steps. Reset masked field pixels to zero after every step.
6. If time allows, train residual diffusion on u minus a frozen DeepONet prediction. Supply both prediction channels as conditioning. Optional ablations add coordinates or a mask/distance channel. Give comparable geometry inputs to regression controls and report any input differences.

These are recommendations from the course brief, not mandatory architecture choices. Keep a regression comparator for any generative variant.

## Evaluation protocol

- Tune on validation only; report test and extrapolation test separately.
- Exclude masked pixels from every loss and metric, including denominators.
- Report relative L2 over valid pixels, relative L2 within three pixels of the mask, spatial-gradient error, and time per prediction.
- Before implementation, agree on edge-band distance convention, gradient norm, finite-difference stencil, and aggregation across designs. Avoid gradient stencils crossing masked pixels; account for the nonuniform coordinate grid when reporting physical-coordinate derivatives. Record these as implementation decisions, since the brief does not fully specify them.
- For diffusion, report variation across multiple samples of the same design, with seeds and sample count. Distinguish single-sample accuracy from ensemble-average accuracy and its extra cost.
- Record parameter counts, GPU model, GPU-hours, training settings, inference batch size, sampling steps, and timing method. Synchronize GPU timing and distinguish model inference from data loading.
- Show truth, prediction, and error for several fixed test examples, including trace-edge zooms. Avoid selecting only favorable examples.

## Proposed deliverables

Reproducible baseline and training commands; saved configuration and seeds; comparison table; edge-detail figures; and an evidence-based answer to the accuracy-versus-cost question. Owners and deadlines remain for the team to decide.
