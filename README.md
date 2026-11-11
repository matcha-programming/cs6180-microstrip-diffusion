# Microstrip Field Diffusion

CS 6180 final project comparing regression with conditional diffusion for microstrip electric-field prediction.

Given four design parameters, each model predicts a two-channel, peak-normalized field on a 64 × 128 grid. The experiments use the fixed course dataset of 9,789 designs and measure field accuracy, trace-edge detail, spatial gradients, and compute cost.

## Results

All five schedules in the [Round 3 learning-rate screen](results/round3/11802771/) completed 220,500 updates. Warmup + cosine had the lowest screening validation MSE (5.747883e-5, with 4.874% relative L2) and was selected for comparison with constant in the frozen confirmation plan. The table below reports mean per-design relative L2 over valid pixels on the held-out splits. Diffusion entries average single-draw errors for seeds 6180–6182. Three-draw ensembles and their extra cost are reported separately.

| Experiment | Model | Test error | Extrapolation error | Results |
|---|---|---:|---:|---|
| 1 | Staff PCA-32 + Gaussian process | 3.67% | 2.51% | [11601747](results/pca_gp/11601747/) |
| 2, original | Plain DeepONet (100-epoch run) | 17.75% | 18.39% | [11597463](results/deeponet_plain/11597463/) |
| 2, continuation | Plain DeepONet to validation plateau | 5.76% | 4.75% | [11638570](results/deeponet_plateau/11638570/) |
| 3 | Fourier DeepONet | 4.93% | 3.81% | [11603443](results/remaining/11603443/) |
| 4–5 | Direct DDPM, DDIM-20 | 6.32% | 4.48% | [11603443](results/remaining/11603443/) |
| 4R | Deterministic U-Net | **2.87%** | **1.15%** | [11604740](results/unet_regression/11604740/) |
| 4, matched to 4R | Position-conditioned DDPM, DDIM-50 | 7.56% | 5.81% | [11604740](results/unet_regression/11604740/) |
| 6 | Residual DDPM, DDIM-50 | 7.99% | 6.50% | [11603443](results/remaining/11603443/) |

Among the verified results in this table, the deterministic U-Net has the lowest overall field and edge errors. The plain DeepONet continuation reduced its original test error from 17.75% to 5.76%, showing that the original 100-epoch cutoff understated its performance. Against its matched diffusion control on the same P100, it needs about 10 ms per prediction versus 586 ms per diffusion draw. Averaging three diffusion draws improves accuracy but adds cost and does not close the field-error gap. Residual diffusion's three-draw average slightly improves extrapolation gradient error; the complete reports retain this tradeoff.

These are single-training-seed, budgeted results. The paired diffusion model reached its two-hour training limit, while 4R stopped early. The comparison does not establish that longer or differently tuned diffusion training cannot improve performance. PCA/GP timing used a CPU; the original diffusion runs used a V100. Only the 4R pair provides a comparison with the same backbone, position inputs, initial weights, and GPU.

## Reports and artifacts

- [Feedback experiments: seven training variants and 256 Palace designs (augmentation gate failed)](results/feedback/20261111/)

- [Round 3: completed schedule comparison and frozen confirmation plan](results/round3/11802771/)
- [Original Fourier, direct diffusion, DDIM, and residual experiments](results/remaining/11603443/)
- [4R and its matched diffusion control](results/unet_regression/11604740/)
- [PCA/GP and plain DeepONet comparison](results/pca_gp/11601747/)

Reports include per-design metrics, configurations, source hashes, learning curves, fixed field/error examples, edge zooms, sampler settings, sample variation, and selected checkpoints. Test and extrapolation results remain separate. Validation selects models and sampling steps; PCA, GP, and normalization fitting use training data only.

## Reproduction

- [Feedback experiment protocol and HPC jobs](experiments/feedback_round4/)

- [Experiment 1: PCA-32 + GP](experiments/pca_gp/)
- [Experiment 2: plain DeepONet](experiments/deeponet_plain/) · [Continuation to a plateau](experiments/deeponet_plateau/)
- [Experiments 3–6: Fourier features and diffusion](experiments/remaining/)
- [Experiment 4R and matched DDPM](experiments/unet_regression/)

## Documentation

- [Course project brief](reference/project-proposal.pdf) · [Text version](reference/project-proposal.txt)
- [Experiment plan](docs/project-plan.md)
- [Dataset and cluster setup](docs/data-and-hpc.md)
- [References and reading guide](reference/README.md)
- [Project context](docs/source-notes.md)
