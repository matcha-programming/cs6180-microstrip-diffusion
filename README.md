# Microstrip Field Diffusion

CS 6180 final project comparing DeepONet regression with conditional diffusion for microstrip electric-field prediction.

Given four design parameters, each model predicts a two-channel, peak-normalized field on a 64 × 128 grid. We will compare overall accuracy, trace-edge detail, and computational cost on the fixed course dataset of 9,789 simulated designs.

## Approach

- Establish the staff PCA + Gaussian-process baseline.
- Train DeepONet with plain and Fourier-feature coordinate encodings.
- Train conditional DDPM with DDIM sampling.
- Evaluate residual diffusion as an extension, subject to the compute budget.

## Documentation

- [Course project brief](reference/project-proposal.pdf) · [Text version](reference/project-proposal.txt)
- [Experiment plan](docs/project-plan.md)
- [Dataset and cluster setup](docs/data-and-hpc.md)
- [References and reading guide](reference/README.md)
- [Project context](docs/source-notes.md)

## Results

Both baselines use the fixed course splits. PCA and GP fitting use training data only. The reported errors are mean relative L2 over unmasked pixels.

| Experiment | Model | Test error | Extrapolation error | Results |
| --- | --- | ---: | ---: | --- |
| 1 | Staff PCA-32 + Gaussian process | 3.67% | 2.51% | [Run 11601747](results/pca_gp/11601747/) |
| 2 | Plain DeepONet | 17.75% | 18.39% | [Run 11597463](results/deeponet_plain/11597463/) |

The staff baseline has lower field, trace-edge, and gradient errors than this plain DeepONet run. The [comparison report](results/pca_gp/11601747/) includes matched metric definitions, timing details, fixed example predictions, and edge zooms. CPU and GPU inference timings use different hardware.

## Reproduction

- [Experiment 1: PCA-32 + GP](experiments/pca_gp/)
- [Experiment 2: plain DeepONet](experiments/deeponet_plain/)
- [Experiments 3–6: Fourier features, DDPM/DDIM, and residual diffusion](experiments/remaining/)

Experiments 3–6 are running on Explorer as job `11603443` on a Tesla V100 (32 GB). The pipeline selects Fourier frequency scale and DDIM sampling steps on validation data, then evaluates the fixed test and extrapolation splits. Each diffusion model has a two-hour training budget; sampling and evaluation run afterward. Results will be added after verification.
