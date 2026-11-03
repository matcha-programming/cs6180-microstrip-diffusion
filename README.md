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

## Status

The first plain DeepONet run completed on Explorer with a Tesla P100 GPU. Mean relative L2 error was 17.75% on the test split and 18.39% on the extrapolation split.

- [Results, checkpoint, and predictions](results/deeponet_plain/11597463/)
- [Training code and submission instructions](experiments/deeponet_plain/)

The staff PCA + GP baseline, Fourier-feature DeepONet, and diffusion experiments remain to be run.
