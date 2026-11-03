# References

Background literature for the regression and diffusion experiments. The seven papers below follow the course reading list.

| Reading | Role in this project | Links |
| --- | --- | --- |
| Ho, Jain, Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020. | Noise-prediction objective and forward/reverse diffusion. | [Record](https://arxiv.org/abs/2006.11239), [PDF](https://arxiv.org/pdf/2006.11239) |
| Song, Meng, Ermon. Denoising Diffusion Implicit Models. ICLR 2021. | Fast sampling; compare step count with error and latency. | [Record](https://arxiv.org/abs/2010.02502), [PDF](https://arxiv.org/pdf/2010.02502) |
| Mardani et al. Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling. arXiv 2023. | Regression plus residual correction; test whether the idea transfers to a unique field target. | [Record](https://arxiv.org/abs/2309.15214), [PDF](https://arxiv.org/pdf/2309.15214) |
| Lippe et al. PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers. NeurIPS 2023. | Motivation for recovering fine-scale components; this project's static fields differ from PDE rollouts. | [Record](https://arxiv.org/abs/2308.05732), [PDF](https://arxiv.org/pdf/2308.05732) |
| Lu et al. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence 2021. | Branch/trunk regression architecture. | [Published version](https://doi.org/10.1038/s42256-021-00302-5), [related preprint](https://arxiv.org/abs/1910.03193), [preprint PDF](https://arxiv.org/pdf/1910.03193) |
| Tancik et al. Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. NeurIPS 2020. | Fourier coordinate encoding for the trunk; measure trace-edge changes. | [Record](https://arxiv.org/abs/2006.10739), [PDF](https://arxiv.org/pdf/2006.10739) |
| Ronneberger, Fischer, Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015. | Encoder-decoder architecture for the denoiser; adapt inputs and output to field/noise prediction. | [Record](https://arxiv.org/abs/1505.04597), [PDF](https://arxiv.org/pdf/1505.04597) |

The DeepONet preprint has a different title/version history from the 2021 journal article. The CorrDiff entry follows the brief's 2023 preprint citation; a later journal version also exists. Cite the version used in the final report.

Suggested order: DeepONet and Fourier Features for the regression work; DDPM, U-Net, then DDIM for direct diffusion; CorrDiff and PDE-Refiner for residual and fine-detail experiments. Record the model choices and ablations informed by each paper in the experiment notes.

The [course brief](project-proposal.pdf) and its [text version](project-proposal.txt) are included here. Paper copies are in `papers/`; `papers/manifest.json` records their source URLs.

## Paper copies

- [ddpm](papers/ddpm.pdf)
- [ddim](papers/ddim.pdf)
- [CorrDiff (PDF)](https://arxiv.org/pdf/2309.15214)
- [pde-refiner](papers/pde-refiner.pdf)
- [deeponet-preprint](papers/deeponet-preprint.pdf)
- [fourier-features](papers/fourier-features.pdf)
- [unet](papers/unet.pdf)
