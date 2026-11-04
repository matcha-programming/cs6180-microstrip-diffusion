# Experiment 3: Fourier-feature DeepONet

Completed on Explorer in job `11603443`, using a Tesla V100-SXM2-32GB. Experiments 4–6 are still running in the same sequential job; this directory currently contains the completed Fourier stage only.

| Model | Test relative L2 | Extrapolation relative L2 |
|---|---:|---:|
| Staff PCA-32 + GP | 3.67% | 2.51% |
| Plain DeepONet | 17.75% | 18.39% |
| Fourier DeepONet | 4.93% | 3.81% |

Values are mean per-design errors over valid pixels. The fixed test and extrapolation splits contain 881 and 979 designs. Fourier features improve substantially over the plain trunk, but PCA + GP remains more accurate on both splits.

Validation masked MSE selected frequency scale 2 from 0.5, 1, and 2. The selected checkpoint is from epoch 91. All three searches together took 506.55 seconds, including validation. No test scores were used for frequency selection. The selected Fourier model is the frozen comparator for both diffusion experiments.

`fourier-metrics.json` contains aggregate errors, timing, source hashes, and selection records. `fourier-11603443.zip` contains the full `fourier/` output tree: all candidate configurations and learning histories, candidate checkpoints, the selected and frozen checkpoints, per-design test metrics, fixed example predictions, and the training log.

Archive SHA-256: `8a67ad02e2aa9effb50b0318c1260aed16e08cf9a5b7905c1f70b6a122ea7f30`.

All 1,860 test and extrapolation design IDs were checked against Experiment 1. Aggregate metrics were checked against the saved per-design records. Metric definitions match the earlier DeepONet run: valid-pixel norms, Euclidean three-pixel edge band, and forward coordinate gradients using valid endpoints.

[Source and reproduction instructions](../../../experiments/remaining/).
