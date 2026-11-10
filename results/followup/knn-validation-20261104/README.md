# Validation-only kNN controls

Distance-weighted field kNN is a cheap check of the proposed similarity representation. All 7,048 training designs supply neighbors; all 881 validation designs supply selection scores. Only training and validation field targets are read. No held-out kNN result is reported here.

| Inputs | Neighbors | Global masked validation MSE | Mean validation relative L2 |
|---|---:|---:|---:|
| Original four inputs | 1 | 0.0001168074 | 8.10% |
| Original four inputs | 5 | 0.0000666680 | 6.44% |
| Original four inputs | 15 | 0.0000649463 | 6.48% |
| Proposed quasi-static coordinates | 1 | 0.0000712303 | 5.82% |
| Proposed quasi-static coordinates | 5 | 0.0000477011 | 4.78% |
| Proposed quasi-static coordinates | **15** | **0.0000475620** | **4.92%** |

The predeclared selection objective is global valid-element MSE, so the selected candidate uses the proposed three coordinates and 15 neighbors. Five neighbors have a lower mean relative L2, a different objective; that does not override the selection rule. Compared with the matched 15-neighbor control, the proposed coordinates lower validation global MSE by about 26.8%.

The existing staff PCA-32 + GP has mean validation relative L2 3.90%, so this cheap control does not establish a new accuracy leader. Differences in model capacity, preprocessing and validation aggregation remain explicit. These scores also do not prove electrostatic scale invariance or physical fidelity.

Coordinate ranges are fitted on training metadata only. The raster fields use the original target convention, and the query mask zeros excluded pixels before scoring. Zero-distance neighbors share all weight, preventing division by zero. All six candidates and all per-design validation scores are retained in the adjacent files. Timing fields include metric calculation and are not inference benchmarks.

Reproduce with [knn_control.py](../../../experiments/followup/knn_control.py). Freeze `validation-selection.json` before evaluating this candidate on held-out targets. The original PCA-GP and 4R remain the main regression references.
