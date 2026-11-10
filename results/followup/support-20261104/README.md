# Fixed-split support audit

The audit uses design metadata only, with no field-target reads or split changes. The dataset SHA-256 is `7670297247077ee223a572b724e45236a6970a1f38da02f9a96e074ca0215af2`.

| Split / coordinates | Within training coordinate bounds | Inside training convex hull |
|---|---:|---:|
| Test / original four inputs | 99.89% | 93.98% |
| `test_extrap` / original four inputs | 0% | 0% |
| Test / proposed quasi-static coordinates | 99.89% | 98.18% |
| `test_extrap` / proposed quasi-static coordinates | 96.53% | 92.54% |

All 979 `test_extrap` designs have h above the training maximum: training h reaches 0.722147 mm and the held-out h begins at 0.722227 mm. It is a valid thickness holdout in the provided input representation.

The proposed alternative is `(log(w/h), log(t/h), epsilon_r)`. It follows a scale-similarity hypothesis for peak-normalized electrostatics with geometrically scaled boundaries. Under that hypothesis, 906/979 thickness-held-out designs are inside the training convex hull in those three coordinates. This provides a more precise regime description; it does not verify the staff solver, prove interpolation accuracy, or justify calling every held-out design non-extrapolative. Coordinate transformations also change hull geometry.

`support.json` includes training-only coordinate ranges and leave-one-out nearest-neighbor reference distances. `per-design.csv` gives every validation/test/extrapolation row, sample ID, nearest training row, distances and membership indicators in both spaces. Coordinate scaling, neighbor trees and convex hulls use training metadata only. Training rows have self-distance zero in the split summary; the separate leave-one-out p95 excludes that self match.

Reproduce with [audit_support.py](../../../experiments/followup/audit_support.py). Four focused unit checks cover similarity coordinates, the distinction between marginal bounds and joint support, invalid geometry and exact-neighbor weighting. Preserve the official split labels in downstream reporting.
