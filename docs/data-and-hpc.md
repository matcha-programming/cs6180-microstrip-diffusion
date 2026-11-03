# Dataset and HPC notes

Dataset specifications and compute estimates follow the [course brief](../reference/project-proposal.pdf). Confirm the file contents and cluster environment before the first run.

## Cluster resources

- Dataset: `/courses/CS6180.202710/shared/final_proj/em2d_task.h5` (approximately 0.63 GB)
- Staff baseline: `/courses/CS6180.202710/shared/final_proj/pca_gp.py`
- GPU partition: `courses-gpu`
- Baseline dependencies: numpy, scipy, h5py, scikit-learn
- Staff estimate: baseline about 25 minutes on 8 CPUs; each model a few GPU-hours. Actual runtimes will be recorded with the experiments.

## HDF5 schema

N = 9,789 simulated designs.

| Key | Shape/count | Meaning |
| --- | --- | --- |
| `x` | (N, 4) | h, ln(w/h), t, epsilon_r, each scaled to [-1, 1] |
| `u` | (N, 2, 64, 128) | Peak-normalized E_y and E_z; rows are z, columns are y |
| `mask` | (N, 64, 128) | True on/near copper; exclude these pixels |
| `y_over_h` | (N, 128) | Horizontal coordinates in units of h |
| `z_over_h` | (N, 64) | Height coordinates in units of h |
| `split/train` | 7,048 | Fixed training indices |
| `split/val` | 881 | Fixed validation indices |
| `split/test` | 881 | Fixed in-distribution test indices |
| `split/test_extrap` | 979 | Thickest 10% of substrates, h >= 0.722 mm |
| `analysis/params_raw` | (N, 4) | Unscaled h (mm), w/h, t (mm), epsilon_r; plots only |

Each design's target is divided by its own peak. Predict field shape, not the original peak or absolute field magnitude. The peak varies about 25-fold. Masked target values are zero by construction and must not improve scores artificially.

The normalized z grid is shared across designs. The y grid depends on w/h, spanning +/-(w/h/2 + 6). The mask lies in rows 31-33, with design-dependent extent. Coordinates can be nonuniform; do not assume equal physical spacing from pixel spacing.

## Setup checks

On Discovery:

```sh
ls -lah /courses/CS6180.202710/shared/final_proj
```

Inspect the baseline's usage and dependency versions before running it. Verify HDF5 keys, shapes, split counts, disjoint splits, mask convention, and finite values. Follow any staff environment instructions in the shared directory. Submit jobs through the cluster scheduler using the appropriate course allocation. Keep the dataset and generated checkpoints outside Git.
