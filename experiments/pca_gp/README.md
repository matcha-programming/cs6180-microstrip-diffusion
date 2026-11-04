# Experiment 1: staff PCA-32 + Gaussian process

Runs the unmodified course baseline with 32 PCA components, 32 independent ARD RBF Gaussian processes, a 1,500-design training subset for kernel optimization, and seed 0. PCA, coefficient standardization, GP kernel optimization, final GP fits, and the five-neighbor sanity check all use training data only.

`staff_pca_gp.py` is the course file copied from `/courses/CS6180.202710/shared/final_proj/pca_gp.py`, SHA-256 `4cf231c86d4270a0a3a6662655e094eabef5d5a409982b183920026a83ae6014`. The wrapper checks this hash before execution. It audits shapes, split coverage and disjointness, masks, finite values, normalization, and coordinate ordering.

## Run on Explorer

Submit from this directory:

```sh
MICROSTRIP_PYTHON=/path/to/python sbatch run.sbatch
```

The script requests eight CPUs and 32 GB of memory on `courses`, under `cs6180.202710`. It uses no GPU. Dependencies are listed in `requirements.txt`.

## Outputs and comparison

Outputs are saved in `runs/pca_gp_JOBID/`.

- `staff-results.json` preserves the staff metrics, PCA reconstruction floors at 8/16/32/64 components, fitted GP kernel descriptions, and kNN sanity check.
- `metrics.json` uses the same metric implementation as DeepONet run 11597463: Euclidean three-pixel edge bands, forward derivatives on the coordinate grid, valid gradient endpoints, and per-design aggregation. Do not compare the staff edge or gradient metrics directly to the DeepONet metrics: the staff uses Chebyshev edge bands and pixel-index gradients.
- Per-design JSON files and the first four predictions, targets, masks, and grids are saved separately for validation, test, and extrapolation.
- `data-inspection.json` records dataset checks. `config.json` records source hashes, seed, software versions, hardware host, and fit scope.

The wrapper observes the staff evaluation calls without changing PCA, GP training, or predictions. Batch-one inference timing includes GP prediction and PCA reconstruction, excludes file I/O, and uses five warmups and twenty repetitions on CPU. Staff batch timings remain in `staff-results.json`. CPU and GPU timings should be reported with their hardware and batching differences.
