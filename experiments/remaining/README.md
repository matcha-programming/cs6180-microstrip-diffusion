# Experiments 3–6

This pipeline compares a Fourier-feature DeepONet, direct conditional diffusion, and residual diffusion using the fixed course splits. It reuses the data loader and metric definitions from the plain DeepONet run.

## Protocol

- Fourier search: fixed 128×2 Gaussian frequency matrix, scales 0.5, 1, and 2; 100 epochs maximum per scale, validation early stopping after 15 epochs. Coordinates are divided by 8 before the Fourier transform. The branch width, trunk width, rank, optimizer, and pixel sampling match the plain model.
- The lower validation masked MSE selects the frozen comparator from the plain and Fourier models. Test targets do not enter this choice.
- Diffusion: four-level conditional U-Net, base width 36, 1,000 linear noise steps, AdamW at 0.0002, batch size 16, EMA decay 0.999. Only the field channels are noised. Training ends at 200 epochs, 30 epochs without validation improvement, or two hours per model, whichever comes first. The recorded stop reason and learning curves determine whether this budget was sufficient.
- Direct targets and residual targets use channel means and standard deviations fitted on valid training pixels. Residual conditioning consists of the two frozen predicted field channels, standardized using training-field statistics. Both models receive the four design parameters through the time embedding.
- The best EMA checkpoint is selected by fixed-seed validation noise MSE. DDIM uses 20 and 50 steps with eta 0; validation relative L2 selects the primary step count before test scoring. Both step counts are reported.
- Masked pixels are zeroed after every step. Predicted clean fields are clipped componentwise to the known peak-normalized range [-1, 1] during sampling. This constraint also applies after adding the frozen prediction in residual diffusion.
- Three fixed sample seeds (6180–6182) provide single-sample results, variation across draws, and three-sample ensemble results. Ensemble cost is reported separately.

Metrics are per-design relative L2, a Euclidean three-pixel edge band, and forward coordinate gradients whose endpoints are both valid. The test and extrapolation test remain separate. The first four designs in each fixed split supply examples; they are not selected by model error.

## Run

Keep this directory alongside `experiments/deeponet_plain/train.py`. On Explorer, submit from this directory:

```sh
sbatch run.sbatch
```

The job requests one GPU for eight hours and runs stages sequentially. The batch script verifies the course dataset checksum and copies the completed plain checkpoint. `status.json`, stage logs, configurations, source hashes, histories, per-design metrics, example arrays, and best checkpoints are written under `runs/<job-id>/` at the campaign root. Optimizer checkpoints (`last.pt`) remain on HPC for recovery.

For a short local integration check:

```sh
python run_pipeline.py --data /path/to/em2d_task.h5 --plain-checkpoint /path/to/plain/best.pt --output /tmp/microstrip-smoke --smoke
python -m unittest discover -s . -p test_methods.py -v
```

Smoke outputs use reduced data, smaller diffusion networks, and shorter sampling schedules. They are marked `smoke_test_only` and are not experiment results.
