# Experiment 4R: deterministic U-Net regression

This control tests whether diffusion improves on direct convolutional regression when both models use the same backbone and design conditioning. It implements the pairing added in [plan commit e541972](https://github.com/matcha-programming/cs6180-microstrip-diffusion/commit/e5419728dc08888c17624ea9452dce53b65e99cc).

## Matched pair

Both models use the same four-level U-Net: base width 36, two residual blocks per level, bottleneck attention, 5,602,322 parameters, and four design parameters embedded through the shared time/design pathway. Fixed row and column index planes span [-1, 1] over the 64×128 image. These are position indices, not the nonuniform physical coordinates. They are identical for every design and in both models, and are never noised.

- **4R:** the two field input planes are zero, time is fixed at zero, and the U-Net predicts the two field channels deterministically. Training minimizes masked MSE in the original peak-normalized field units. The output channel transform uses valid training pixels only.
- **Paired 4:** the two field input planes contain the noisy standardized field. The same U-Net predicts noise at one of 1,000 diffusion timesteps, with masked noise MSE. Training and 20/50-step DDIM evaluation reuse the existing direct-diffusion implementation.

The job checks identical initial weights, parameter counts, and position planes before launching both models sequentially on one GPU. Each training process records its actual initialization hash, which the runner verifies against the pair configuration. Both use seed 6180, AdamW at 0.0002, batch size 16, EMA decay 0.999, gradient clipping at 1, and the same AMP policy. Each has a 200-epoch/two-hour maximum and patience 30. Actual epoch counts and elapsed GPU time are reported, since equal limits do not guarantee equal work.

4R selects its EMA checkpoint by full validation masked field MSE. Diffusion selects its EMA checkpoint by fixed-seed validation noise MSE, then selects DDIM steps by validation field relative L2. Test and extrapolation targets do not enter selection. Both use the same known component bound [-1,1] at evaluation and zero masked outputs; 4R also reports unclipped errors. The deterministic model has one forward pass and no ensemble cost. Diffusion reports all three seeds and their three-draw average separately.

The original Experiment 4 run in job `11603443` has no position planes. Keep it as a separate result. The position-conditioned DDPM in this pair is the controlled comparator for 4R; it is not a continuation of the original checkpoint.

## Run

Keep this folder alongside `experiments/remaining/` and `experiments/deeponet_plain/train.py`. On Explorer, from this directory:

```sh
sbatch run.sbatch
```

Results go to `runs/<job-id>/regression/` and `runs/<job-id>/diffusion/` at the campaign root, with stage logs, `pair-config.json`, GPU details, configuration, histories, best EMA checkpoints, matched per-design metrics, and fixed examples. Optimizer recovery checkpoints remain on HPC.

```sh
python -m unittest discover -s . -p test_pair.py -v
python run_pair.py --data /path/to/em2d_task.h5 --regressor /path/to/frozen_regressor.pt --output /tmp/pair-smoke --smoke
```

Smoke outputs are explicitly marked and are excluded from reported experiments. The published comparison must include both the original and position-conditioned diffusion variants, the deterministic U-Net, PCA+GP, and the plain/Fourier DeepONets, with training budget and hardware differences stated.
