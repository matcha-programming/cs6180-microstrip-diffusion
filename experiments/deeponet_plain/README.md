# Experiment 2: plain DeepONet

Regression baseline with separate branch and coordinate networks. Each is a two-hidden-layer MLP of width 128 with SiLU activations. Rank is 64 per field channel; total parameter count is 67,074. Pixel coordinates in units of h are divided by 8 before the trunk. No Fourier features are used.

## Training

Use the staff HDF5 file and its fixed splits. Seed: 6180. AdamW learning rate: 0.001; weight decay: 0.000001. Batch size: 16 designs. Each training step samples 1,024 valid pixels per design with replacement. Validation uses every valid pixel. Train for up to 100 epochs with patience 15; select the checkpoint with the lowest validation masked MSE.

The scheduler request is one GPU on `courses-gpu`, four CPUs, 16 GB RAM, and a one-hour wall-time limit. The program refuses to run the experiment on CPU if CUDA is unavailable.

## Submit on Explorer

Place this directory in your own cluster workspace. Activate a Python environment containing PyTorch with CUDA support, h5py, NumPy, and SciPy. From this directory:

```sh
python3 -c 'import torch, h5py, numpy, scipy; print(torch.__version__)'
sbatch run.sbatch
```

To use a specific environment interpreter:

```sh
MICROSTRIP_PYTHON=/path/to/environment/bin/python sbatch run.sbatch
```

The submission script uses the `cs6180.202710` course allocation.

## Outputs

Outputs go to `runs/deeponet_plain_JOBID/`: `config.json`, `history.jsonl`, `best.pt`, `metrics.json`, per-design metrics, and four example predictions for each test split. The job log is `deeponet-JOBID.log`.

Test and extrapolation results are computed separately after training from the best validation checkpoint. Reported metrics are the mean, median, and standard deviation of per-design relative L2, edge relative L2, coordinate-gradient relative L2, and masked MSE. Edge pixels are within a Euclidean pixel distance of three from the mask. Forward-difference gradients use the nonuniform y/h and z/h grids and exclude stencils touching the mask. Inference timing is batch-one, model-only, warmed up and synchronized on CUDA.

## Validation status

The downloaded 9,789-design dataset passed shape, finite-value, coordinate-order, and split-disjointness checks. A CPU smoke test completed two small training batches, checkpoint save/load, and evaluation on four examples from each test split. Smoke-test scores are not experiment results. HPC run 11597463 completed successfully on a Tesla P100 GPU. See [results and artifacts](../../results/deeponet_plain/11597463/).
