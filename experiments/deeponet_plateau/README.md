# Plain DeepONet to a validation plateau

Continue the published original model and AdamW state from its best checkpoint, epoch 95 of job 11597463. The architecture, four inputs, coordinates, sampled-pixel loss, optimizer and learning rate are unchanged. The original source and checkpoint are checked against pinned SHA-256 hashes. This is a new HPC continuation, not the separate-machine trial shown in the discussion; its reported 4.55-fold validation-MSE improvement is a reference to verify, not an achieved result of this run.

The original checkpoint has no random-generator state. The first continuation uses seed 6180 for a new sampling stream; it is not a bit-exact resumption of epoch 96. Subsequent continuations save and restore model, optimizer, all random states, best weights and plateau counters. Original artifacts remain unchanged.

After each additional 50 epochs, measure the relative decrease in the best full-validation masked MSE since the previous window boundary. Stop after **two consecutive, non-overlapping windows with less than 1% improvement**. There is no scientific epoch cap. This operational stopping criterion is not proof of a global optimum. Test and extrapolation scores do not affect training or stopping. Evaluate the best validation checkpoint only after the plateau; report both held-out splits using the original metric functions.

Each scheduled segment requests one V100, four CPUs, 16 GB memory and eight hours. It checkpoints and pauses after seven training hours if the plateau has not been reached. Such a pause must be resumed after inspecting the scheduler; it must not be labeled convergence. A process lock prevents concurrent workers writing the same run. The launch script recognizes last.pt and resumes it. Do not submit while a prior segment is pending or running.

Submit from the isolated extracted campaign root with `sbatch experiments/deeponet_plateau/run.sbatch`. Outputs are under `runs/plain-plateau/`. Publish the final selected checkpoint, full history, configuration, status, metrics, per-design records and examples; retain last.pt for recovery outside GitHub.
