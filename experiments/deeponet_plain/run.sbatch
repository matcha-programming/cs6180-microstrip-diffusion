#!/bin/bash
#SBATCH --job-name=microstrip-deeponet-plain
#SBATCH --partition=courses-gpu
#SBATCH --account=cs6180.202710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=deeponet-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the experiment directory}"
PYTHON_BIN="${MICROSTRIP_PYTHON:-python3}"
"${PYTHON_BIN}" - <<'PY'
import h5py, numpy, scipy, torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in this Python environment; training was not started.')
print('PyTorch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0), flush=True)
PY
exec "${PYTHON_BIN}" -u train.py \
  --data /courses/CS6180.202710/shared/final_proj/em2d_task.h5 \
  --output "runs/deeponet_plain_${SLURM_JOB_ID}" \
  --device cuda --epochs 100 --patience 15 --batch-size 16 --pixels 1024 \
  --width 128 --rank 64 --lr 0.001 --seed 6180
