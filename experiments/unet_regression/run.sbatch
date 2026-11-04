#!/bin/bash
#SBATCH --job-name=microstrip-4r-pair
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=paired-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from experiments/unet_regression}"
CAMPAIGN_ROOT="$(cd ../.. && pwd)"
RESULT_ROOT="${CAMPAIGN_ROOT}/runs/${RUN_ID:-${SLURM_JOB_ID}}"
mkdir -p "${RESULT_ROOT}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR="${RESULT_ROOT}/.mpl-cache"
DATA=/courses/CS6180.202710/shared/final_proj/em2d_task.h5
REGRESSOR=/scratch/he.zit/cs6180-microstrip-experiments/remaining-20261103/runs/11603443/fourier/frozen_regressor.pt
echo '7670297247077ee223a572b724e45236a6970a1f38da02f9a96e074ca0215af2' "${DATA}" | sha256sum -c -
echo '131263244fb6b90dd65f0c29be13a8566e4f83818b2c80bbf64775c86e9d2438' "${REGRESSOR}" | sha256sum -c -
nvidia-smi > "${RESULT_ROOT}/gpu.txt"
lscpu > "${RESULT_ROOT}/cpu.txt"
/scratch/he.zit/envs/meep-gpu/bin/python -u run_pair.py \
  --data "${DATA}" --regressor "${REGRESSOR}" --output "${RESULT_ROOT}"
