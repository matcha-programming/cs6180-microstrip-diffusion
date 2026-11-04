#!/bin/bash
#SBATCH --job-name=microstrip-experiments-3-6
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=remaining-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from experiments/remaining}"
CAMPAIGN_ROOT="$(cd ../.. && pwd)"
RESULT_ROOT="${CAMPAIGN_ROOT}/runs/${RUN_ID:-${SLURM_JOB_ID}}"
mkdir -p "${CAMPAIGN_ROOT}/data" "${CAMPAIGN_ROOT}/inputs" "${RESULT_ROOT}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR="${RESULT_ROOT}/.mpl-cache"
PYTHON_BIN="${MICROSTRIP_PYTHON:-/scratch/he.zit/envs/meep-gpu/bin/python}"
DATA_SOURCE=/courses/CS6180.202710/shared/final_proj/em2d_task.h5
PLAIN_SOURCE=/scratch/he.zit/cs6180-microstrip-experiments/deeponet_plain/runs/deeponet_plain_11597463/best.pt
if [ ! -f "${CAMPAIGN_ROOT}/data/em2d_task.h5" ]; then
  cp "${DATA_SOURCE}" "${CAMPAIGN_ROOT}/data/em2d_task.h5"
fi
if [ ! -f "${CAMPAIGN_ROOT}/inputs/plain_best.pt" ]; then
  cp "${PLAIN_SOURCE}" "${CAMPAIGN_ROOT}/inputs/plain_best.pt"
fi
echo '7670297247077ee223a572b724e45236a6970a1f38da02f9a96e074ca0215af2' "${CAMPAIGN_ROOT}/data/em2d_task.h5" | sha256sum -c -
nvidia-smi > "${RESULT_ROOT}/gpu.txt"
lscpu > "${RESULT_ROOT}/cpu.txt"
"${PYTHON_BIN}" -u run_pipeline.py \
  --data "${CAMPAIGN_ROOT}/data/em2d_task.h5" \
  --plain-checkpoint "${CAMPAIGN_ROOT}/inputs/plain_best.pt" \
  --output "${RESULT_ROOT}"
