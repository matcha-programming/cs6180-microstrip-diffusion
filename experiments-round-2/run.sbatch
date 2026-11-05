#!/bin/bash
#SBATCH --job-name=microstrip-round2
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=round2-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from experiments-round-2}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${MICROSTRIP_PYTHON:-python3}"
STAGE="${ROUND2_STAGE:?Set ROUND2_STAGE to draws, continue, lr, or min-snr}"
DATA="${ROUND2_DATA:-/courses/CS6180.202710/shared/final_proj/em2d_task.h5}"
RESULT_ROOT="${ROUND2_OUTPUT:-../runs/round2-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}}"
mkdir -p "${RESULT_ROOT}"
nvidia-smi > "${RESULT_ROOT}/gpu-${SLURM_ARRAY_TASK_ID:-0}.txt"
lscpu > "${RESULT_ROOT}/cpu-${SLURM_ARRAY_TASK_ID:-0}.txt"
COMMAND=("${PYTHON_BIN}" -u run.py --stage "${STAGE}" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --data "${DATA}" --output "${RESULT_ROOT}")
if [[ "${STAGE}" == "continue" ]]; then
  COMMAND+=(--resume "${ROUND2_RESUME:?Set ROUND2_RESUME to the paired diffusion last.pt on HPC}")
fi
exec "${COMMAND[@]}"
