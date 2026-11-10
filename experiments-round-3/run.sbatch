#!/bin/bash
#SBATCH --job-name=microstrip-round3
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=round3-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from experiments-round-3}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="${MICROSTRIP_PYTHON:-python3}"
PHASE="${ROUND3_PHASE:-screen}"
RESULT_ROOT="${ROUND3_OUTPUT:-../runs/round3-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}}"
mkdir -p "${RESULT_ROOT}"
TASK_TAG="${PHASE}-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}-${SLURM_ARRAY_TASK_ID:-0}"
nvidia-smi > "${RESULT_ROOT}/gpu-${TASK_TAG}.txt"
lscpu > "${RESULT_ROOT}/cpu-${TASK_TAG}.txt"
COMMAND=("${PYTHON_BIN}" -u run.py --phase "${PHASE}" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --output "${RESULT_ROOT}")
if [[ "${PHASE}" == "confirm" ]]; then
  COMMAND+=(--plan "${ROUND3_PLAN:?Set ROUND3_PLAN to the frozen confirmation-plan.json}")
fi
exec "${COMMAND[@]}"
