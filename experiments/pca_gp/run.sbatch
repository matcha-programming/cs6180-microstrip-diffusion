#!/bin/bash
#SBATCH --job-name=microstrip-pca-gp-32
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=pca-gp-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the experiment directory}"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/microstrip-mpl-${SLURM_JOB_ID}"
PYTHON_BIN="${MICROSTRIP_PYTHON:-/shared/centos7/anaconda3/2022.05/bin/python3}"
"${PYTHON_BIN}" -u run_baseline.py \
  --data /courses/CS6180.202710/shared/final_proj/em2d_task.h5 \
  --output "runs/pca_gp_${SLURM_JOB_ID}"
