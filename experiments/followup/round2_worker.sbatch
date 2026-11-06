#!/bin/bash
#SBATCH --job-name=microstrip-r2
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=round2-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
exec /scratch/he.zit/envs/meep-gpu/bin/python -u experiments/followup/campaign.py worker
