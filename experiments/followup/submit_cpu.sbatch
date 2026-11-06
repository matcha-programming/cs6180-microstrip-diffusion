#!/bin/bash
#SBATCH --job-name=microstrip-r2-submit
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=round2-launch-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
exec /scratch/he.zit/envs/meep-gpu/bin/python -u experiments/followup/campaign.py submit
