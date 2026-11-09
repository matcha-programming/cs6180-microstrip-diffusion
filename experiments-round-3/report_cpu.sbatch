#!/bin/bash
#SBATCH --job-name=microstrip-r3-report
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
set -euo pipefail
cd "${ROUND3_CAMPAIGN_ROOT:?Set by the preflight submitter}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
exec /scratch/he.zit/envs/meep-gpu/bin/python -u experiments-round-3/submit.py report
