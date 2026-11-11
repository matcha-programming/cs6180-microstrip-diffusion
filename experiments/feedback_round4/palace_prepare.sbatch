#!/bin/bash
#SBATCH --job-name=palace-data-prepare
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/palace-prepare-%j.log
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec /scratch/he.zit/envs/meep-gpu/bin/python -u experiments/feedback_round4/palace_data.py prepare --root data/palace-pilot --count 256
