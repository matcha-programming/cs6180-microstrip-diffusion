#!/bin/bash
#SBATCH --job-name=palace-data-pack
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/palace-pack-%j.log
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec tools/python/bin/python -u experiments/feedback_round4/palace_data.py pack --root data/palace-pilot
