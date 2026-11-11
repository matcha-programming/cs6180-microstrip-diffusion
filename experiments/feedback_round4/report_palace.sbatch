#!/bin/bash
#SBATCH --job-name=palace-data-report
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/palace-data-report-%j.log
set -euo pipefail
export MPLCONFIGDIR="$PWD/tools/matplotlib-cache"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
tools/python/bin/python experiments/feedback_round4/report_palace.py
