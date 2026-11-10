#!/bin/bash
#SBATCH --job-name=r3-plateau-resume
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/plateau-resume-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
exec /scratch/he.zit/envs/meep-gpu/bin/python -u experiments/round3_recovery/resume.py
