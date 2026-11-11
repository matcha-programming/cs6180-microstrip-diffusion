#!/bin/bash
#SBATCH --job-name=microstrip-val-audit
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/audit-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
exec /scratch/he.zit/envs/meep-gpu/bin/python -u experiments/feedback_round4/audit.py \
 --checkpoint /scratch/he.zit/cs6180-microstrip-experiments/round3-schedules-20261107-ec9dcf5e41-b5b556/runs/screen/R3-S2-warmup-cosine-seed6180/best-field.pt \
 --output runs/validation-audit
