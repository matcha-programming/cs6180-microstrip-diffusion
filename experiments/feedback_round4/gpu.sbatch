#!/bin/bash
#SBATCH --job-name=microstrip-feedback
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/feedback-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=/scratch/he.zit/envs/meep-gpu/bin/python
nvidia-smi
exec "$PY" -u experiments/feedback_round4/train.py --config experiments/feedback_round4/campaign.json --task "${SLURM_ARRAY_TASK_ID}" --output runs/screen
