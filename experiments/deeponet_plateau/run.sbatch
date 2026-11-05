#!/bin/bash
#SBATCH --job-name=microstrip-plain-plateau
#SBATCH --partition=courses-gpu
#SBATCH --account=cs6180.202710
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=plain-plateau-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
resume=()
if [[ -f runs/plain-plateau/last.pt ]]; then resume=(--resume); fi
exec /scratch/he.zit/envs/meep-gpu/bin/python -u experiments/deeponet_plateau/train.py \
  --data /courses/CS6180.202710/shared/final_proj/em2d_task.h5 \
  --initial results/deeponet_plain/11597463/best.pt \
  --output runs/plain-plateau "${resume[@]}"
