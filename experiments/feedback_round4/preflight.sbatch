#!/bin/bash
#SBATCH --job-name=microstrip-r4-preflight
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/preflight-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=/scratch/he.zit/envs/meep-gpu/bin/python
"$PY" -m unittest discover -s experiments/feedback_round4 -p test_feedback.py -v
for task in 0 1 2 3 4 5 6; do
 "$PY" -u experiments/feedback_round4/train.py --config experiments/feedback_round4/campaign.json --task "$task" --output "runs/smoke-${SLURM_JOB_ID}" --smoke
done
touch preflight-passed
