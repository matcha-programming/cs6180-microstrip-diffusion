#!/bin/bash
#SBATCH --job-name=microstrip-r4-report
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/report-%j.log
set -euo pipefail
exec /scratch/he.zit/envs/meep-gpu/bin/python experiments/feedback_round4/report.py --root runs/screen --output report
