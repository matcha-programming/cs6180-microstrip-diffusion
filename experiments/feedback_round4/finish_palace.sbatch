#!/bin/bash
#SBATCH --job-name=palace-verify-pack
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/palace-finish-%j.log
set -euo pipefail
module load OpenMPI/4.1.6 OpenBLAS/0.3.29
export LD_LIBRARY_PATH="$PWD/tools/gl-runtime/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
tools/python/bin/python -u experiments/feedback_round4/verify_palace_dataset.py cases --root data/palace-pilot
bash experiments/feedback_round4/palace_pack.sbatch
tools/python/bin/python -u experiments/feedback_round4/verify_palace_dataset.py dataset --root data/palace-pilot
