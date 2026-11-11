#!/bin/bash
#SBATCH --job-name=palace-data-generate
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/palace-generate-%A_%a.log
set -euo pipefail
module load OpenMPI/4.1.6 OpenBLAS/0.3.29
export LD_LIBRARY_PATH="$PWD/tools/gl-runtime/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec tools/python/bin/python -u experiments/feedback_round4/palace_data.py run --root data/palace-pilot --index "${SLURM_ARRAY_TASK_ID}" --palace "$PWD/tools/palace/install/bin/palace"
