#!/bin/bash
#SBATCH --job-name=palace-gl-runtime
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/gl-runtime-%j.log
set -euo pipefail
module load anaconda3/2024.06
export CONDA_PKGS_DIRS=/scratch/he.zit/conda-pkgs
conda create --yes --override-channels --channel conda-forge --prefix "$PWD/tools/gl-runtime" libglu
conda list --explicit --prefix "$PWD/tools/gl-runtime" > tools/gl-runtime-explicit.txt
