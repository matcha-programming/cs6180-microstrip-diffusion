#!/bin/bash
#SBATCH --job-name=palace-arpack-recovery
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/palace-arpack-%j.log
set -euo pipefail
module load cmake/3.30.2 OpenMPI/4.1.6 OpenBLAS/0.3.29
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Palace v0.19 requires an eigensolver even for an electrostatic-only workload.
# Keep completed MFEM/HYPRE builds and their explicit LAPACK configuration.
command -v gfortran
cmake -S tools/palace -B tools/palace/build -DPALACE_WITH_ARPACK=ON -DCMAKE_Fortran_COMPILER=gfortran
cmake --build tools/palace/build -j "${SLURM_CPUS_PER_TASK}"
bash experiments/feedback_round4/finalize_palace.sbatch
