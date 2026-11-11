#!/bin/bash
#SBATCH --job-name=palace-parmetis-recovery
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/palace-parmetis-%j.log
set -euo pipefail
module load cmake/3.30.2 OpenMPI/4.1.6 OpenBLAS/0.3.29
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT=$PWD
# Palace v0.19 requires ParMETIS even when direct solvers are disabled, but its
# superbuild only adds the dependency when a direct solver is enabled. Build the
# same pinned PETSc fork with the options in cmake/ExternalMETIS.cmake. Leave
# the completed MFEM, ARPACK and training sources unchanged.
SOURCE="$ROOT/tools/parmetis-required"
if [[ ! -d "$SOURCE/.git" ]]; then
  git clone --depth 1 --branch v4.0.3-p11 https://bitbucket.org/petsc/pkg-parmetis.git "$SOURCE"
fi
test "$(git -C "$SOURCE" describe --tags --exact-match)" = v4.0.3-p11
git -C "$SOURCE" rev-parse HEAD > tools/parmetis-source-sha.txt
cmake -S "$SOURCE" -B tools/parmetis-required-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_INSTALL_PREFIX="$ROOT/tools/palace/install" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DGKLIB_PATH="$ROOT/tools/palace/build/extern/metis/GKlib" \
  -DMETIS_PATH="$ROOT/tools/palace/install" -DSHARED=0 -DMSVC=0
cmake --build tools/parmetis-required-build -j "${SLURM_CPUS_PER_TASK}"
cmake --install tools/parmetis-required-build
cmake --build tools/palace/build -j "${SLURM_CPUS_PER_TASK}"
bash experiments/feedback_round4/finalize_palace.sbatch
