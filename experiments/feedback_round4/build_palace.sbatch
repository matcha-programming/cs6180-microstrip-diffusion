#!/bin/bash
#SBATCH --job-name=palace-build-r4
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/palace-build-%j.log
set -euo pipefail
module load cmake/3.30.2 OpenMPI/4.1.6 OpenBLAS/0.3.29
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PIP_CACHE_DIR=/scratch/he.zit/pip-cache
ROOT=$PWD
TOOLS="$ROOT/tools"
mkdir -p "$TOOLS"
if [[ ! -d "$TOOLS/palace/.git" ]]; then
 git clone --depth 1 --branch v0.19.0 https://github.com/awslabs/palace.git "$TOOLS/palace"
fi
git -C "$TOOLS/palace" rev-parse HEAD > "$TOOLS/palace-source-sha.txt"
cmake -S "$TOOLS/palace" -B "$TOOLS/palace/build" \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$TOOLS/palace/install" \
 -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx \
 -DPALACE_WITH_SUPERLU=OFF -DPALACE_WITH_SLEPC=OFF -DPALACE_WITH_ARPACK=OFF \
 -DPALACE_WITH_GSLIB=ON -DPALACE_WITH_LIBXSMM=OFF -DPALACE_WITH_MAGMA=OFF \
 -DPALACE_WITH_SUNDIALS=OFF -DPALACE_WITH_OPENMP=OFF
cmake --build "$TOOLS/palace/build" -j "${SLURM_CPUS_PER_TASK}"
"$TOOLS/palace/install/bin/palace" --version > "$TOOLS/palace-version.txt" 2>&1
/scratch/he.zit/envs/meep-gpu/bin/python -m venv --system-site-packages "$TOOLS/python"
"$TOOLS/python/bin/python" -m pip install gmsh==4.15.0 vtk==9.5.2
"$TOOLS/python/bin/python" -c 'import gmsh, vtk, scipy, h5py; print(gmsh.__version__, vtk.vtkVersion.GetVTKVersion())' > "$TOOLS/python-check.txt"
"$TOOLS/python/bin/python" -m pip freeze > "$TOOLS/python-freeze.txt"
touch "$TOOLS/build-complete"
