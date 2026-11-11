#!/bin/bash
#SBATCH --job-name=palace-build-recover
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/palace-build-recover-%j.log
set -euo pipefail
module load cmake/3.30.2 OpenMPI/4.1.6 OpenBLAS/0.3.29
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PIP_CACHE_DIR=/scratch/he.zit/pip-cache
TOOLS="$PWD/tools"
# The MFEM example link omitted BLAS/LAPACK although static HYPRE uses them.
# Preserve the completed compilation and enable MFEM's explicit LAPACK linkage.
cmake -S "$TOOLS/palace/build/extern/mfem" -B "$TOOLS/palace/build/extern/mfem-build" -DMFEM_USE_LAPACK=ON
cmake --build "$TOOLS/palace/build" -j "${SLURM_CPUS_PER_TASK}"
"$TOOLS/palace/install/bin/palace" --version > "$TOOLS/palace-version.txt" 2>&1
/scratch/he.zit/envs/meep-gpu/bin/python -m venv --system-site-packages "$TOOLS/python"
"$TOOLS/python/bin/python" -m pip install gmsh==4.15.0 vtk==9.5.2
"$TOOLS/python/bin/python" -c 'import gmsh, vtk, scipy, h5py; print(gmsh.__version__, vtk.vtkVersion.GetVTKVersion())' > "$TOOLS/python-check.txt"
"$TOOLS/python/bin/python" -m pip freeze > "$TOOLS/python-freeze.txt"
touch "$TOOLS/build-complete"
