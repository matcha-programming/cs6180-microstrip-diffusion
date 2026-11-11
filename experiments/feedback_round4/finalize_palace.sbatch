#!/bin/bash
#SBATCH --job-name=palace-env-check
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/palace-env-%j.log
set -euo pipefail
module load OpenMPI/4.1.6 OpenBLAS/0.3.29
export LD_LIBRARY_PATH="$PWD/tools/gl-runtime/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PIP_CACHE_DIR=/scratch/he.zit/pip-cache
TOOLS="$PWD/tools"
test -x "$TOOLS/palace/install/bin/palace"
"$TOOLS/palace/install/bin/palace" --version > "$TOOLS/palace-version.txt" 2>&1
/scratch/he.zit/envs/meep-gpu/bin/python -m venv --system-site-packages "$TOOLS/python"
"$TOOLS/python/bin/python" -m pip install gmsh==4.15.0 vtk==9.5.2
"$TOOLS/python/bin/python" -c 'import gmsh, vtk, scipy, h5py; print(gmsh.__version__, vtk.vtkVersion.GetVTKVersion())' > "$TOOLS/python-check.txt"
"$TOOLS/python/bin/python" -m pip freeze > "$TOOLS/python-freeze.txt"
touch "$TOOLS/build-complete"
