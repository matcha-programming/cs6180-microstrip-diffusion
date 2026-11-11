#!/bin/bash
#SBATCH --job-name=palace-offline-runtime
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/palace-offline-%j.log
set -euo pipefail
module load OpenMPI/4.1.6 OpenBLAS/0.3.29
export LD_LIBRARY_PATH="$PWD/tools/gl-runtime/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TOOLS="$PWD/tools"
test -x "$TOOLS/palace/install/bin/palace"
"$TOOLS/palace/install/bin/palace" --version > "$TOOLS/palace-version.txt" 2>&1
"$TOOLS/python/bin/python" - <<'PY'
import hashlib, json, pathlib
root = pathlib.Path('tools/palace-linux-wheels')
for name, expected in json.loads((root / 'sha256.json').read_text()).items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print('Offline wheel checksums verified', flush=True)
PY
"$TOOLS/python/bin/python" -m pip install --no-index --find-links "$TOOLS/palace-linux-wheels" gmsh==4.15.0 vtk==9.5.2
"$TOOLS/python/bin/python" -c 'import gmsh, vtk, scipy, h5py; assert gmsh.__version__ == "4.15.0"; assert vtk.vtkVersion.GetVTKVersion() == "9.5.2"; print(gmsh.__version__, vtk.vtkVersion.GetVTKVersion())' > "$TOOLS/python-check.txt"
"$TOOLS/python/bin/python" -m pip freeze > "$TOOLS/python-freeze.txt"
touch "$TOOLS/build-complete"
