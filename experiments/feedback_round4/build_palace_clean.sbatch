#!/bin/bash
#SBATCH --job-name=palace-clean-recovery
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/palace-clean-%j.log
set -euo pipefail
# Anaconda's CPATH overrides CMake's system include paths, mixing fmt 9 headers
# with Palace's fmt 12 library. Rebuild the application in a fresh directory.
module purge
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH LIBRARY_PATH
module load cmake/3.30.2 OpenMPI/4.1.6 OpenBLAS/0.3.29
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OMPI_CXX="$PWD/tools/compiler-gcc13/bin/x86_64-conda-linux-gnu-g++"
export LD_LIBRARY_PATH="$PWD/tools/compiler-gcc13/lib:${LD_LIBRARY_PATH:-}"
test -x "$OMPI_CXX"
/scratch/he.zit/envs/meep-gpu/bin/python - <<'PY'
import json, pathlib, subprocess
root = pathlib.Path.cwd()
config = json.loads((root / 'tools/palace-gcc13-configure.json').read_text())
args = config['argv']
index = args.index('-B') + 1
args[index] = str(root / 'tools/palace/build-gcc13-clean-app')
config['environment'] = 'Anaconda unloaded; CPATH and language include/library search overrides unset'
(root / 'tools/palace-clean-configure.json').write_text(json.dumps(config, indent=2) + '\n')
subprocess.run(args, check=True)
PY
cmake --build tools/palace/build-gcc13-clean-app -j "${SLURM_CPUS_PER_TASK}"
cmake --install tools/palace/build-gcc13-clean-app
bash experiments/feedback_round4/finalize_palace.sbatch
