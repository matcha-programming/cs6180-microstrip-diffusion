#!/bin/bash
#SBATCH --job-name=palace-gcc13-recovery
#SBATCH --account=cs6180.202710
#SBATCH --partition=courses
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/palace-gcc13-%j.log
set -euo pipefail
module load cmake/3.30.2 OpenMPI/4.1.6 OpenBLAS/0.3.29 anaconda3/2024.06
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CONDA_PKGS_DIRS=/scratch/he.zit/conda-pkgs
ROOT=$PWD
COMPILER="$ROOT/tools/compiler-gcc13"
# GCC 11's libstdc++ cannot compile Palace v0.19's constexpr std::string code.
# Install a private compiler in scratch and rebuild only the Palace application.
# The original build, dependencies and logs remain available for inspection.
if [[ ! -x "$COMPILER/bin/x86_64-conda-linux-gnu-g++" ]]; then
  conda create --yes --override-channels -c conda-forge --prefix "$COMPILER" 'gxx_linux-64=13'
fi
conda list --prefix "$COMPILER" --explicit > tools/compiler-gcc13-explicit.txt
export OMPI_CXX="$COMPILER/bin/x86_64-conda-linux-gnu-g++"
export LD_LIBRARY_PATH="$COMPILER/lib:${LD_LIBRARY_PATH:-}"
"$OMPI_CXX" --version > tools/compiler-gcc13-version.txt
cat > tools/compiler-gcc13-check.cpp <<'CPP'
#include <string>
constexpr std::string label() { return "Palace"; }
int main() { return label() == "Palace" ? 0 : 1; }
CPP
mpicxx -std=c++20 tools/compiler-gcc13-check.cpp -o tools/compiler-gcc13-check
tools/compiler-gcc13-check
/scratch/he.zit/envs/meep-gpu/bin/python - <<'PY'
import json, os, pathlib, subprocess
root = pathlib.Path.cwd()
source = root / 'tools/palace'
record = (source / 'build/palace-cmake/tmp/palace-cfgcmd.txt').read_text().strip()
assert record.startswith("cmd='") and record.endswith("'")
args = [arg.replace('$<SEMICOLON>', ';') for arg in record[5:-1].split(';')]
assert args[1] == '<SOURCE_DIR>'
args[1:2] = ['-S', str(source / 'palace'), '-B', str(source / 'build-gcc13-app')]
compiler_lib = str(root / 'tools/compiler-gcc13/lib')
for i, arg in enumerate(args):
    if arg.startswith('-DCMAKE_INSTALL_RPATH='):
        args[i] = arg + ';' + compiler_lib
args.append('-DCMAKE_BUILD_RPATH=' + compiler_lib)
(root / 'tools/palace-gcc13-configure.json').write_text(json.dumps({
    'argv': args, 'OMPI_CXX': os.environ['OMPI_CXX'],
    'dependency_build': 'GCC 11.5; application rebuilt with GCC 13',
}, indent=2) + '\n')
subprocess.run(args, check=True)
PY
cmake --build tools/palace/build-gcc13-app -j "${SLURM_CPUS_PER_TASK}"
cmake --install tools/palace/build-gcc13-app
bash experiments/feedback_round4/finalize_palace.sbatch
