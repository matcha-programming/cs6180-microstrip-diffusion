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
# The Conda compiler uses its own sysroot and does not search /usr/include.
# Copy only the headers for the system zlib library already selected by MFEM;
# adding all of /usr/include would mix incompatible system and sysroot headers.
mkdir -p tools/system-zlib-headers
cp -p /usr/include/zlib.h /usr/include/zconf.h tools/system-zlib-headers/
sha256sum tools/system-zlib-headers/*.h > "tools/system-zlib-headers-${SLURM_JOB_ID}.sha256"
/scratch/he.zit/envs/meep-gpu/bin/python - <<'PY'
import json, os, pathlib, subprocess
root = pathlib.Path.cwd()
config = json.loads((root / 'tools/palace-gcc13-configure.json').read_text())
args = config['argv']
index = args.index('-B') + 1
args[index] = str(root / 'tools/palace/build-gcc13-clean-app')
headers = str(root / 'tools/system-zlib-headers')
args = [arg for arg in args if not arg.startswith('-DCMAKE_CXX_FLAGS=')]
args.extend(['-DCMAKE_CXX_FLAGS=-isystem ' + headers, '-DZLIB_INCLUDE_DIR=' + headers])
# The original static ARPACK library contains non-PIE relocations. Match that
# build at executable link time; no solver or training source changes are needed.
args = [arg for arg in args if not arg.startswith('-DCMAKE_EXE_LINKER_FLAGS=')]
# ARPACK's Fortran link interface adds GCC 11's library directory. Put the new
# C++ runtime first so GCC 13 objects do not link against the older libstdc++.
args.append('-DCMAKE_EXE_LINKER_FLAGS=-no-pie -L' + str(root / 'tools/compiler-gcc13/lib'))
config['argv'] = args
catch2 = root / 'tools/palace/build-gcc13-app/_deps/catch2-src'
head = subprocess.check_output(['git', '-C', str(catch2), 'rev-parse', 'HEAD'], text=True).strip()
tag = subprocess.check_output(['git', '-C', str(catch2), 'rev-parse', 'v3.8.1^{commit}'], text=True).strip()
assert head == tag == '2b60af89e23d28eefc081bc930831ee9d45ea58b'
assert not subprocess.check_output(['git', '-C', str(catch2), 'status', '--porcelain'], text=True).strip()
# The same pinned source was already downloaded by the prior build. This
# avoids another compute-node network fetch without disabling the unit tests.
args.append('-DFETCHCONTENT_SOURCE_DIR_CATCH2=' + str(catch2))
config['environment'] = 'Anaconda unloaded; CPATH and language include/library search overrides unset'
config['catch2_git_commit'] = head
(root / f"tools/palace-clean-configure-{os.environ['SLURM_JOB_ID']}.json").write_text(json.dumps(config, indent=2) + '\n')
subprocess.run(args, check=True)
PY
cmake --build tools/palace/build-gcc13-clean-app -j "${SLURM_CPUS_PER_TASK}"
cmake --install tools/palace/build-gcc13-clean-app
bash experiments/feedback_round4/finalize_palace.sbatch
