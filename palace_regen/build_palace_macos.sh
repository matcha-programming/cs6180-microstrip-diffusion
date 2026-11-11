#!/usr/bin/env bash
# Build Palace (CPU, serial MPI ranks) on Apple-silicon macOS with Homebrew tools.
# Prerequisites: brew install cmake pkgconf mpich openblas   (pulls in gcc for gfortran)
# Usage: build_palace_macos.sh <palace source dir> [jobs]
set -euo pipefail
SRC=$(cd "$1" && pwd); JOBS=${2:-8}
export CC=clang CXX=clang++ FC=gfortran
export OPENBLAS_DIR=$(brew --prefix openblas)
# CMake 4 rejects cmake_minimum_required < 3.5 in some third-party dependencies.
export CMAKE_POLICY_VERSION_MINIMUM=3.5
mkdir -p "$SRC/build" && cd "$SRC/build"
cmake "$SRC" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$SRC/install" \
  -DPALACE_WITH_ARPACK=ON -DPALACE_WITH_SLEPC=OFF \
  -DPALACE_WITH_SUPERLU=ON -DPALACE_WITH_STRUMPACK=OFF -DPALACE_WITH_MUMPS=OFF \
  -DPALACE_WITH_GSLIB=ON \
  -DPALACE_WITH_LIBXSMM=OFF -DPALACE_WITH_MAGMA=OFF -DPALACE_WITH_SUNDIALS=OFF \
  -DPALACE_WITH_OPENMP=OFF
make -j"$JOBS"
"$SRC/install/bin/palace" --version || ls "$SRC/install/bin"
