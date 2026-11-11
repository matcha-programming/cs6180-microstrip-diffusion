#!/usr/bin/env bash
# Re-simulate nine dataset designs under each candidate solver setup.
# Usage: run_matrix.sh <em2d_task.h5> <palace binary> <runs dir>
set -euo pipefail
H5=$1; PAL=$2; OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/.venv/bin/python -I $HERE/regen.py --h5 $H5 --palace $PAL --out $OUT --np 4"
ROWS="0 1 500 7185 8317 3592 52 4862 8053"
$PY --rows $ROWS --problem electrostatic --order 1 --ref 1
$PY --rows $ROWS --problem electrostatic --order 3 --ref 0.5
$PY --rows $ROWS --problem boundarymode --order 1 --ref 1 --freq 1 10 30
$PY --rows $ROWS --problem boundarymode --order 3 --ref 0.5 --freq 10
$PY --rows $ROWS --problem boundarymode --order 1 --uniform 1.0 --freq 10
$PY --rows $ROWS --problem electrostatic --order 1 --uniform 1.0
