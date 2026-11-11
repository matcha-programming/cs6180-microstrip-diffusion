# Palace re-simulation of em2d_task.h5

Re-simulates dataset designs with Palace and reproduces the stored targets `u`. The full write-up, with figures and per-design tables, is [report/palace-recreation.html](report/palace-recreation.html).

## What the staff pipeline appears to be

This is reconstructed from the data, not from staff code.

1. **Solve.** Palace `Electrostatic` at `Solver.Order` 1, on the 2-D window y in ±(w/2 + 6h), z in [0, 8h]:
   - the trace is the terminal, a w × t rectangle cut out of the mesh on top of the substrate
   - Ground is at z = 0
   - the walls are left natural, which means zero normal D
   - the substrate has permittivity ε_r for 0 < z < h, with air above
2. **Sample.** The potential V is sampled at the 64 × 128 grid. y is `linspace(-(w/2 + 6h), w/2 + 6h, 128)`, and z is `z_over_h * h`. Points inside the trace take the trace potential.
3. **Differentiate.** E = −grad V by central chord differences, `np.gradient(V, axis) / np.gradient(coord)`.
4. **Normalize.** u = E / max |E| over unmasked pixels, with masked pixels set to 0.

**Agreement after one scale factor:**
- t/h < 0.03: 0.7–0.9%
- t/h 0.07–0.14, setting aside the 1–3 spike pixels under the trace: 1.5–3.6%
- thick traces: 6–18%, limited by mesh texture

Sampling Palace's E directly is 5–18% off, mostly at rows 31–33, which straddle the dielectric step.

## Files

- `build_palace_macos.sh` builds Palace with Homebrew cmake, pkgconf, mpich and openblas (plus gcc for gfortran).
- `regen.py` handles one design: it builds the mesh, writes the config, runs Palace and samples E with Palace probes. It supports `--problem boundarymode` too.
- `staffpipe.py` applies the reconstructed staff sampling. It needs runs made with `--save 1`.
- `resample.py` does the exact P1 interpolation of V and the chord differences. Avoid vtkProbeFilter here: it silently misses interior points.
- `compare.py` computes the metrics against the stored targets.
- `probe_profile.py` samples densely in z around the interface.
- `report_figs.py` makes the figures and tables in `report/figs/`, and `report/build_report.py` builds the page from them. The page build needs python-markdown.
- `run_matrix.sh` runs the solver and frequency comparison.

## Run

```bash
.venv/bin/python -I regen.py --h5 ../data/em2d_task.h5 --palace PATH/TO/palace \
  --out runs/new --rows 0 52 --problem electrostatic --order 1 --uniform 1.0 --save 1
.venv/bin/python -I staffpipe.py --h5 ../data/em2d_task.h5 --runs runs/new
```

Set up the tools first. numpy, scipy, h5py, matplotlib and vtk come from your Python, and gmsh comes from pip:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install gmsh
```

Run output goes to `runs/`. Both `runs/` and `.venv/` are git-ignored.
