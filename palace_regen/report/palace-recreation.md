<div class="verdict" markdown="1">

**Short answer.** Palace is built on this Mac, and it reproduces the staff targets once the field is computed the way the staff evidently computed it. The stored `u` is not Palace's electric field. It is a finite-difference derivative of Palace's electrostatic potential, sampled on the 64 &times; 128 grid. With that recipe, the two thin-trace designs match to 0.7% and 0.9% after one scale factor. The medium designs match to 1.5–3.6% once their one to three spike pixels are left out. Thick traces match to 6–18%, which is limited by the staff mesh, not by the model.

</div>

## Bottom line

- **Palace v0.19.0 builds and runs on a laptop.** I compiled it from source on an M1 Max with Homebrew's MPICH, OpenBLAS and gfortran. One 2-D design meshes and solves in 1–13 seconds.
- **The targets are differences of the potential, not the field.** The staff used Palace's electrostatic solver at order 1, treating the dataset window as the whole domain. They sampled the potential V at the grid points and took E from central chord differences, (V[i+1] &minus; V[i&minus;1]) / (x[i+1] &minus; x[i&minus;1]). That reproduces two fingerprints that sampling the solver's own E field cannot. One is the jump ratio between rows 30 and 31. The other is E<sub>z</sub> varying linearly along mid-air rows while E<sub>y</sub> stays constant.
- **The odd interface rows in the audit are a sampling artifact.** Rows 31–33 straddle the dielectric step at z = h, so their difference stencils mix substrate and air values. Sampling Palace's E directly leaves those rows 20–26% off, and whole designs 5–18% off. With chord differences, the 31/30 jump ratio matches the dataset within 1% on the five designs with t/h below 0.15, and within 2% at t/h = 0.2.
- **The sign-flip spikes are not reproduced, but they are now localised.** Each spike matches one or two bad potential samples directly under the trace, at rows 30–31.
    - In design 52, the data fit samples that are too low by about 45% and 17% of the trace potential.
    - Zeroing the row-31 sample instead flips E<sub>z</sub> at row 30 and makes it the normalization peak, but the spike comes out three times too strong.
    - So the bad samples are partial errors, for example interpolation from the wrong cell, rather than misses filled with 0.
    - VTK's probe filter can produce errors of this kind: version 9.6.1, which pyvista's `sample` uses, silently missed points well inside Palace's cells in 12 of my 82 runs.
    - Confirming the cause needs the staff's sampling code.
- **Mode solves do not fit better.** Palace's 2-D mode solver at 1–10 GHz gives the same fields as electrostatics. At 30 GHz it fits worse, so the dataset is quasi-static.

## What I set up

- **Palace source:** a shallow clone of the `v0.19.0` release tag, kept outside this repo. The full history is 650 MB.
- **Palace build:**
    - The install takes 129 MB and the build tree 1.2 GB, which can be deleted afterwards.
    - The build took about 20 minutes, most of it cloning MFEM and hypre.
- **Homebrew packages needed:** `cmake`, `pkgconf`, `mpich` and `openblas`, plus `gcc` for gfortran.
- **Build options:**
    - On: ARPACK, which Palace requires; SuperLU; GSLIB, which point probes need.
    - Off: SLEPc, LIBXSMM, MAGMA, SUNDIALS and OpenMP.
    - The recipe is in `palace_regen/build_palace_macos.sh`.
- **Pipeline:** `palace_regen/` on this branch. The scripts:
    - `regen.py`: Gmsh mesh, Palace config and run.
    - `staffpipe.py`: the reconstructed staff sampling.
    - `compare.py`: metrics.
    - `probe_profile.py`: dense sampling near the interface.
    - `report_figs.py`: this page's figures.
- **Data:** nine dataset rows spanning t/h from 0.025 to 0.495:
    - two spike designs from the audit (52 and 3592) and one test_extrap design (4862)
    - designs 7185 and 8317, the worst-staircase and sign-flipped-peak cases from the audit
- **Version note:** the dataset was built at 01:28 on 2 November, before the v0.19.0 release. The staff therefore ran v0.18.1 or a development build. Nothing here depends on that, since 2-D meshes have been supported since v0.17.

## How the staff data was made

### Domain and boundary conditions

The dataset window is the whole simulation domain: y from &minus;(w/2 + 6h) to w/2 + 6h, and z from 0 to 8h.

- Row 0 has E<sub>y</sub> = 0 to 1e-16, which is a grounded plane at z = 0.
- The side columns have E<sub>y</sub> near zero, and the top row has E<sub>z</sub> near zero. That is Palace's natural zero-charge condition on walls left unassigned.
- The trace is the 1 V terminal, a w &times; t rectangle cut out of the mesh on top of a substrate of thickness h.

I rebuilt the grid in float64 as `linspace(-(w/2 + 6h), w/2 + 6h, 128)`. The stored float32 coordinates can put the end columns about 1e-7 outside the walls in relative terms, where probes fail. The potential is interpolated exactly inside each triangle of Palace's ParaView output. I first used VTK 9.6.1's probe filter, but it silently missed some interior points.

### The targets are differences of the potential

Element-wise, Palace's order-1 electrostatic field is constant, so sampling it gives runs of identical values everywhere. The dataset has such runs near the ground plane, where E<sub>y</sub> is exactly zero. In mid-air, though, E<sub>y</sub> is constant along rows while E<sub>z</sub> varies linearly. Chord differences of a piecewise-linear potential produce exactly this pattern:

- **Near the ground**, the fine z-grid keeps rows i&minus;1, i and i+1 inside one element, so the difference is that element's constant gradient.
- **In mid-air**, the rows are up to 0.46h apart, so the vertical stencil spans different elements and varies along the row.
- **At row 31**, the stencil runs from row 30 (0.90h, substrate) to row 32 (1.23h, air), across the step at z = h. That gives the blended value the audit flagged.

{{FIG:interface|Design row 0, column 20 (y = &minus;4.4h), around the dielectric step. The grey curve is the converged Palace field, squares sample it at the grid rows, and circles take chord differences of the sampled potential. Crosses are the dataset. Row 31 lies below the step but carries most of the air-side value. Only the circles follow it.}}

The jump ratio E<sub>z</sub>(row 31) / E<sub>z</sub>(row 30) is a sharp test because it depends on the stencil and hardly at all on the mesh:

{{TABLE:ratio}}

Away from the interface the two methods agree, and the difference is confined to the stencil rows:

{{FIG:rows|Per-row error on design row 0 after one scale factor, with both methods on the same mesh. Sampling E fails at rows 31–32 (shaded). Differencing the potential stays at 1–2% in every row, and most of that is the global scale.}}

The texture statistics give an independent check. Each cell below is the fraction of horizontally adjacent pixel pairs with bitwise-identical, nonzero values:

{{TABLE:sig}}

Sampling E gives identical E<sub>z</sub> pairs in mid-air at the same rate as E<sub>y</sub> pairs. The dataset almost never has them. Chord differences on a mesh of about 2t come within about 0.1 of all four numbers for design 7185.

### Mesh

The texture matches elements about the size of the trace thickness t. Overall error is lowest with uniform elements of 0.5–1t. The identical-run statistics near the ground fit about 2t better. A graded mesh, about 0.5t at the trace and 2t at the walls, is the best match for the thinnest design and no better than uniform elements elsewhere. Each cell below is the error after one scale factor:

{{TABLE:mesh}}

For thick traces (t/h above about 0.2), every mesh I tried leaves 6–18%. Their elements span several pixels, so the targets carry a texture that depends on the exact element layout. That cannot be matched without the staff's mesh script. These numbers are an upper bound on how much of a thick-trace target is mesh texture, not a measurement of it.

### Solver type and frequency

Electrostatics and Palace's 2-D mode solver were each sampled directly at the grid points. Each value below is the error after one scale factor:

{{TABLE:solvers}}

The 1 GHz and 10 GHz mode solves are indistinguishable from electrostatics. At 30 GHz the error doubles on the tall-substrate designs, rows 0 and 4862 with h of 0.6–0.8 mm, through dispersion. The staff's drop rule, sqrt(&epsilon;<sub>r</sub>)(2w + 0.8h) &le; 9.08 mm, matches the usual estimate of the first higher-order microstrip mode cutoff at 33 GHz, which is 30 GHz plus 10%. My reading is that a frequency study was planned, but the stored fields are quasi-static.

## Results by design

The fixed recipe used throughout is Palace electrostatic at order 1, a uniform mesh of about 1t, chord differences of V, and the trace potential inside the conductor. "Palace E" is the converged field (order 3, finer graded mesh) sampled at the grid points. All errors are relative L2 over unmasked pixels. Every column except "Chord V, raw" is taken after one best-fit scale factor.

{{TABLE:designs}}

- **The pixels left out are mostly spike pixels.** On designs 3592, 8053 and 52 the worst ones sit at rows 29–30 under the trace and carry 76–96% of the remaining squared error.
- **Design 52 shows how a spike propagates.** Its spike at row 30, column 69 is the normalization peak. That pixel and three of its neighbours are off. That is the footprint of one wrong potential sample at row 30 plus one at row 31, the latter in the masked row.
- **The raw error is larger mainly because of normalization.** s is the singular pixel at row 30, so it moves with the mesh. This is the ±3% edge-phase effect from the audit.

{{FIG:maps|E<sub>z</sub> for three designs: dataset, reproduction on the best of the ten meshes for each, and difference, after one scale factor, in pixel space. Masked pixels are blank, and each difference panel has its own scale. The thin design differs only by faint mesh texture, a few thousandths of the peak. Most of the medium design's error is one spike pixel under the trace, which carries 76% of the squared error, on top of texture at about 1.6% of the peak. The thick one differs in mesh texture throughout the substrate.}}

## What this means for the project

- **The targets are the staff's preprocessing of Palace output, not Palace's field.** A physics prior that puts the dielectric jump at z = h is penalised at rows 31–33. So is a coordinate model that resolves it cleanly. The stencil artifact is part of every target, so models should learn it rather than fight it.
- **"Treat Palace as golden ground truth" needs this recipe to hold.** A fresh Palace E field differs from the targets by 5–18% (scaled), mostly at rows 31–32. Any new evaluation data must go through the same potential-sampling and chord-difference step.
- **New designs are now cheap to generate.** At 1–13 seconds per design, the pipeline can add designs or test points, for example at extrapolated h. Fidelity is about 1–4% for t/h below about 0.15, setting spike pixels aside, and texture-limited above that.
- **Spike pixels can be detected and repaired.** They show as E<sub>z</sub> at rows 29–30 under the trace far below its neighbours, often with the wrong sign. Re-simulating a flagged design with this pipeline replaces them with clean values.
- **The texture floor for thick traces is real but bounded.** The element layout is a deterministic but highly non-smooth function of x, so smooth models will average it out. The 6–18% above is an upper bound on that component, since our mesh is not theirs.

## Still unknown

- **The staff's mesh script.** We know the element size is about 0.5–2t, the layout is unknown.
- **How they sampled V, and why a few samples just under the trace are wrong.** One or two partly wrong values account for each spike.
- **Order 1, inferred.** The identical-value runs need a piecewise-linear potential, and nothing else tested produces them.
- **The Palace version, and whether the frequency columns were ever used.**

The second item sharpens staff request 2, the preprocessing code at `593e5c0`. The question becomes: confirm chord differences of V, give the mesh sizing, and say how V was sampled, which would explain the bad samples under the trace. I have not sent it.

## Reproduce

```bash
cd palace_regen
.venv/bin/python -I regen.py --h5 ../data/em2d_task.h5 --palace PATH/TO/palace --out runs/new --rows 0 52 --problem electrostatic --order 1 --uniform 1.0 --save 1
.venv/bin/python -I staffpipe.py --h5 ../data/em2d_task.h5 --runs runs/new
```

`staffpipe.py` writes `staff_u_trace_chord.npz` next to each run, holding `u` in the dataset's convention, and prints its metrics against the stored targets. Run output goes to `runs/`, which is git-ignored; the runs behind this page took 1.7 GB.
