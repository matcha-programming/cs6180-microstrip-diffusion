# Next steps after the first comparison

## Questions to resolve

1. Does diffusion close the gap to the simple baseline and matched convolutional regression after adequate training and validation-based checkpoint selection?
2. How much of the error reflects stored-target artifacts, sampling variability, or true field-shape error?
3. Does the comparison hold across genuinely different normalized geometries and physical regimes?

The current results establish a budgeted comparison on one dataset. They do not establish that diffusion is generally inferior, that every run converged, or that a model with lower stored-target MSE is closer to physical truth.

## Work now

### Execute the existing Round 2 campaign

Use the committed [training and sampling protocol](../experiments-round-2/README.md), with the [submission wrapper](../experiments/followup/README.md). Continue the full optimizer checkpoint; run fresh learning-rate controls at 0.0001, 0.0002 and 0.0004; include the predeclared Min-SNR gamma-5 ablation. Re-evaluate the two historical direct diffusion checkpoints at 1, 2, 4, 8 and 16 draws on the fixed validation panel.

Retain field-selected and noise-selected checkpoints separately. Use full validation to freeze the final choice, then evaluate held-out splits once. Report actual optimizer updates, stopping reasons, hardware and elapsed time. A plateau must be observed; increasing a time limit alone does not demonstrate convergence. Compare the Min-SNR run to the fresh unweighted learning-rate control, not to continuation with a different optimization history.

The new campaign requests V100 GPUs and keeps the paired FP32 policy. Historical P100-versus-V100 timing is not a speed comparison. Re-measure regression and diffusion on the same device before reporting latency ratios for the new campaign. More sampling draws measure an accuracy/cost curve, not extra independent training evidence or calibrated physical uncertainty.

The teammate’s DeepONet retraining and Fourier tuning remain separate. At the user’s subsequent request, an independent [plain DeepONet continuation](../experiments/deeponet_plateau/README.md) starts from the published epoch-95 model and optimizer and runs to a measured validation plateau. It does not resume or claim the separate-machine trial’s results. Preserve the teammate’s branch and runs.

### Check split support and baseline strength

The [metadata audit](../results/followup/support-20261104/README.md) confirms that all 979 official `test_extrap` designs are outside training support along substrate thickness. Under the proposed quasi-static similarity coordinates, 906 of those designs lie inside the training convex hull. Report both observations. Do not rename the official split or describe it as universally non-extrapolative.

The dimensional argument is conditional: electrostatic equations with geometrically similar boundaries and peak-normalized outputs can remove a common length scale. The staff solver and preprocessing are not fully documented. Finite-frequency effects, meshing, sampling, and normalization can violate the simple similarity picture.

The [cheap kNN validation study](../results/followup/knn-validation-20261104/README.md) tests whether this feature choice helps an uncomplicated regressor. It does not replace PCA-32 + GP or 4R, and it has not used held-out target scores. Freeze its selected feature space and neighbor count before a later held-out comparison.

## Physics-fidelity gate before generating a larger dataset

The solver reconstruction lives on [`palace-rebuild` at f8d7df4](https://github.com/matcha-programming/cs6180-microstrip-diffusion/tree/f8d7df46d8201ef995b4d2b8efb985cb10d0af3e/palace_regen). That snapshot documents nine selected designs and a reconstructed potential-sampling recipe. Treat it as a useful hypothesis, not staff-confirmed provenance. The discussion's broader 147-design evaluation is not present in the inspected snapshot; obtain its exact report, row IDs, solver settings and output hashes before quoting that result.

Start with eight designs, four training and four validation, selected by metadata-only quantiles of t/h. Do not select them by model error. For each design, retain:

- The stored target, unchanged.
- A reconstructed coarse solve followed by the potential/chord-difference sampling convention.
- Three successively refined, order-3 electrostatic solves at refinement factors 1, 0.5 and 0.25, with the same geometry, boundary conditions and probe grid.
- Both the solver E field and the sampled-potential derivative. These are different observables near the dielectric interface and must not be mixed in one accuracy ranking.

Measure successive-refinement relative L2, trace-edge error, peak change and valid probe coverage. Predeclare a pilot acceptance target of below 1% whole-field change and below 3% edge change on the final two refinements; report failures and additional refinement requirements rather than calling the last run ground truth. Retain unnormalized fields at the same imposed potential as well as independently peak-normalized fields. Inspect probe misses before calculating convergence. Do not fit a separate scale factor against each evaluation target as a primary score. A scaled diagnostic, if useful, must be labeled separately.

Use this pilot to estimate solver time, storage and error. Do not generate 10× more data until the convention and convergence checks pass. Train/validation/test design IDs and geometry families must be frozen before that expansion. More samples of the same geometry distribution answer a data-scaling question, not a broader-design generalization question.

## Broader designs after the fidelity gate

Add a controlled two-trace family with trace widths, spacing and excitation recorded explicitly. The current four-number input cannot distinguish arbitrary two-trace layouts or independent voltages. Both regression and diffusion need the same geometry and excitation representation, such as raster material/terminal maps plus coordinates, and matched convolutional backbones. Hold out geometry families or normalized width/spacing regimes rather than only the common length scale. Keep the original single-trace benchmark for continuity.

Start with a small stratified pilot and learning curves before a large training set. Preserve raw solver outputs, masks, grids, material/terminal maps, solver version, meshing settings, convergence records, seeds, design IDs, and preprocessing hashes. A scientifically broader result requires these controlled comparisons; a larger sample count alone is insufficient.

## Report beyond a single average error

Retain the shared valid-pixel, Euclidean three-pixel edge and nonuniform-coordinate gradient metrics. Add per-design p95 errors and error-versus-geometry summaries, fixed-example edge zooms, peak magnitude/location error, and separate false-positive and missed high-field regions using thresholds set before held-out evaluation. Report inference batch size, device, sampler steps, ensemble size, measured wall time, and solver-comparison time.

Physics diagnostics must respect interfaces: a jump in normal E across permittivity is not automatically a violation. Use consistent material-aware flux conditions before claiming a residual is physical error. Single-model sampling spread is distinct from target noise, numerical discretization, and uncertainty across independent training seeds.

For a design use case, first define an observable, such as a valid-region high-field hotspot or a terminal quantity supported by the solver outputs. The course's peak-normalized targets cannot establish absolute field magnitude or voltage-dependent breakdown risk. Any inverse-design candidate must be rechecked with a converged solver before making a physical claim.
