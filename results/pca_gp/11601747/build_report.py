from pathlib import Path
import json
import hashlib
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
deep = root / 'results/deeponet_plain/11597463'
gp = json.loads((out / 'metrics.json').read_text())
staff = json.loads((out / 'staff-results.json').read_text())
dn = json.loads((deep / 'metrics.json').read_text())
cfg = json.loads((out / 'config.json').read_text())
assert 'COMPLETED' in (out / 'slurm.txt').read_text() and '0:0' in (out / 'slurm.txt').read_text()
assert 'BASELINE_COMPLETE' in (out / 'training.log').read_text()
assert staff['k'] == 32 and staff['n_train'] == 7048
for split in ['val','test','test_extrap']:
    assert math.isclose(gp[split]['relative_l2']['mean'], staff['pca_gp'][split]['rel_l2']['mean'], rel_tol=1e-10)
    rows = json.loads((out / f'{split}_per_design.json').read_text())
    assert len(rows) == gp[split]['designs']
    for metric, stats in gp[split].items():
        if metric == 'designs': continue
        assert math.isclose(np.mean([r[metric] for r in rows]), stats['mean'], rel_tol=1e-10)

fig,axs=plt.subplots(4,2,figsize=(10,11),constrained_layout=True)
for row,(split,j) in enumerate([('test',0),('test',1),('test_extrap',0),('test_extrap',1)]):
    a=np.load(out/f'{split}_examples.npz')
    for channel in range(2):
        ax=axs[row,channel]
        im=ax.pcolormesh(a['y_over_h'][j],a['z_over_h'][j],np.ma.array(a['truth'][j,channel],mask=a['mask'][j]),shading='nearest',cmap='RdBu_r',vmin=-1,vmax=1)
        ax.set(title=f'{split} row {a["row_indices"][j]} · '+['Ey','Ez'][channel],xlabel='y/h',ylabel='z/h',facecolor='.8')
fig.colorbar(im,ax=axs,label='Field component / design peak magnitude',shrink=.8)
fig.suptitle('Fixed target examples; gray pixels are excluded by the mask')
fig.savefig(out/'dataset-examples.png',dpi=140);plt.close(fig)

for zoom in [False,True]:
    fig=plt.figure(figsize=(18,11),constrained_layout=True)
    gs=fig.add_gridspec(4,7,width_ratios=[1,1,1,.055,1,1,.055])
    for si,split in enumerate(['test','test_extrap']):
        a=np.load(out/f'{split}_examples.npz'); b=np.load(deep/f'{split}_examples.npz')
        assert np.array_equal(a['row_indices'],b['row_indices'])
        assert np.array_equal(a['truth'],b['truth'])
        y,z=a['y_over_h'][0],a['z_over_h'][0]; mask=a['mask'][0]
        error_limit=max(np.abs((a['prediction'][0]-a['truth'][0])[:,~mask]).max(),np.abs((b['prediction'][0]-a['truth'][0])[:,~mask]).max())
        for channel in range(2):
            row=si*2+channel
            fields=[a['truth'][0,channel],a['prediction'][0,channel],b['prediction'][0,channel],np.abs(a['prediction'][0,channel]-a['truth'][0,channel]),np.abs(b['prediction'][0,channel]-a['truth'][0,channel])]
            for j,data in enumerate(fields):
                ax=fig.add_subplot(gs[row,[0,1,2,4,5][j]])
                im=ax.pcolormesh(y,z,np.ma.array(data,mask=mask),shading='nearest',cmap='RdBu_r' if j<3 else 'magma',vmin=-1 if j<3 else 0,vmax=1 if j<3 else error_limit)
                label=(f'{split} row {a["row_indices"][0]} · '+['Ey','Ez'][channel]+'\nz/h') if j==0 else 'z/h'
                ax.set(title=['Truth','PCA-32 + GP','Plain DeepONet','PCA-GP |error|','DeepONet |error|'][j],xlabel='y/h',ylabel=label,facecolor='.8')
                if zoom:
                    iz,iy=np.where(mask)
                    ax.set_xlim(max(y[0],y[iy.min()]-.5),min(y[-1],y[iy.max()]+.5))
                    ax.set_ylim(max(z[0],z[iz.min()]-.5),min(z[-1],z[iz.max()]+.5))
                if j==2:fig.colorbar(im,cax=fig.add_subplot(gs[row,3]),extend='both')
                if j==4:fig.colorbar(im,cax=fig.add_subplot(gs[row,6]))
    fig.suptitle(('Trace-edge detail' if zoom else 'Full fields')+' · first fixed design from each test split; shared color scales')
    fig.savefig(out/('edge-comparison.png' if zoom else 'field-comparison.png'),dpi=140);plt.close(fig)

lines=['# Experiment 1: PCA-32 + Gaussian-process baseline','',f'Explorer job **11601747** completed successfully on `{cfg["hostname"]}` with eight CPUs and no GPU. The staff baseline uses 32 PCA components and 32 independent Gaussian processes. All fitting uses the fixed training split only.','','## Comparison with Experiment 2','','Metrics below use the same implementation for both models. Values are means over designs; lower is better.','','| Split | Model | Relative L2 | Edge relative L2 | Gradient relative L2 | Masked MSE |','| --- | --- | ---: | ---: | ---: | ---: |']
for split,label in [('test','Test (881)'),('test_extrap','Extrapolation (979)')]:
    for model,m in [('Staff PCA-32 + GP',gp),('Plain DeepONet',dn)]:
        a=m[split];lines.append(f'| {label} | {model} | {a["relative_l2"]["mean"]:.6f} | {a["edge_relative_l2"]["mean"]:.6f} | {a["gradient_relative_l2"]["mean"]:.6f} | {a["masked_mse"]["mean"]:.9f} |')
lines+=['',f'The staff fitting and evaluation pipeline took **{staff["wall_s"]:.1f} seconds**, excluding the wrapper\'s initial data audit. This is a single run: staff seed 0 and DeepONet seed 6180. There is no estimate of variation across training seeds. Standard deviations in the JSON files describe variation across designs.','','## Timing','','| Model | Hardware | Warmed-up batch-one prediction |','| --- | --- | ---: |',f'| PCA-32 + GP | 8 CPUs on {cfg["hostname"]} | {gp["batch_one_prediction_seconds"]*1000:.3f} ms |',f'| Plain DeepONet | Tesla P100-PCIE-12GB | {dn["batch_one_prediction_seconds"]*1000:.3f} ms |','','Both timings exclude file I/O. PCA-GP timing includes GP prediction and PCA reconstruction; DeepONet timing covers its model forward pass. Hardware differs, so these timings are not a controlled comparison of algorithm speed. The staff batch timings are retained separately in `staff-results.json`.','','## Dataset inspection','','`data-inspection.json` records shapes, split counts and hashes, finite-value checks, zero masked targets, and coordinate ordering. Fixed split sizes are 7,048 training, 881 validation, 881 test, and 979 extrapolation. The mask excludes 6–72 pixels per design. Field magnitude peaks are normalized to approximately 1. The horizontal grid varies with design; derivatives use each design\'s actual coordinates.','','![Fixed target examples](dataset-examples.png)','','## Field and trace-edge comparison','','The figures use the first design in each fixed test split, selected before inspecting model errors. Field scales and error scales are shared between models. Gray pixels are masked.','','![Full-field comparison](field-comparison.png)','','![Trace-edge comparison](edge-comparison.png)','','## Metric conventions','','`metrics.json` matches DeepONet run 11597463: valid pixels only; a Euclidean three-pixel edge band; forward derivatives divided by the actual y/h and z/h spacing; valid endpoints for derivative stencils; and per-design aggregation.','','The unmodified staff script uses Chebyshev edge bands and pixel-index gradients. Its original results, PCA reconstruction floors at 8/16/32/64 components, fitted kernel descriptions, and kNN sanity check are preserved in `staff-results.json`. Do not mix those edge or gradient values with the comparison table. PCA “floor” scores are oracle reconstruction diagnostics obtained by projecting known target fields. With masking and derivative metrics, they need not be strict lower bounds.','','Some optimized kernels reached the staff search bounds. Those settings were retained; the convergence warnings are recorded in `training.log`.','','## Reproduction and artifacts','','See [Experiment 1 source](../../../experiments/pca_gp/) and [Experiment 2 results](../../deeponet_plain/11597463/). `config.json` records software versions, source hashes, seed, and fit scope. `training.log` and `slurm.txt` record execution and scheduler status. Per-design JSON files and the first four predictions, targets, masks, and grids are retained for each evaluation split. `sha256sums.txt` covers every artifact in this directory.','','Run `python build_report.py` from this directory to regenerate the report and figures. NumPy and Matplotlib are required; the Experiment 2 result files must remain at their repository paths.','','The staff source is preserved byte for byte, with SHA-256 `4cf231c86d4270a0a3a6662655e094eabef5d5a409982b183920026a83ae6014`. The wrapper adds observation and reporting without changing PCA, GP fitting, or predictions.']
(out/'README.md').write_text('\n'.join(lines)+'\n')
(out/'sha256sums.txt').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n' for f in sorted(out.iterdir()) if f.is_file() and f.name!='sha256sums.txt'))
print('Validated baseline records and generated comparison report.')
