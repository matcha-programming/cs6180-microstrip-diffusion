"""Build the comparison only from completed, full-data experiment outputs."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(Path(path).read_text())


def validate(path):
    value=read(path)
    if value.get('smoke_test_only'):
        raise ValueError(f'Refusing smoke-test results: {path}')
    return value


def figures(folder):
    for split in ('test','test_extrap'):
        data=np.load(folder/f'{split}_examples.npz') if (folder/f'{split}_examples.npz').exists() else None
        if data is None:continue
        truth=data['truth'];prediction=data['prediction'];mask=data['mask']
        y=data['y_over_h'];z=data['z_over_h'];n=len(truth)
        for zoom in (False,True):
            fig,axes=plt.subplots(n,6,figsize=(18,3*n),squeeze=False,layout='constrained')
            for i in range(n):
                for channel in range(2):
                    values=(truth[i,channel],prediction[i,channel],prediction[i,channel]-truth[i,channel])
                    vmax=max(float(np.abs(values[0]).max()),float(np.abs(values[1]).max()),1e-9)
                    for col,value in enumerate(values):
                        ax=axes[i,channel*3+col]
                        bound=vmax if col<2 else max(float(np.abs(value).max()),1e-9)
                        zz=z if z.ndim==1 else z[i]
                        mesh=ax.pcolormesh(y[i],zz,np.ma.masked_where(mask[i],value),cmap='RdBu_r',vmin=-bound,vmax=bound,shading='auto',rasterized=True)
                        ax.set_title(f"row {int(data['row_indices'][i])} · {'Ey' if channel==0 else 'Ez'} · {('truth','prediction','error')[col]}",fontsize=9)
                        ax.set_xlabel('y/h');ax.set_ylabel('z/h')
                        if zoom:
                            rr,cc=np.where(mask[i])
                            if len(rr):
                                ax.set_xlim(y[i,max(0,cc.min()-8)],y[i,min(len(y[i])-1,cc.max()+8)])
                                ax.set_ylim(zz[max(0,rr.min()-8)],zz[min(len(zz)-1,rr.max()+8)])
                        fig.colorbar(mesh,ax=ax,shrink=.7)
            fig.savefig(folder/f"{split}-{'edge' if zoom else 'fields'}.png",dpi=140)
            plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--plain',type=Path,required=True)
    ap.add_argument('--gp',type=Path,required=True)
    args=ap.parse_args();run=args.run
    if read(run/'status.json')['status']!='complete':raise ValueError('Pipeline has not completed')
    f=validate(run/'fourier/metrics.json');plain=validate(args.plain/'metrics.json');gp=validate(args.gp/'metrics.json')
    diff={mode:validate(run/mode/'metrics.json') for mode in ('direct','residual')}
    configurations={mode:read(run/mode/'config.json') for mode in diff}
    for split,expected in [('test',881),('test_extrap',979)]:
        for name,value in [('PCA+GP',gp),('plain',plain),('Fourier',f)]:
            if value[split]['designs']!=expected:raise ValueError(f'{name} {split} count mismatch')
        for mode,value in diff.items():
            for setting in value['step_counts'].values():
                if setting[split]['ensemble']['designs']!=expected:raise ValueError(f'{mode} {split} count mismatch')
    lines=['# Experiments 3–6: Fourier features and conditional diffusion','',
           f"HPC job: `{f['slurm_job_id']}`. GPU: {f['gpu']}. All figures use the first four designs in each fixed split.",'',
           '## Accuracy','',
           'Entries are mean per-design errors; lower is better. Diffusion single-sample entries average the three seed-specific means. Ensemble entries average three fields before scoring.','',
           '| Model | Split | Relative L2 | Edge L2 | Gradient error | Masked MSE |',
           '|---|---|---:|---:|---:|---:|']
    for name,value in [('PCA32 + GP',gp),('Plain DeepONet',plain),('Fourier DeepONet',f)]:
        for split in ('test','test_extrap'):
            v=value[split]
            lines.append(f"| {name} | {split} | {v['relative_l2']['mean']:.5f} | {v['edge_relative_l2']['mean']:.5f} | {v['gradient_relative_l2']['mean']:.5f} | {v['masked_mse']['mean']:.6g} |")
    for mode,value in diff.items():
        for steps,settings in value['step_counts'].items():
            primary=' (selected)' if int(steps)==value['selected_steps'] else ''
            for split in ('test','test_extrap'):
                for sample in ('single_sample_mean','ensemble'):
                    v=settings[split][sample]
                    if sample=='ensemble':v={k:x['mean'] for k,x in v.items() if isinstance(x,dict)}
                    label='one draw' if sample=='single_sample_mean' else 'three-draw mean'
                    lines.append(f"| {mode.title()} DDIM-{steps}{primary}, {label} | {split} | {v['relative_l2']:.5f} | {v['edge_relative_l2']:.5f} | {v['gradient_relative_l2']:.5f} | {v['masked_mse']:.6g} |")
    comparator=plain if f['frozen_regressor_kind']=='plain' else f
    lines+=['','## Selection and compute','',
            f"Fourier scale {f['selected_scale']} was selected by validation masked MSE. The frozen regression comparator is the {f['frozen_regressor_kind']} model. Fourier search training and validation took {f['total_search_train_seconds']/60:.2f} minutes across all three scales.",'',
            '| Model | Parameters | Training + validation hours | Best epoch | Stop reason | Selected DDIM steps | One draw, model-only seconds | Three draws, model-only seconds |',
            '|---|---:|---:|---:|---|---:|---:|---:|']
    for mode,value in diff.items():
        cfg=configurations[mode];sample=value['step_counts'][str(value['selected_steps'])]
        lines.append(f"| {mode.title()} diffusion | {cfg['parameters']:,} | {value['train_seconds']/3600:.3f} | {value['best_epoch']} | {value['stop_reason']} | {value['selected_steps']} | {sample['batch_one_prediction_seconds']:.6f} | {sample['ensemble_prediction_seconds']:.6f} |")
    lines+=['',f"The frozen regressor takes {comparator['batch_one_prediction_seconds']:.6f} seconds per design in its recorded timing. Residual sampling times above use cached conditioning; add one regression prediction for a new design. Training times include validation and checkpoint selection, but exclude data loading, frozen-prediction caching, and final DDIM evaluation. GPU hours describe the allocated GPU, not active-kernel time. PCA+GP timing used a CPU and should not be read as a hardware-matched speed comparison.",'',
            'A two-hour training-budget stop is a budgeted comparison, not evidence that diffusion has converged. Learning curves and validation scores are retained for this distinction.','',
            '## Artifacts','',
            '- `fourier/`: candidate histories, selected Fourier checkpoint, frozen comparator, metrics, and fixed examples.',
            '- `direct/` and `residual/`: configuration, history, EMA checkpoint, validation sampler selection, and 20/50-step results.',
            '- Each DDIM folder records all three seeds, per-design metrics, ensemble metrics, sample variance, and example draws.',
            '- Optimizer recovery files remain on HPC and are excluded from the publication bundle.','',
            'The metric definitions and sampling constraints are documented in the experiment source README. Raw metrics retain seed variation, medians, and standard deviations.']
    (run/'README.md').write_text('\n'.join(lines)+'\n')
    figures(run/'fourier')
    for mode in diff:
        for folder in sorted((run/mode).glob('ddim_*')):
            for split in ('test','test_extrap'):
                # Reuse the ensemble arrays without duplicating them in the saved report bundle.
                source=folder/f'{split}_ensemble_examples.npz';target=folder/f'{split}_examples.npz'
                if source.exists():
                    target.write_bytes(source.read_bytes())
            figures(folder)
            for target in folder.glob('*_examples.npz'):
                if target.name in ('test_examples.npz','test_extrap_examples.npz'):target.unlink()
    files=[p for p in run.rglob('*') if p.is_file() and p.name not in ('last.pt','sha256sums.txt') and '.mpl-cache' not in p.parts]
    (run/'sha256sums.txt').write_text(''.join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(run)}\n" for p in sorted(files)))


if __name__=='__main__':main()
