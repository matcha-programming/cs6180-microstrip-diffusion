#!/usr/bin/env python3
"""Rebuild the results comparison from saved metrics, without loading checkpoints.

Run from any directory: python results/update_summary.py
Use --watch to refresh after local result files change; --no-xlsx needs only Python.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from model_architectures import architecture_for, COUNT_CONVENTION, SOURCE_HASHES

RESULTS = Path(__file__).resolve().parent
PROJECT_PLAN = RESULTS.parent / 'docs/project-plan.md'
PLAN_EXPERIMENTS = {
    '1': 'Fixed-split audit and PCA-32 + GP baseline',
    '2': 'Plain DeepONet',
    '3': 'Fourier-feature DeepONet',
    '4': 'Direct conditional DDPM, original and position-paired variants',
    '4R': 'Deterministic U-Net control paired with experiment 4',
    '5': 'DDIM-20/50 sampling evaluation shared by diffusion methods',
    '6': 'Residual diffusion around a frozen DeepONet prediction',
    'R2-A': 'Draw-count and variance studies on existing direct checkpoints',
    'R2-B': 'Longer training by continuing the paired direct model',
    'R2-C': 'Fresh paired-model learning-rate screen',
    'R2-D': 'Optional Min-SNR loss-weighting ablation',
}
PLAN_ORDER = {plan_id: index for index, plan_id in enumerate(PLAN_EXPERIMENTS)}
METRICS = ('relative_l2', 'edge_relative_l2', 'gradient_relative_l2', 'masked_mse')
DEFINITIONS = [
    'Errors are means over designs on valid pixels; lower is better. Relative L2, edge and gradient errors are dimensionless fractions displayed as percentages.',
    'Test contains 881 designs; test_extrap contains 979. Validation panels of 128 designs and full validation of 881 designs are labeled separately.',
    'Single-draw average scores each draw separately, then averages scores. Ensemble scores the averaged fields. These are different quantities.',
    'Historical diffusion single-draw averages use three draws. Round-two seed-6180 rows use one draw; three-draw averages are also retained for comparison.',
    'Mean design MSE averages per-design MSE. Global pixel MSE weights all valid elements and is the round-two selection criterion. Do not interchange these aggregations.',
    'Timing is batch-one model/sampler time, excluding I/O and scoring. Hardware and caching conventions differ. Ensemble times are serial estimates; residual sampling additionally needs the frozen regressor.',
    'Training time includes validation/checkpointing where recorded; round-two final full validation is additional. Continuation reports both new and historical-plus-new time. PCA time is the fit/evaluation wrapper; Fourier time is the three-scale search.',
    'Each trained candidate has one training seed. Sampling draws and across-design variation do not measure variation across training seeds. Smoke runs are excluded.',
]


class Collector:
    def __init__(self, root):
        self.root = Path(root)
        self.scores = []
        self.methods = {}
        self.architectures = {}
        self.sources = {}
        self.warnings = []
        if PROJECT_PLAN.exists():
            self.sources['../docs/project-plan.md'] = hashlib.sha256(PROJECT_PLAN.read_bytes()).hexdigest()
        self.sources['model_architectures.py'] = hashlib.sha256((RESULTS / 'model_architectures.py').read_bytes()).hexdigest()

    def rel(self, path):
        return str(Path(path).relative_to(self.root))

    def read(self, path, default=None):
        if not Path(path).exists():
            return {} if default is None else default
        raw = Path(path).read_bytes()
        self.sources[self.rel(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    def archived_config(self, directory, archive_name, preferred=None):
        path = directory / archive_name
        if not path.exists():
            return {}, None
        with zipfile.ZipFile(path) as archive:
            names = sorted(n for n in archive.namelist() if n.endswith('/config.json'))
            if preferred:
                names = [n for n in names if preferred in n]
            if not names:
                return {}, None
            raw = archive.read(names[0])
            source = self.rel(path) + '!' + names[0]
            self.sources[source] = hashlib.sha256(raw).hexdigest()
            return json.loads(raw), source

    @staticmethod
    def smoke(*objects):
        return any(obj.get('smoke_test_only') or obj.get('smoke') for obj in objects)

    def describe(self, path, data, config):
        rel = self.rel(path)
        if 'pca_gp/' in rel:
            return 'PCA-32 + GP', 'PCA compression and independent ARD RBF GPs', 'GP marginal likelihood', 'Training-only PCA/GP fit'
        if 'deeponet_plain/' in rel:
            return 'Plain DeepONet', 'Design branch and plain coordinate trunk', 'Masked field MSE', 'Validation field MSE'
        if path.name == 'fourier-metrics.json':
            return 'Fourier DeepONet', 'Design branch and Fourier coordinate trunk', 'Masked field MSE', 'Validation MSE across Fourier scales'
        if path.name == 'regression-metrics.json':
            return '4R U-Net', 'Paired U-Net, fixed index planes, zero field/time inputs', 'Masked field MSE', 'Validation field MSE, EMA'
        if path.name == 'metrics-residual.json':
            return 'Residual diffusion', 'U-Net with frozen Fourier regression conditioning', 'Masked epsilon MSE', 'Validation noise MSE, then DDIM steps'
        if 'unet_regression/' in rel:
            return 'Paired diffusion', 'Paired U-Net and fixed row/column index planes', 'Masked epsilon MSE', 'Validation noise MSE, then DDIM steps'
        if path.parent.name.startswith('R2-'):
            name = path.parent.name
            loss = 'Min-SNR epsilon MSE (gamma ' + str(config.get('snr_gamma')) + ')' if config.get('loss_weighting') == 'min-snr' else 'Masked epsilon MSE'
            return name, 'Paired U-Net and fixed row/column index planes', loss, 'Field/noise checkpoints; full validation DDIM confirmation'
        if 'step_counts' in data:
            return 'Original direct diffusion', 'Design-conditioned U-Net, no fixed position planes', 'Masked epsilon MSE', 'Validation noise MSE, then DDIM steps'
        return str(config.get('model', config.get('experiment', path.parent.name))), str(config.get('conditioning', config.get('backbone', 'See configuration'))), str(config.get('training_objective', config.get('loss', 'See configuration'))), str(config.get('selection', 'See configuration'))

    def plan_identity(self, path, data, config):
        """Map method ownership separately from the shared sampling experiment."""
        rel = self.rel(path)
        explicit = config.get('plan_id', data.get('plan_id'))
        if explicit is not None:
            return str(explicit), str(config.get('related_plan_ids', ''))
        round2 = re.match(r'^(R2-[A-D])(?:-|$)', path.parent.name)
        if round2:
            return round2.group(1), '4; 5'
        if 'pca_gp/' in rel:
            return '1', ''
        if 'deeponet_plain/' in rel:
            return '2', ''
        if path.name == 'fourier-metrics.json':
            return '3', ''
        if path.name == 'regression-metrics.json':
            return '4R', '4'
        if path.name == 'metrics-residual.json':
            return '6', '5'
        if 'step_counts' in data:
            return '4', '5'
        self.warnings.append(f'{rel}: project-plan experiment ID is unmapped; add plan_id to configuration or extend Collector.plan_identity')
        return 'Unmapped', ''

    def method(self, path, data, config=None, config_source=None):
        config = config or self.read(path.parent / 'config.json')
        name, architecture, loss, selection = self.describe(path, data, config)
        plan_id, related_plan_ids = self.plan_identity(path, data, config)
        key = self.rel(path)
        total = data.get('historical_plus_round2_train_seconds', data.get('train_seconds', data.get('total_search_train_seconds', data.get('total_wrapper_seconds'))))
        new = data.get('new_train_seconds', total)
        checkpoints = data.get('checkpoints', {})
        epochs = ', '.join(f"{k}: {v['epoch']}" for k, v in checkpoints.items()) if checkpoints else data.get('best_epoch')
        self.methods[key] = {
            'key': key, 'method': name, 'run': self.rel(path.parent),
            'plan_id': plan_id, 'related_plan_ids': related_plan_ids,
            'origin_run': config.get('output'),
            'checkpoint_hashes': sorted(set([v['sha256'] for v in data.get('checkpoints', {}).values()]
                                           + ([config['checkpoint_sha256']] if config.get('checkpoint_sha256') else []))),
            'architecture': architecture, 'loss': loss,
            'learning_rate': config.get('learning_rate', config.get('lr')),
            'parameters': data.get('parameters', config.get('parameters')),
            'precision': ('AMP' if config.get('mixed_precision', config.get('amp')) else 'FP32') if config.get('device') == 'cuda' or config.get('mixed_precision') is not None else 'n.a.',
            'new_train_seconds': new, 'total_train_seconds': total,
            'best_epoch': epochs, 'stop_reason': data.get('stop_reason', 'not recorded'),
            'hardware': data.get('gpu', config.get('gpu', config.get('device_name', '8 CPUs' if 'pca_gp/' in key else 'not recorded'))),
            'seed': config.get('seed'), 'selection': selection,
            'source': self.rel(path), 'config_source': config_source or (self.rel(path.parent / 'config.json') if (path.parent / 'config.json').exists() else None),
        }
        architecture_config = dict(config)
        if plan_id == 'R2-A' and 'base' not in architecture_config:
            checkpoint = config.get('checkpoint', '')
            if '/results/' in checkpoint:
                historical_path = self.root / checkpoint.split('/results/', 1)[1]
                archive_name = {'best-direct.pt': 'direct-results.zip', 'best-diffusion.pt': 'diffusion-results.zip'}.get(historical_path.name)
                if archive_name:
                    historical, _ = self.archived_config(historical_path.parent, archive_name)
                    architecture_config = {**historical, **config}
        architecture = architecture_for(plan_id, architecture_config)
        fields = ('learned_modules', 'layer_breakdown', 'input_shape', 'output_shape', 'layer_shapes')
        self.methods[key].update({field: architecture[field] for field in fields})
        self.methods[key]['architecture_profile'] = architecture['profile']
        self.methods[key]['architecture_source'] = '; '.join('../' + p for p in architecture['sources']) or None
        if self.methods[key]['parameters'] is None and architecture_config.get('parameters') is not None:
            self.methods[key]['parameters'] = architecture_config['parameters']
        if architecture['profile'] != 'Unavailable':
            profile = {field: architecture[field] for field in ('profile', 'learned_modules', 'layer_breakdown', 'layer_shapes', 'calculated_parameters', 'layers', 'sources')}
            if profile['profile'] in self.architectures:
                profile['sources'] = sorted(set(profile['sources'] + self.architectures[profile['profile']]['sources']))
            self.architectures[profile['profile']] = profile
            for source in architecture['sources']:
                self.sources['../' + source] = hashlib.sha256((RESULTS.parent / source).read_bytes()).hexdigest()
        return key

    def selected_method(self, selection, campaign):
        """Resolve separately saved evaluations to their original training/study row."""
        source_run = selection['run']
        candidates = [m['key'] for m in self.methods.values() if m.get('origin_run') == source_run]
        if not candidates:
            candidates = [m['key'] for m in self.methods.values()
                          if Path(m['run']).name == Path(source_run).name
                          and selection['checkpoint_sha256'] in m.get('checkpoint_hashes', [])]
        if len(candidates) > 1:
            raise ValueError('Ambiguous source method for held-out selection: ' + source_run)
        if candidates:
            expected_hashes = self.methods[candidates[0]].get('checkpoint_hashes', [])
            if expected_hashes and selection['checkpoint_sha256'] not in expected_hashes:
                raise ValueError('Frozen checkpoint does not belong to its source method: ' + source_run)
            return candidates[0]
        path = campaign / Path(source_run).name / 'validation.json'
        if path.exists():
            key = self.rel(path)
            if key not in self.methods:
                self.training(path, self.read(path))
            if key in self.methods:
                return key
        raise ValueError('Import the source training/draw-study results before its held-out evaluation: ' + source_run)

    def score(self, key, split, aggregate, report, draws, steps, source, selected=True, **extra):
        values = {}
        for metric in METRICS:
            raw = aggregate.get(metric)
            value = raw.get('mean') if isinstance(raw, dict) else raw
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{source}: missing or invalid {metric}')
            values[metric] = value
        record = dict(method_key=key, method=self.methods[key]['method'], run=self.methods[key]['run'], split=split,
                      plan_id=self.methods[key]['plan_id'], related_plan_ids=self.methods[key]['related_plan_ids'],
                      report=report, draws=draws, steps=steps, source=source,
                      selected=selected, designs=aggregate.get('designs'), **values, **extra)
        record['id'] = '|'.join(str(record.get(k, '')) for k in ('source', 'split', 'report', 'draws', 'steps', 'checkpoint'))
        self.scores.append(record)

    def deterministic(self, path, data):
        config = self.read(path.parent / 'config.json')
        config_source = None
        if path.name == 'regression-metrics.json':
            config, config_source = self.archived_config(path.parent, 'regression-results.zip')
        elif path.name == 'fourier-metrics.json':
            scale = format(data['selected_scale'], 'g')
            config, config_source = self.archived_config(path.parent, f'fourier-{path.parent.name}.zip', f'/scale_{scale}/')
        if self.smoke(data, config):
            return
        key = self.method(path, data, config, config_source)
        if path.name == 'fourier-metrics.json':
            candidate = next(c for c in data['candidates'] if c['scale'] == data['selected_scale'])
            self.methods[key]['best_epoch'] = candidate.get('best_epoch')
        if 'best_validation_masked_mse' in data:
            self.methods[key]['selection'] += f"; recorded validation MSE {data['best_validation_masked_mse']:.8g}"
        for split in ('val', 'test', 'test_extrap'):
            if split in data:
                self.score(key, split, data[split], 'deterministic', 1, None, self.rel(path),
                           prediction_seconds=data.get('batch_one_prediction_seconds'))

    def diffusion(self, path, data):
        archive_name = {'metrics-direct.json': 'direct-results.zip', 'metrics-residual.json': 'residual-results.zip', 'diffusion-metrics.json': 'diffusion-results.zip'}.get(path.name)
        config, config_source = self.archived_config(path.parent, archive_name) if archive_name else ({}, None)
        if self.smoke(data, config):
            return
        key = self.method(path, data, config, config_source)
        count = len(data['sample_seeds'])
        for step, entry in data['step_counts'].items():
            for split in ('test', 'test_extrap'):
                if split not in entry:
                    continue
                for report, metric_key, timing in [('single-draw average', 'single_sample_mean', 'batch_one_prediction_seconds'), ('ensemble', 'ensemble', 'ensemble_prediction_seconds')]:
                    aggregate = dict(entry[split][metric_key])
                    aggregate.setdefault('designs', entry[split]['ensemble'].get('designs'))
                    self.score(key, split, aggregate, report, count, int(step), self.rel(path),
                               selected=int(step) == data['selected_steps'],
                               prediction_seconds=entry.get(timing), timing_estimated=report == 'ensemble')

    def training(self, path, data):
        config = self.read(path.parent / 'config.json')
        if self.smoke(data, config):
            return
        if not (path.parent / 'complete.json').exists():
            self.warnings.append(f'{self.rel(path)}: incomplete training output skipped')
            return
        key = self.method(path, data, config)
        selection = self.read(path.parent.parent / 'selection.json')
        for checkpoint, candidate in data['checkpoints'].items():
            for steps, aggregate in candidate['steps'].items():
                self.score(key, 'val', aggregate, 'single-draw average', 3, int(steps), self.rel(path),
                           selected=selection.get('checkpoint_sha256') == candidate['sha256'] and selection.get('steps') == int(steps),
                           checkpoint=checkpoint, epoch=candidate['epoch'],
                           masked_mse_global=aggregate.get('masked_mse_global'))

    def draws(self, path, data):
        config = self.read(path.parent / 'config.json')
        if self.smoke(data, config):
            return
        if not (path.parent / 'complete.json').exists():
            self.warnings.append(f'{self.rel(path)}: incomplete sampling output skipped')
            return
        split = data['split']
        campaign = path.parent.parent
        selection = self.read(campaign / 'selection.json') if split in ('test', 'test_extrap') else {}
        if selection:
            if data['checkpoint_sha256'] != selection['checkpoint_sha256'] or config.get('steps') != selection['steps']:
                raise ValueError(f'{path}: held-out checkpoint or steps disagree with frozen selection')
            if config.get('selection_manifest_sha256') and hashlib.sha256((campaign / 'selection.json').read_bytes()).hexdigest() != config['selection_manifest_sha256']:
                raise ValueError(f'{path}: frozen manifest differs from the evaluated manifest')
            if sorted(map(int, data['sizes'])) != selection['report_sizes']:
                raise ValueError(f'{path}: reported draw counts differ from the frozen manifest')
            key = self.selected_method(selection, campaign)
        else:
            key = self.method(path, data, config)
            self.methods[key].update(method=path.parent.name, architecture=f"{config.get('backbone', 'recorded')} historical checkpoint; no training", loss='Sampling only', selection='Predeclared validation draw study', best_epoch=config.get('checkpoint_epoch'), stop_reason='sampling only')
        for size, entry in data['sizes'].items():
            k = int(size)
            self.score(key, split, entry['ensemble'], 'single draw (seed 6180)' if k == 1 else 'ensemble', k, config['steps'], self.rel(path),
                       prediction_seconds=entry['estimated_serial_ensemble_seconds'], timing_estimated=k > 1,
                       variance=entry.get('mean_design_variance'), checkpoint=Path(selection['checkpoint']).stem if selection else 'historical')
            if k > 1:
                aggregate = dict(entry['single_draw_mean'], designs=entry['ensemble']['designs'])
                self.score(key, split, aggregate, 'single-draw average', k, config['steps'], self.rel(path),
                           prediction_seconds=entry['batch_one_single_draw_seconds'], variance=entry.get('mean_design_variance'), checkpoint=Path(selection['checkpoint']).stem if selection else 'historical')

    def collect(self):
        for path in sorted(self.root.rglob('validation.json')):
            data = self.read(path)
            if 'checkpoints' in data:
                self.training(path, data)
        metric_files = [(path, self.read(path)) for path in sorted(self.root.rglob('*metrics*.json'))]
        # Resolve historical R2-A studies before separately imported held-out reports.
        metric_files.sort(key=lambda item: (isinstance(item[1], dict) and item[1].get('split') in ('test', 'test_extrap'), str(item[0])))
        for path, data in metric_files:
            if not isinstance(data, dict):
                self.warnings.append(f'{self.rel(path)}: unsupported metric schema')
            elif self.smoke(data):
                continue
            elif 'sizes' in data and 'split' in data:
                self.draws(path, data)
            elif 'step_counts' in data:
                self.diffusion(path, data)
            elif any(split in data for split in ('test', 'test_extrap', 'val')):
                self.deterministic(path, data)
            else:
                self.warnings.append(f'{self.rel(path)}: unsupported metric schema')
        ids = [r['id'] for r in self.scores]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate score IDs')
        heldout = [s for s in self.scores if s['split'] in ('test', 'test_extrap')]
        for row in heldout:
            if row['designs'] != {'test': 881, 'test_extrap': 979}[row['split']]:
                raise ValueError(f"Unexpected held-out design count: {row['source']}")
        def plan_order(method):
            return PLAN_ORDER.get(method['plan_id'], len(PLAN_ORDER)), method['run'], method['method']
        self.scores.sort(key=lambda row: plan_order(self.methods[row['method_key']]))
        for method in self.methods.values():
            rows = [r for r in self.scores if r['method_key'] == method['key']]
            validation_sizes = sorted({r['designs'] for r in rows if r['split'] == 'val'})
            scope = [('Full validation' if n == 881 else 'Validation panel') + f' ({n})' for n in validation_sizes]
            heldout_splits = {r['split'] for r in rows if r['split'] in ('test', 'test_extrap')}
            if heldout_splits == {'test', 'test_extrap'}:
                scope.append('Test + extrapolation')
            elif heldout_splits == {'test'}:
                scope.append('Test')
            elif heldout_splits == {'test_extrap'}:
                scope.append('Extrapolation')
            method['results_available'] = '; '.join(scope) or 'No scored outputs'
        fingerprint = hashlib.sha256(json.dumps(self.sources, sort_keys=True).encode()).hexdigest()
        return dict(schema_version=3, definitions=DEFINITIONS + [COUNT_CONVENTION], source_fingerprint=fingerprint,
                    plan_experiments=PLAN_EXPERIMENTS, plan_order=list(PLAN_EXPERIMENTS),
                    architectures=list(self.architectures.values()),
                    scores=self.scores, methods=sorted(self.methods.values(), key=plan_order),
                    source_sha256=self.sources, warnings=self.warnings)


def comparison_rows(data):
    grouped = {}
    for score in data['scores']:
        if score['split'] not in ('test', 'test_extrap') or not score['selected'] or score['draws'] not in (1, 3):
            continue
        key = (score['method_key'], score['report'], score['draws'], score['steps'])
        grouped.setdefault(key, {})[score['split']] = score
    methods = {m['key']: m for m in data['methods']}
    rows = []
    for key, splits in grouped.items():
        row = dict(method=methods[key[0]]['method'], plan_id=methods[key[0]]['plan_id'], related_plan_ids=methods[key[0]]['related_plan_ids'], run=methods[key[0]]['run'], report=key[1], draws=key[2], steps=key[3], hardware=methods[key[0]]['hardware'])
        for split in ('test', 'test_extrap'):
            score = splits.get(split, {})
            row.update({split + '_' + metric: score.get(metric) for metric in METRICS})
            row[split + '_source'] = score.get('source')
        row['prediction_ms'] = (splits.get('test') or splits.get('test_extrap')).get('prediction_seconds')
        if row['prediction_ms'] is not None:
            row['prediction_ms'] *= 1000
        rows.append(row)
    return rows


def pct(value):
    return 'n.a.' if value is None else f'{100 * value:.2f}%'


def scientific(value):
    return 'n.a.' if value is None else f'{value:.7g}'


def markdown(data, comparisons):
    lines = ['# Results and methods comparison', '', '<!-- Generated by update_summary.py; edit source results or the collector, then refresh. -->', '',
             'Refresh after adding or importing experiment outputs:', '', '```sh', 'python results/update_summary.py', '```', '',
             'Use `--watch` to refresh when local result files change. Use `--no-xlsx` for a Python-only Markdown/CSV/JSON refresh. The workbook requires `@oai/artifact-tool`; its Node runtime is detected automatically or supplied with `ARTIFACT_NODE_MODULES`.', '',
             '[Excel workbook](comparison.xlsx) · [All held-out scores as CSV](comparison.csv) · [Complete structured summary](comparison.json)', '',
             '## Experiment IDs from the project plan', '',
             'IDs follow [project-plan.md](../docs/project-plan.md). Experiment 5 is the DDIM sampling stage shared by diffusion rows, rather than a separately trained model. Round-two tasks retain their R2 IDs; related IDs identify the original direct-diffusion and sampling experiments.', '',
             '| Plan ID | Experiment |', '| --- | --- |',
             *[f'| {plan_id} | {description} |' for plan_id, description in PLAN_EXPERIMENTS.items()], '',
             '## Held-out comparison', '', 'Rows use the checkpoint/sampler fixed from validation or predeclared historical settings. Common one- and three-draw reports are shown here; all evaluated samplers and larger R2-A draw counts remain in the CSV/JSON. Additional R2-A/C/D held-out scores are descriptive comparisons of fixed candidates and do not replace the original global campaign selection. Missing values are shown as n.a.', '',
             '| Plan ID | Method | Run/task | Report | DDIM steps | Test L2 | Extrap L2 | Test edge | Extrap edge | Prediction ms | Hardware | Sources |',
             '| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    if data.get('workbook_status') == 'not refreshed':
        lines[4:4] = ['**The workbook was not refreshed in this generation (`--no-xlsx`).** Rerun without that option to synchronize it with these tables.', '']
    for row in comparisons:
        sources = sorted(set(filter(None, (row['test_source'], row['test_extrap_source']))))
        links = ', '.join(f'[{i + 1}]({p})' for i, p in enumerate(sources))
        report = row['report'] + (f" (K={row['draws']})" if row['draws'] > 1 else '')
        timing = 'n.a.' if row['prediction_ms'] is None else f"{row['prediction_ms']:.3f}"
        lines.append(f"| {row['plan_id']} | {row['method']} | {row['run']} | {report} | {row['steps'] or 'n.a.'} | {pct(row['test_relative_l2'])} | {pct(row['test_extrap_relative_l2'])} | {pct(row['test_edge_relative_l2'])} | {pct(row['test_extrap_edge_relative_l2'])} | {timing} | {row['hardware']} | {links} |")
    lines += ['', '## Gradient error and masked MSE', '', '| Plan ID | Method | Report | Test gradient | Extrap gradient | Test mean design MSE | Extrap mean design MSE |', '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in comparisons:
        lines.append(f"| {r['plan_id']} | {r['method']} | {r['report']} (K={r['draws']}) | {pct(r['test_gradient_relative_l2'])} | {pct(r['test_extrap_gradient_relative_l2'])} | {scientific(r['test_masked_mse'])} | {scientific(r['test_extrap_masked_mse'])} |")
    lines += ['', '## Round-two validation trials', '', 'Each row is the lowest full-validation global pixel MSE among the saved field/noise checkpoints and DDIM-20/50 for that trial. The frozen winner flag comes from the saved selection manifest, independently of test scores.', '', '| Plan ID | Run/task | Checkpoint | Epoch | DDIM steps | Validation L2 | Global pixel MSE | Frozen winner | Source |', '| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |']
    trials = {}
    for s in data['scores']:
        if s['split'] == 'val' and s.get('masked_mse_global') is not None:
            previous = trials.get(s['method_key'])
            if previous is None or (s['masked_mse_global'], s['steps']) < (previous['masked_mse_global'], previous['steps']):
                trials[s['method_key']] = s
    for s in trials.values():
        lines.append(f"| {s['plan_id']} | {s['run']} | {s['checkpoint']} | {s['epoch']} | {s['steps']} | {pct(s['relative_l2'])} | {s['masked_mse_global']:.7g} | {'yes' if s['selected'] else 'no'} | [metrics]({s['source']}) |")
    lines += ['', '## Validation draw studies', '', 'The full per-K individual-score and ensemble-score records are in the workbook and JSON. These studies use a 128-design validation panel.', '', '| Plan ID | Study | Draws K | Ensemble L2 | Mean individual-draw L2 | Mean design variance | Source |', '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for s in data['scores']:
        if s['split'] != 'val' or s.get('checkpoint') != 'historical' or s['report'] == 'single-draw average':
            continue
        single = next((q for q in data['scores'] if q['source'] == s['source'] and q['draws'] == s['draws'] and q['report'] == 'single-draw average'), s)
        variance = 'n.a.' if s.get('variance') is None else f"{s['variance']:.7g}"
        lines.append(f"| {s['plan_id']} | {s['method']} | {s['draws']} | {pct(s['relative_l2'])} | {pct(single['relative_l2'])} | {variance} | [metrics]({s['source']}) |")
    lines += ['', '## Methods and training', '', '| Plan ID | Method / task | Related plan IDs | Architecture and conditioning | Loss | LR | Parameters | New training h | Total training h | Best checkpoint epoch | Hardware | Source |', '| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |']
    for m in data['methods']:
        new = 'n.a.' if m['new_train_seconds'] is None else f"{m['new_train_seconds']/3600:.3f}"
        total = 'n.a.' if m['total_train_seconds'] is None else f"{m['total_train_seconds']/3600:.3f}"
        lines.append(f"| {m['plan_id']} | {m['method']} | {m['related_plan_ids'] or 'n.a.'} | {m['architecture']} | {m['loss']} | {m['learning_rate'] if m['learning_rate'] is not None else 'n.a.'} | {m['parameters'] if m['parameters'] is not None else 'n.a.'} | {new} | {total} | {m['best_epoch'] if m['best_epoch'] is not None else 'n.a.'} | {m['hardware']} | [metrics]({m['source']}) |")
    lines += ['', '## Architecture and result coverage', '',
              'Architecture shapes are derived from verified source and saved configurations. Parameter totals are checked against the saved configuration. `B` is batch size and `P` is queried spatial points; full fields contain 64 x 128 points. The Methods workbook tab also contains indexed input, output, and weight shapes for each architecture profile.', '',
              '| Plan ID | Method / task | Results available | Architecture profile | Learned modules | Layer breakdown | Input shape | Output shape |',
              '| --- | --- | --- | --- | ---: | --- | --- | --- |']
    for m in data['methods']:
        lines.append(f"| {m['plan_id']} | {m['method']} | {m['results_available']} | {m['architecture_profile']} | {m['learned_modules'] if m['learned_modules'] is not None else 'n.a.'} | {m['layer_breakdown']} | {m['input_shape'] or 'n.a.'} | {m['output_shape'] or 'n.a.'} |")
    lines += ['', '| Architecture profile | Stage shapes | Source |', '| --- | --- | --- |']
    for profile in data['architectures']:
        links = ', '.join(f'[{Path(p).name}](../{p})' for p in profile['sources'])
        lines.append(f"| {profile['profile']} | {profile['layer_shapes']} | {links} |")
    lines += ['', '## Interpretation and metric conventions', '', *['- ' + text for text in data['definitions']], '', '## Adding experiments', '',
              'Place each completed run in a separate directory under `results/`, keeping its metrics and configuration together. The collector discovers deterministic test/validation metrics, historical diffusion step-count metrics, round-two training validation files, and draw-study metrics recursively. For round-two campaigns, keep the task folders and selection.json together. Repeated runs remain separate rows.', '',
              'New method families may need a description in `Collector.describe`. Match their experiment ID to project-plan.md by adding `plan_id` to the run configuration or extending `Collector.plan_identity`. New metric schemas must get an explicit parser; unsupported metric files are listed below rather than silently scored. Source files stay authoritative. The generator replaces this README, comparison.csv, comparison.json, and comparison.xlsx. It never changes experiment outputs or selects configurations using held-out scores.', '',
              'The watcher observes local files only. Copy remote outputs into `results/` before refreshing. No remote polling, scheduled task, or background process is installed.', '',
              '## Import coverage', '', f"Parsed {len(data['scores'])} score records from {len(data['methods'])} method/run records.", '',
              *(data['warnings'] or ['No unsupported metric files were found.']), '', f"Source fingerprint: `{data['source_fingerprint']}`", '']
    return '\n'.join(lines)


def xlsx_runtime():
    candidates = [Path(os.environ['ARTIFACT_NODE_MODULES'])] if os.environ.get('ARTIFACT_NODE_MODULES') else []
    candidates.append(Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules')
    for modules in candidates:
        if (modules / '@oai/artifact-tool').is_dir():
            bundled_node = modules.parent / 'bin/node'
            node = str(bundled_node) if bundled_node.exists() else shutil.which('node')
            if node:
                return node, modules
    raise RuntimeError('Workbook refresh requires @oai/artifact-tool. Set ARTIFACT_NODE_MODULES or pass --no-xlsx for Markdown/CSV/JSON only.')


def write_outputs(root, output, no_xlsx=False, preview=None):
    data = Collector(root).collect()
    data['comparison'] = comparison_rows(data)
    data['workbook_status'] = 'not refreshed' if no_xlsx else 'refreshed'
    output.mkdir(parents=True, exist_ok=True)
    if not no_xlsx:
        node, modules = xlsx_runtime()
        with tempfile.TemporaryDirectory(prefix='microstrip-summary-') as temp:
            temp = Path(temp)
            (temp / 'node_modules').symlink_to(modules, target_is_directory=True)
            shutil.copy2(RESULTS / 'build_summary.mjs', temp / 'build_summary.mjs')
            payload = temp / 'data.json'
            payload.write_text(json.dumps(data, indent=2) + '\n')
            command = [node, str(temp / 'build_summary.mjs'), str(payload), str(output / 'comparison.xlsx')]
            if preview:
                command += [str(preview)]
            subprocess.run(command, check=True)
    (output / 'comparison.json').write_text(json.dumps(data, indent=2) + '\n')
    (output / 'README.md').write_text(markdown(data, data['comparison']))
    heldout = [r for r in data['scores'] if r['split'] in ('test', 'test_extrap')]
    fields = sorted(set().union(*(r.keys() for r in heldout)))
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(heldout)
    print(f"Refreshed {len(data['methods'])} methods/runs, {len(data['scores'])} scores, {len(data['comparison'])} held-out comparison rows in {output}")
    return data


def signature(root):
    files = [(str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
             for p in root.rglob('*') if p.suffix in ('.json', '.zip') and p.name != 'comparison.json']
    if PROJECT_PLAN.exists():
        files.append(('../docs/project-plan.md', PROJECT_PLAN.stat().st_size, PROJECT_PLAN.stat().st_mtime_ns))
    for source in [RESULTS / 'model_architectures.py', *(RESULTS.parent / p for p in SOURCE_HASHES)]:
        if source.exists():
            files.append((str(source), source.stat().st_size, source.stat().st_mtime_ns))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=RESULTS)
    parser.add_argument('--output', type=Path, default=RESULTS)
    parser.add_argument('--no-xlsx', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--preview-dir', type=Path)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    previous = None
    while True:
        current = signature(args.root)
        if current != previous:
            try:
                write_outputs(args.root, args.output, args.no_xlsx, args.preview_dir)
                previous = current
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
                if not args.watch:
                    raise
                print(f'Refresh deferred; will retry: {error}', file=sys.stderr)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
