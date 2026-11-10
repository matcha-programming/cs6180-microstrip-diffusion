"""Evaluate fixed R2-A/C/D checkpoints on both held-out splits, without training.

Run prepare on a scheduled CPU node, worker on a scheduled GPU node, and pack
on a scheduled CPU node. Original source and campaign files remain read-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

TASKS = ('R2-A-original', 'R2-A-paired', 'R2-C-lr-0.0001',
         'R2-C-lr-0.0002', 'R2-C-lr-0.0004', 'R2-D-min-snr-5')
SPLITS = ('test', 'test_extrap')
EXPECTED_COUNTS = {'test': 881, 'test_extrap': 979}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def create(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')


def historical_selection(run, campaign, sizes):
    """Freeze existing historical choices transparently, without model selection."""
    cfg = read(run / 'config.json')
    completion = read(run / 'complete.json')
    if cfg.get('smoke_test_only') or cfg.get('smoke') or completion.get('smoke_test_only') or completion.get('status') != 'complete':
        raise ValueError('Historical draw study is incomplete or synthetic')
    index = 0 if run.name == 'R2-A-original' else 1
    declared = campaign['draw_studies'][index]
    expected_name = ('R2-A-original', 'R2-A-paired')[index]
    if declared['name'] != run.name or run.name != expected_name or cfg['steps'] != declared['steps'] or cfg['split'] != 'val':
        raise ValueError('Historical checkpoint/sampler differs from predeclared R2-A task')
    checkpoint = Path(cfg['checkpoint'])
    if checkpoint.name != Path(declared['checkpoint']).name or digest(checkpoint) != cfg['checkpoint_sha256']:
        raise ValueError('Historical checkpoint changed after the validation draw study')
    return {
        'round': 2, 'selection_split': 'val', 'smoke_test_only': False,
        'selection_metric': 'predeclared historical checkpoint and sampler',
        'selection_basis': 'Historical checkpoint and DDIM steps fixed before the original R2-A validation study; no new choice among candidates.',
        'full_validation_confirmation': False,
        'validation_panel_count': len(cfg['row_indices']),
        'validation_rows': cfg['row_indices'],
        'checkpoint': str(checkpoint), 'checkpoint_sha256': cfg['checkpoint_sha256'],
        'steps': cfg['steps'], 'run': str(run.resolve()),
        'epoch': cfg.get('checkpoint_epoch'), 'backbone': cfg['backbone'],
        'data_sha256': cfg['data_sha256'], 'split_hashes': cfg['split_hashes'],
        'report_sizes': sizes, 'seed_start': 6180,
        'note': 'Additional descriptive held-out evaluation requested after the original campaign. This manifest does not claim a new full-validation model search.',
    }


def prepare(configuration, chooser=None):
    source = Path(configuration['source_root'])
    campaign = Path(configuration['source_campaign'])
    output = Path(configuration['output_root'])
    if (output / 'plan.json').exists():
        raise FileExistsError('Evaluation plan already exists; inspect it instead of preparing another')
    for name, expected in configuration['source_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError('Source differs from the reviewed code: ' + name)
    data = Path(configuration['data'])
    spec = read(source / 'experiments-round-2/campaign.json')
    if digest(data) != spec['dataset_sha256']:
        raise ValueError('Dataset checksum mismatch')
    if chooser is None:
        sys.path.insert(0, str(source / 'experiments-round-2'))
        from freeze_selection import choose
        chooser = choose
    cases = []
    for name in TASKS:
        run = campaign / name
        cfg = read(run / 'config.json')
        if cfg['data_sha256'] != spec['dataset_sha256']:
            raise ValueError('Task used a different dataset: ' + name)
        sizes = [1, 2, 3, 4, 8, 16] if name.startswith('R2-A-') else [1, 3]
        selection = historical_selection(run, spec, sizes) if name.startswith('R2-A-') else chooser([run], sizes)
        if selection['data_sha256'] != cfg['data_sha256'] or selection['split_hashes'] != cfg['split_hashes']:
            raise ValueError('Frozen selection differs from source task data: ' + name)
        selection.update(
            evaluation_purpose='Descriptive test/extrapolation comparison of all predeclared R2-A/C/D trials; no test-based retuning.',
            source_task=name, source_campaign=str(campaign.resolve()),
            plan_id=name[:4],
        )
        cases.append(dict(name=name, selection=selection))
    reference = read(campaign / 'selection.json')
    for split in SPLITS:
        directory = campaign / ('final-' + split)
        if read(directory / 'complete.json').get('status') != 'complete':
            raise ValueError('Existing R2-B reference is incomplete')
        metrics = read(directory / 'metrics.json')
        if metrics['checkpoint_sha256'] != reference['checkpoint_sha256']:
            raise ValueError('Existing R2-B result differs from its frozen reference')
        if any(metrics['sizes'][str(k)]['ensemble']['designs'] != EXPECTED_COUNTS[split] for k in (1, 3)):
            raise ValueError('Existing R2-B reference does not cover the full held-out split')
    # Persist no manifests until all candidates have passed the preflight checks.
    for case in cases:
        target = output / 'evaluations' / case['name'] / 'selection.json'
        create(target, case['selection'])
        case['selection_manifest'] = str(target.resolve())
        case['selection_sha256'] = digest(target)
    plan = {
        **configuration, 'cases': cases, 'common_report_sizes': [1, 3],
        'sampling_seed_start': 6180, 'splits': list(SPLITS),
        'hardware_request': 'Tesla V100-SXM2-32GB, FP32 sampling',
        'reference': {'source': str(campaign / 'selection.json'), 'selection': reference,
                      'note': 'Reuse existing R2-B one-/three-draw final scores; no training or R2-B reevaluation.'},
        'protocol_note': 'Sampler steps differ according to each fixed validation choice. R2-A also reports larger K. Timing comparisons require matching recorded GPU and protocol.',
    }
    create(output / 'plan.json', plan)
    print(json.dumps({'state': 'prepared', 'cases': [dict(name=c['name'], steps=c['selection']['steps'], report_sizes=c['selection']['report_sizes']) for c in cases]}), flush=True)
    return plan


def commands_for(plan, index):
    case = plan['cases'][index]
    selected = case['selection']
    for split in SPLITS:
        target = Path(plan['output_root']) / 'evaluations' / case['name'] / split
        command = [sys.executable, str(Path(plan['source_root']) / 'experiments-round-2/draw_study.py'),
                   '--data', plan['data'], '--checkpoint', selected['checkpoint'],
                   '--steps', str(selected['steps']), '--sizes', *map(str, selected['report_sizes']),
                   '--seed-start', '6180', '--split', split, '--limit', '0', '--confirm-test',
                   '--selection', case['selection_manifest'], '--output', str(target)]
        yield split, target, command


def worker(output, index):
    plan = read(output / 'plan.json')
    case = plan['cases'][index]
    selection = Path(case['selection_manifest'])
    if digest(selection) != case['selection_sha256']:
        raise ValueError('Selection manifest changed after freezing')
    case_root = output / 'evaluations' / case['name']
    for split, target, command in commands_for(plan, index):
        complete = target / 'complete.json'
        if complete.exists() and read(complete).get('status') == 'complete':
            print('ALREADY_COMPLETE', case['name'], split, flush=True)
            continue
        if target.exists():
            raise FileExistsError('Partial output needs explicit recovery: ' + str(target))
        create(case_root / (split + '-launch.json'), {'command': command, 'selection_sha256': case['selection_sha256'], 'slurm_job_id': os.getenv('SLURM_JOB_ID')})
        with (case_root / (split + '.log')).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        metrics = read(target / 'metrics.json')
        if any(row['ensemble']['designs'] != EXPECTED_COUNTS[split] for row in metrics['sizes'].values()):
            raise ValueError('Evaluation did not cover the full split')
        print('HELDOUT_COMPLETE', case['name'], split, flush=True)
    create(case_root / 'evaluation-complete.json', {'status': 'complete', 'selection_sha256': case['selection_sha256']})


def pack(output):
    plan = read(output / 'plan.json')
    for case in plan['cases']:
        root = output / 'evaluations' / case['name']
        if read(root / 'evaluation-complete.json')['status'] != 'complete':
            raise ValueError('Incomplete case: ' + case['name'])
        for split in SPLITS:
            if read(root / split / 'complete.json')['status'] != 'complete':
                raise ValueError('Incomplete split: ' + case['name'] + '/' + split)
    create(output / 'evaluation-campaign-complete.json', {'status': 'complete', 'tasks': list(TASKS), 'splits': list(SPLITS), 'plan_sha256': digest(output / 'plan.json')})
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output / 'sha256sums.txt').write_text(''.join(f'{digest(p)}  {p.relative_to(output)}\n' for p in files))
    archive_path = output.with_suffix('.zip')
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(p for p in output.rglob('*') if p.is_file()):
            archive.write(file, file.relative_to(output))
    print('HELDOUT_CAMPAIGN_COMPLETE', str(archive_path), digest(archive_path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'worker', 'pack'))
    parser.add_argument('--configuration', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--task-index', type=int, default=int(os.getenv('SLURM_ARRAY_TASK_ID', '0')))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(read(args.configuration))
    elif args.action == 'worker':
        worker(args.output_root, args.task_index)
    else:
        pack(args.output_root)


if __name__ == '__main__':
    main()
