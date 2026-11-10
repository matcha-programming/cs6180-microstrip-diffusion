"""Resume only the existing incomplete plateau schedule; never submit jobs."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/screen/R3-S4-warmup-plateau-seed6180'


def main():
    import torch
    if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run inside the scheduled V100 GPU allocation')
    if 'V100' not in torch.cuda.get_device_name():
        raise RuntimeError('Expected the original V100 GPU class')
    active = subprocess.check_output(['squeue', '-h', '-u', 'he.zit', '-o', '%i|%j'], text=True)
    for line in active.splitlines():
        job, name = line.split('|', 1)
        if job == os.environ['SLURM_JOB_ID']:
            continue
        if job.startswith('11716516') or name == 'r3-plateau-resume':
            raise RuntimeError('Original training or another recovery is active: ' + line)
    # Never unlink the lock file: all workers must lock the same inode.
    with (RUN / '.resume.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads((ROOT / 'source-manifest.json').read_text())
        for name, expected in manifest['files'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Frozen source mismatch: ' + name)
        checkpoint = torch.load(RUN / 'last.pt', map_location='cpu', weights_only=False)
        cfg = checkpoint['config']
        if cfg['round'] != 3 or cfg['schedule'] != 'warmup-plateau' or cfg['seed'] != 6180 or cfg['max_updates'] != 220500:
            raise RuntimeError('Unexpected recovery checkpoint protocol')
        updates = checkpoint['progress']['updates']
        if not 0 < updates < 220500 or checkpoint['scheduler']['completed'] != updates:
            raise RuntimeError('Completed or inconsistent recovery checkpoint; inspect before acting')
        complete = json.loads((RUN / 'complete.json').read_text())
        if complete['status'] != 'incomplete' or complete['updates'] != updates:
            raise RuntimeError('Completion/checkpoint mismatch; inspect possible crash or another recovery')
        # Extend only elapsed-time allowance. Horizon and schedule remain identical.
        hours = checkpoint['elapsed'] / 3600 + 5
        del checkpoint
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        backup = ROOT / 'recovery' / ('plateau-' + stamp + '-' + os.environ['SLURM_JOB_ID'])
        backup.mkdir(parents=True, exist_ok=False)
        for name in ('config.json', 'history.jsonl', 'validation.json', 'complete.json', 'last.pt', 'best-field.pt', 'best-noise.pt'):
            shutil.copy2(RUN / name, backup / name)
        command = [sys.executable, '-u', str(ROOT / 'experiments-round-3/train.py'),
                   '--data', cfg['data'], '--output', str(RUN), '--resume', str(RUN / 'last.pt'),
                   '--phase', cfg['phase'], '--schedule', cfg['schedule'], '--max-train-hours', str(hours)]
        for key in ('backbone', 'base', 'batch_size', 'epochs', 'lr', 'min_lr', 'warmup_epochs', 'seed', 'field_interval', 'validation_count', 'field_steps'):
            command += ['--' + key.replace('_', '-'), str(cfg[key])]
        if cfg['amp']:
            command.append('--amp')
        receipt = {'state': 'starting', 'job_id': os.environ['SLURM_JOB_ID'], 'source_updates': updates,
                   'target_updates': 220500, 'backup': str(backup), 'command': command}
        (backup / 'resume-launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
        subprocess.run(command, cwd=ROOT, check=True)
        result = json.loads((RUN / 'validation.json').read_text())
        print(json.dumps({'resume_finished': True, 'equal_update_budget_complete': result['equal_update_budget_complete'],
                          'updates': result['updates_in_round3'], 'stop_reason': result['stop_reason']}), flush=True)
        if not result['equal_update_budget_complete']:
            raise SystemExit('Segment budget ended: inspect scheduler and checkpoint before any further continuation')


if __name__ == '__main__':
    main()
