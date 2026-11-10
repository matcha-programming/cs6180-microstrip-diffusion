"""One-shot local monitor/importer for an already submitted held-out campaign.

Only scheduler status and the compact, compute-generated summary log are read
through the established SSH session. Full per-design artifacts stay on HPC.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tempfile
import time
import zlib

REPO = Path(__file__).resolve().parents[2]


def save_status(path, state, **fields):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'state': state, 'updated_utc': datetime.now(timezone.utc).isoformat(), **fields}, indent=2) + '\n')
    temporary.replace(path)


def monitor(journal, status):
    jobs = journal['jobs']
    if not all(str(job).isdigit() for job in jobs.values()):
        raise ValueError('Expected numeric scheduler IDs')
    summary_log = shlex.quote(journal['summary_log'])
    script = f'''while :; do
if [ -f {summary_log} ]; then
  case "$(tail -n 1 {summary_log})" in SUMMARY_FILE_COUNT*) printf 'SUMMARY_READY\\n'; exit 0;; esac
fi
task_state=$(sacct -X -n -j {jobs['summary']} --format=State | awk 'NF {{print $1; exit}}')
case "$task_state" in FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*) printf 'SUMMARY_JOB_FAILED %s\\n' "$task_state"; exit 1;; esac
printf 'STATUS_BEGIN\\n'
squeue -h -j {jobs['worker']} -o '%i %t %M %R'
printf 'STATUS_END\\n'
sleep 30
done'''
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
           '-o', 'ServerAliveCountMax=120', 'explorer', script]
    for attempt in range(5):
        save_status(status, 'monitoring', jobs=jobs, remote_root=journal['remote_root'])
        with subprocess.Popen(ssh, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as process:
            lines = []
            for line in process.stdout:
                text = line.rstrip()
                if text == 'STATUS_BEGIN':
                    lines = []
                elif text == 'STATUS_END':
                    progress = '; '.join(lines) or 'GPU tasks have left the queue; awaiting packaging/export'
                    print(progress, flush=True)
                    save_status(status, 'monitoring', jobs=jobs, progress=lines, remote_root=journal['remote_root'])
                elif text == 'SUMMARY_READY':
                    process.wait()
                    return
                elif text.startswith('SUMMARY_JOB_FAILED'):
                    raise RuntimeError(text)
                else:
                    lines.append(text)
            stderr = process.stderr.read()
            code = process.wait()
        if code != 255:
            raise RuntimeError(f'Monitor exited {code}: {stderr}')
        save_status(status, 'ssh_reconnect_retry', jobs=jobs, attempt=attempt + 1, error=stderr)
        time.sleep(30)
    raise RuntimeError('The established SSH connection was lost; reconnect Explorer and rerun the collector.')


def import_payload(payload, destination):
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing import: ' + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.heldout-import-', dir=destination.parent) as temporary:
        temporary = Path(temporary)
        for relative, entry in payload['files'].items():
            path = PurePosixPath(relative)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unexpected export path: ' + relative)
            raw = base64.b64decode(entry['base64'], validate=True)
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('Export checksum mismatch: ' + relative)
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        complete = json.loads((temporary / 'evaluation-campaign-complete.json').read_text())
        if complete['status'] != 'complete' or len(complete['tasks']) != 6:
            raise ValueError('Expected all six completed cases')
        for name in complete['tasks']:
            for split in ('test', 'test_extrap'):
                root = temporary / 'evaluations' / name / split
                if json.loads((root / 'complete.json').read_text())['status'] != 'complete':
                    raise ValueError('Incomplete exported split')
                metrics = json.loads((root / 'metrics.json').read_text())
                expected = 881 if split == 'test' else 979
                if any(metrics['sizes'][str(k)]['ensemble']['designs'] != expected for k in (1, 3)):
                    raise ValueError('Export does not cover the full split')
        (temporary / 'import-scope.json').write_text(json.dumps({
            'scope': payload['scope'], 'remote_root': payload['remote_root'],
            'full_archive': payload['archive'], 'imported_files': len(payload['files']),
            'verification': 'Every imported byte stream matched the checksum of the compute-generated export.',
        }, indent=2) + '\n')
        os.replace(temporary, destination)
    return destination


def collect(journal_path, no_xlsx=False):
    journal = json.loads(journal_path.read_text())
    status = journal_path.parent / 'collection-status.json'
    destination = REPO / 'results/round2/11638349' / ('heldout-' + journal_path.parent.name)
    try:
        if not destination.exists():
            monitor(journal, status)
            save_status(status, 'importing', jobs=journal['jobs'], destination=str(destination))
            result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'explorer',
                                     'cat ' + shlex.quote(journal['summary_log'])], text=True, capture_output=True, check=True, timeout=90)
            encoded = result.stdout.split('SUMMARY_EXPORT_BEGIN\n', 1)[1].split('\nSUMMARY_EXPORT_END', 1)[0]
            payload = json.loads(zlib.decompress(base64.b64decode(encoded)))
            import_payload(payload, destination)
        save_status(status, 'refreshing_comparison', jobs=journal['jobs'], destination=str(destination))
        command = [sys.executable, str(REPO / 'results/update_summary.py'), '--preview-dir', str(journal_path.parent / 'comparison-preview')]
        if no_xlsx:
            command.append('--no-xlsx')
        subprocess.run(command, cwd=REPO, check=True)
        save_status(status, 'complete', jobs=journal['jobs'], destination=str(destination),
                    workbook=str(REPO / 'results/comparison.xlsx'), full_archive=journal['remote_root'] + '.zip')
        print('HELDOUT_RESULTS_IMPORTED', str(destination), flush=True)
    except Exception as error:
        save_status(status, 'failed', jobs=journal['jobs'], error=str(error), destination=str(destination))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--no-xlsx', action='store_true')
    args = parser.parse_args()
    collect(args.journal.resolve(), args.no_xlsx)


if __name__ == '__main__':
    main()
