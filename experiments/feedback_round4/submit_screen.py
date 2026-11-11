"""Submit once, only after source checks and successful GPU preflight."""
import hashlib,json,subprocess
from pathlib import Path
root=Path.cwd(); journal=root/'screen-submission.json'
if journal.exists():raise RuntimeError('Submission already attempted; inspect Slurm and this journal')
manifest=json.loads((root/'microstrip-feedback-manifest.json').read_text())
for name,expected in manifest['files'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError(f'Changed source: {name}')
if not (root/'preflight-passed').exists():raise ValueError('Preflight did not finish')
preflight=(root/'preflight-job-id.txt').read_text().strip();capacity=(root/'capacity-job-id.txt').read_text().strip()
for p in (root/f'runs/smoke-{preflight}').glob('*/complete.json'):
 c=json.loads(p.read_text())
 if c['status']!='complete' or not c['smoke']:raise ValueError('Invalid smoke record')
if len(list((root/f'runs/smoke-{preflight}').glob('*/complete.json')))!=7:raise ValueError('Seven smoke records required')
cap=json.loads((root/f'slurm/capacity-{capacity}.log').read_text().strip().splitlines()[-1])
if cap['status']!='passed' or cap['batch_size']!=32:raise ValueError('Batch 32 check missing')
audit=json.loads((root/'runs/validation-audit/checks.json').read_text())
if audit['status']!='passed':raise ValueError('Validation audit failed')
state=dict(state='submitting',source_manifest_sha256=hashlib.sha256((root/'microstrip-feedback-manifest.json').read_bytes()).hexdigest(),preflight_job_id=preflight,capacity_job_id=capacity,capacity_check=cap,parallel_limit=2)
with journal.open('x') as f:json.dump(state,f,indent=2)
def save():journal.write_text(json.dumps(state,indent=2)+'\n')
def job(args):
 value=subprocess.check_output(['sbatch','--parsable',*args],text=True).strip().split(';')[0]
 if not value.isdigit():raise ValueError(value)
 return value
state['array_job_id']=job(['--array=0-6%2','experiments/feedback_round4/gpu.sbatch']);state['state']='array_submitted';save()
state['report_job_id']=job(['--dependency=afterany:'+state['array_job_id'],'experiments/feedback_round4/report.sbatch']);state['state']='submitted';save();print(json.dumps(state))
