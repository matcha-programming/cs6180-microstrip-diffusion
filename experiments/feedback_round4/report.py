"""Summarize completed screen runs without ranking incomplete budgets."""
import argparse,csv,json
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 rows=[];missing=[]
 for p in sorted(a.root.iterdir()):
  if not p.is_dir():continue
  if not (p/'complete.json').exists():missing.append(p.name);continue
  c=json.loads((p/'complete.json').read_text())
  if c['status']!='complete' or c.get('smoke'):missing.append(p.name);continue
  cfg=json.loads((p/'config.json').read_text());v=json.loads((p/'validation.json').read_text())
  for steps,score in v['steps'].items():rows.append(dict(run=p.name,method=cfg['method'],loss=cfg['loss'],base=cfg['base'],batch_size=cfg['batch_size'],epochs=cfg['epochs'],updates=c['updates'],parameters=cfg['parameters'],best_epoch=v['epoch'],steps=int(steps),checkpoint_sha256=v['checkpoint_sha256'],**score))
 if rows:
  with (a.output/'validation.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 primary=sorted([r for r in rows if r['steps']==(1 if r['method']=='regression' else 20)],key=lambda r:r['masked_mse_global'])
 (a.output/'summary.json').write_text(json.dumps(dict(status='complete' if len(primary)==7 and not missing else 'partial',complete_primary_rows=len(primary),incomplete=missing,primary=primary,scope='100-epoch single-training-seed validation screen; not full-budget confirmation; no held-out scoring'),indent=2)+'\n')
if __name__=='__main__':main()
