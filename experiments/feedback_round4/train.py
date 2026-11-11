"""Fresh, equal-example-budget ablations. Held-out targets are never loaded."""
import argparse, copy, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from core import s, ROOT, masked_loss, training_pair, evaluate
import sys
sys.path.insert(0,str(ROOT/'experiments-round-3'))
from schedules import Schedule, ScheduleSpec


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--task',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--data',default=s.DEFAULT_DATA)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args(); campaign=json.loads(args.config.read_text()); cfg=campaign['training'][args.task].copy()
    cfg.update(campaign['common']); cfg['smoke']=args.smoke
    out=args.output/cfg['name']; out.mkdir(parents=True,exist_ok=False)
    device=s.setup(cfg['seed'],'cuda'); data=s.FieldData(args.data)
    if args.smoke:
        cfg.update(epochs=2,batch_size=2,field_interval=1,validation_count=2,steps=[2],sampling_seeds=[6180])
        data.splits={k:v[:4] for k,v in data.splits.items()}
    ma,st=data.statistics(); mean,std=s.tensors(ma,st,device)
    panel=s.validation_rows(data,cfg['validation_count'])
    model=s.make_model('paired',cfg['base']).to(device); ema=copy.deepcopy(model).eval().requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=0)
    spe=math.ceil(len(data.splits['train'])/cfg['batch_size']); horizon=spe*cfg['epochs']
    scheduler=Schedule(ScheduleSpec('warmup-cosine',horizon,min(5*spe,horizon-2),cfg['lr'],1e-6))
    cfg.update(target_mean=ma.tolist(),target_std=st.tolist(),parameters=sum(p.numel() for p in model.parameters()),
      split_hashes=data.split_hashes(),data_sha256=s.sha(args.data),max_updates=horizon,
      source_sha256={str(p.relative_to(ROOT)):s.sha(p) for p in Path(__file__).parent.glob('*.py')},
      dependencies=s.provenance(device),validation_rows=panel.tolist(),
      selection='fixed validation panel global masked field MSE at 20 NFE; regression 1 NFE',
      comparison_budget='100 full passes over training split; batch-size variant has fewer optimizer updates',
      flow_path='(1-t)*Gaussian + t*standardized_field; velocity target=field-Gaussian; Euler sampler',
      ema_policy='decay=0.999**(examples/16), equal smoothing per example',
      schedule_sha256=s.sha(ROOT/'experiments-round-3/schedules.py'),
      slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    if cfg['data_sha256']!=campaign['dataset_sha256']:
        raise ValueError('Unexpected dataset bytes')
    s.dump(out/'config.json',cfg); alpha=s.legacy.schedule(device); updates=0; best=float('inf'); begin=time.monotonic()
    def save(epoch):
        s.save_checkpoint(out/'last.pt',dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
          config=cfg,epoch=epoch,updates=updates,scheduler=scheduler.state_dict(),numpy_rng=np.random.get_state(),
          torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),elapsed=time.monotonic()-begin))
    with (out/'history.jsonl').open('w',buffering=1) as hist:
        for epoch in range(1,cfg['epochs']+1):
            model.train(); order=np.random.permutation(data.splits['train']); loss_total=0.; items=0
            for start in range(0,len(order),cfg['batch_size']):
                b=data.batch(order[start:start+cfg['batch_size']],device); valid=(~b['mask'])[:,None]
                target=((b['u']-mean)/std)*valid
                inp,t,truth=training_pair(target,b['mask'],cfg['method'],alpha)
                opt.zero_grad(set_to_none=True); scheduler.apply(opt)
                loss=masked_loss(model(inp,t,b['x']),truth,b['mask'],cfg['loss'])
                if not torch.isfinite(loss): raise ValueError('Non-finite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); scheduler.advance(); updates+=1
                with torch.no_grad():
                    for a,p in zip(ema.parameters(),model.parameters()): a.lerp_(p,1-.999**(len(t)/16))
                loss_total+=loss.item()*len(t); items+=len(t)
            row=dict(epoch=epoch,updates=updates,train_objective=loss_total/items,lr_next=scheduler.next_lr,
                     elapsed=time.monotonic()-begin)
            if epoch%cfg['field_interval']==0 or epoch==cfg['epochs']:
                score,_=evaluate(ema,data,panel,mean,std,device,cfg['method'],2 if args.smoke else 20,batch_size=16)
                row['field']=score
                if score['masked_mse_global']<best:
                    best=score['masked_mse_global']; s.save_checkpoint(out/'best-field.pt',dict(model=ema.state_dict(),config=cfg,epoch=epoch,updates=updates))
            hist.write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True); save(epoch)
            if time.monotonic()-begin>cfg['max_hours']*3600: break
    complete=updates==horizon
    if not complete:
        s.dump(out/'complete.json',dict(status='incomplete',updates=updates,target_updates=horizon)); return
    chosen=torch.load(out/'best-field.pt',map_location='cpu',weights_only=False); ema.load_state_dict(chosen['model'])
    summary={}; all_records={}
    steps_list=[1] if cfg['method']=='regression' else cfg['steps']
    seeds=[6180] if cfg['method']=='regression' else cfg['sampling_seeds']
    for steps in steps_list:
        score,recs=evaluate(ema,data,data.splits['val'],mean,std,device,cfg['method'],steps,seeds,cfg['batch_size'])
        summary[str(steps)]=score; all_records[str(steps)]=recs
    s.dump(out/'validation.json',dict(split='val',checkpoint_sha256=s.sha(out/'best-field.pt'),epoch=chosen['epoch'],steps=summary))
    s.dump(out/'per-design.json',all_records)
    s.dump(out/'complete.json',dict(status='complete',smoke=args.smoke,updates=updates,target_updates=horizon,
      elapsed_seconds=time.monotonic()-begin))

if __name__=='__main__':main()
