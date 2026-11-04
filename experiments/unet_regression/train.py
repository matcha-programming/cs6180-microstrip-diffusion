"""Experiment 4R: deterministic, position-conditioned U-Net regression."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from paired_model import PositionUNet, paired_metadata
from common import DEFAULT_DATA,setup,dump,reg,score_predictions,summarize,examples,latency


def predict(model,data,ids,mean,std,clamp=True):
    zeros=torch.zeros((len(ids),2,64,128),device=ids.device)
    t=torch.zeros(len(ids),device=ids.device,dtype=torch.long)
    field=model(zeros,t,data['x'][ids])*std+mean
    if clamp:field=field.clamp(-1,1)
    return field.masked_fill(data['mask'][ids,None],0)


@torch.inference_mode()
def validation(model,data,rows,mean,std,batch=16):
    model.eval();total=0;count=0
    for start in range(0,len(rows),batch):
        ids=torch.as_tensor(rows[start:start+batch],device=data['x'].device)
        valid=(~data['mask'][ids])[:,None];field=predict(model,data,ids,mean,std)
        total+=float(((field-data['u'][ids]).square()*valid).sum());count+=int(valid.sum())*2
    return total/count


@torch.inference_mode()
def predictions(model,data,rows,mean,std,batch=32,clamp=True):
    model.eval();result=[]
    for start in range(0,len(rows),batch):
        ids=torch.as_tensor(rows[start:start+batch],device=data['x'].device)
        result.append(predict(model,data,ids,mean,std,clamp).cpu().numpy())
    return np.concatenate(result)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default=DEFAULT_DATA);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=200);ap.add_argument('--max-train-hours',type=float,default=2)
    ap.add_argument('--base',type=int,default=36);ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--seed',type=int,default=6180);ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    device=setup(args.seed,args.device);args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'complete.json').exists():return
    arrays,data,splits,source_ids=reg.load_data(args.data,device)
    if args.smoke:splits={k:v[:2] for k,v in splits.items()};args.epochs=1;args.batch_size=2
    # Reuse the same training-only channel transform as paired direct diffusion.
    from diffusion import train_statistics
    mean,std=train_statistics(data['u'],data['mask'],splits['train'])
    # Training multiplies model outputs by std, so these constants must be normal tensors.
    mean,std=mean.clone(),std.clone()
    model=PositionUNet(base=args.base).to(device);ema=copy.deepcopy(model).eval().requires_grad_(False)
    parameters=sum(p.numel() for p in model.parameters())
    assert args.smoke or 5_000_000<=parameters<=20_000_000
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0)
    amp=device.type=='cuda' and torch.cuda.get_device_capability(device)[0]>=7
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update({'experiment':'4R','model':'deterministic PositionUNet','parameters':parameters,
                   'learning_rate':2e-4,'ema_decay':.999,'mixed_precision':amp,
                   'target_mean':mean.flatten().tolist(),'target_std':std.flatten().tolist(),
                   'normalization_fit':'training valid pixels only',
                   'loss':'masked MSE in original peak-normalized field units; output channel scaling matches paired diffusion',
                   'field_input':'two zero planes, no noisy field or target input','time_input':0,
                   'output_constraint':'same known [-1,1] component bound as paired DDIM; mask zero at evaluation',
                   **paired_metadata(device)})
    dump(args.output/'config.json',config);print('REGRESSION_CONFIG',json.dumps(config),flush=True)
    best=float('inf');best_epoch=0;stale=0;elapsed=0;start_epoch=1;stop_reason='epochs'
    if (args.output/'last.pt').exists():
        last=torch.load(args.output/'last.pt',map_location=device,weights_only=False)
        for k in ('base','seed','source_sha256','paired_source_sha256'):
            if config[k]!=last['config'][k]:raise ValueError(f'Resume mismatch: {k}')
        model.load_state_dict(last['model']);ema.load_state_dict(last['ema']);optimizer.load_state_dict(last['optimizer'])
        scaler.load_state_dict(last['scaler']);start_epoch=last['epoch']+1;best=last['best'];best_epoch=last['best_epoch'];stale=last['stale'];elapsed=last['elapsed']
        np.random.set_state(last['numpy_rng']);torch.set_rng_state(last['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in last['cuda_rng']])
    begin=time.perf_counter()
    with (args.output/'history.jsonl').open('a',buffering=1) as history:
        for epoch in range(start_epoch,args.epochs+1):
            if elapsed+time.perf_counter()-begin>=args.max_train_hours*3600:stop_reason='training_budget';break
            model.train();loss_sum=0;batches=0;order=np.random.permutation(splits['train']);started=time.perf_counter();budget=False
            for start in range(0,len(order),args.batch_size):
                ids=torch.as_tensor(order[start:start+args.batch_size],device=device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                    field=predict(model,data,ids,mean,std,clamp=False)
                    loss=reg.masked_mse(field.float(),data['u'][ids].float(),data['mask'][ids])
                if not torch.isfinite(loss):raise RuntimeError('Non-finite regression loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                scaler.step(optimizer);scaler.update()
                with torch.no_grad():
                    for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,.001)
                loss_sum+=loss.item();batches+=1
                if args.smoke:break
                if elapsed+time.perf_counter()-begin>=args.max_train_hours*3600:budget=True;break
            value=validation(ema,data,splits['val'],mean,std,args.batch_size)
            if value<best:
                best=value;best_epoch=epoch;stale=0
                torch.save({'model':ema.state_dict(),'config':config,'epoch':epoch,'validation_masked_mse':best},args.output/'best.pt')
            else:stale+=1
            reg.sync(device);row={'epoch':epoch,'train_masked_mse':loss_sum/batches,'validation_masked_mse':value,'seconds':time.perf_counter()-started}
            history.write(json.dumps(row)+'\n');print('REGRESSION_EPOCH',json.dumps(row),flush=True)
            torch.save({'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                        'config':config,'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,'elapsed':elapsed+time.perf_counter()-begin,
                        'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []},args.output/'last.pt')
            if stale>=30:stop_reason='early_stopping';break
            if budget:stop_reason='training_budget';break
    elapsed+=time.perf_counter()-begin
    checkpoint=torch.load(args.output/'best.pt',map_location=device,weights_only=False);ema.load_state_dict(checkpoint['model'])
    result={'experiment':'4R','smoke_test_only':args.smoke,'parameters':parameters,'best_epoch':best_epoch,
            'best_validation_masked_mse':best,'train_seconds':elapsed,'train_gpu_hours':elapsed/3600 if device.type=='cuda' else 0,
            'stop_reason':stop_reason,'unclipped_metrics':{},**paired_metadata(device)}
    for split in ('test','test_extrap'):
        pred=predictions(ema,data,splits[split],mean,std);records=score_predictions(pred,splits[split],arrays,source_ids)
        result[split]=summarize(records);dump(args.output/f'{split}_per_design.json',records)
        examples(args.output/f'{split}_examples.npz',pred,splits[split],arrays)
        raw=predictions(ema,data,splits[split],mean,std,clamp=False)
        result['unclipped_metrics'][split]=summarize(score_predictions(raw,splits[split],arrays,source_ids))
    ids=torch.as_tensor(splits['val'][:1],device=device)
    result['batch_one_prediction_seconds']=latency(lambda:predict(ema,data,ids,mean,std),device)
    dump(args.output/'metrics.json',result);dump(args.output/'complete.json',{'status':'complete','smoke_test_only':args.smoke})
    print('REGRESSION_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
