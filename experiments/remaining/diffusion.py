"""Experiments 4–6: masked DDPM training, DDIM evaluation, and residual diffusion."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from common import DEFAULT_DATA,dump,metadata,setup,reg,load_regressor,regression_predictions,score_predictions,summarize,examples,sha,latency
from models import ConditionalUNet,schedule,masked_noise_loss,ddim_update


@torch.inference_mode()
def train_statistics(target,mask,indices):
    total=torch.zeros(2,device=target.device,dtype=torch.float64);square=total.clone();count=0
    for start in range(0,len(indices),64):
        ids=torch.as_tensor(indices[start:start+64],device=target.device)
        valid=(~mask[ids])[:,None];a=target[ids].double()
        total+=(a*valid).sum((0,2,3));square+=(a.square()*valid).sum((0,2,3));count+=int(valid.sum())
    mean=total/count;std=(square/count-mean.square()).clamp_min(1e-12).sqrt()
    return mean.float()[None,:,None,None],std.float()[None,:,None,None]


@torch.inference_mode()
def validation_noise(model,target,data,rows,condition,alpha,batch=16):
    model.eval();g=torch.Generator(device=target.device).manual_seed(9137);total=0;count=0
    for start in range(0,len(rows),batch):
        ids=torch.as_tensor(rows[start:start+batch],device=target.device)
        t=torch.randint(len(alpha),(len(ids),),device=target.device,generator=g)
        valid=(~data['mask'][ids])[:,None]
        noise=torch.randn(target[ids].shape,device=target.device,generator=g)*valid
        a=alpha[t,None,None,None];noisy=(a.sqrt()*target[ids]+(1-a).sqrt()*noise)*valid
        pred=model(noisy,t,data['x'][ids],None if condition is None else condition[ids])
        total+=float(((pred-noise).square()*valid).sum());count+=int(valid.sum())*2
    return total/count


@torch.inference_mode()
def sample_batch(model,data,ids,condition,baseline,mean,std,alpha,steps,generator):
    mask=data['mask'][ids];valid=(~mask)[:,None]
    noisy=torch.randn((len(ids),2,64,128),device=ids.device,generator=generator)*valid
    grid=np.linspace(len(alpha)-1,0,steps,dtype=np.int64)
    base=0 if baseline is None else baseline[ids]
    def bounded_clean(clean):
        # Peak normalization gives a known component bound in field units, before standardization.
        field=(clean*std+mean+base).clamp(-1,1)
        return (field-base-mean)/std
    for i,t in enumerate(grid):
        times=torch.full((len(ids),),int(t),device=ids.device,dtype=torch.long)
        epsilon=model(noisy,times,data['x'][ids],None if condition is None else condition[ids])
        previous=alpha[int(grid[i+1])] if i+1<len(grid) else alpha.new_tensor(1.0)
        noisy=ddim_update(noisy,epsilon,alpha[int(t)],previous,mask,bounded_clean)
    return (noisy*std+mean+base).masked_fill(mask[:,None],0)


@torch.inference_mode()
def predict_split(model,data,rows,condition,baseline,mean,std,alpha,steps,seed,batch=32):
    model.eval();g=torch.Generator(device=data['x'].device).manual_seed(seed);pred=[]
    for start in range(0,len(rows),batch):
        ids=torch.as_tensor(rows[start:start+batch],device=data['x'].device)
        pred.append(sample_batch(model,data,ids,condition,baseline,mean,std,alpha,steps,g).cpu().numpy())
    return np.concatenate(pred)


def evaluate_sampling(args,model,data,arrays,splits,source_ids,condition,baseline,mean,std,alpha):
    result={'sampler':'DDIM','eta':0.0,'sample_seeds':[6180,6181,6182],
            'smoke_test_only':args.smoke,'model':args.mode,'step_counts':{}}
    counts=[2,3] if args.smoke else [20,50]
    # Freeze sampler choice on full validation fields before any test scoring.
    val_scores={}
    for steps in counts:
        pred=predict_split(model,data,splits['val'],condition,baseline,mean,std,alpha,steps,6180)
        val_scores[steps]=summarize(score_predictions(pred,splits['val'],arrays,source_ids))
    result['selected_steps']=min(counts,key=lambda s:val_scores[s]['relative_l2']['mean'])
    result['validation_selection']=val_scores
    dump(args.output/'sampling-selection.json',result)
    for steps in counts:
        output=args.output/f'ddim_{steps}';output.mkdir(exist_ok=True)
        settings={}
        for split in ['test','test_extrap']:
            predictions=[];summaries=[];begin=time.perf_counter()
            for seed in result['sample_seeds']:
                prediction=predict_split(model,data,splits[split],condition,baseline,mean,std,alpha,steps,seed)
                records=score_predictions(prediction,splits[split],arrays,source_ids)
                dump(output/f'{split}_seed_{seed}_per_design.json',records)
                examples(output/f'{split}_seed_{seed}_examples.npz',prediction,splits[split],arrays)
                predictions.append(prediction);summaries.append(summarize(records))
            stacked=np.stack(predictions);ensemble=stacked.mean(axis=0)
            ensemble_records=score_predictions(ensemble,splits[split],arrays,source_ids)
            dump(output/f'{split}_ensemble_per_design.json',ensemble_records)
            examples(output/f'{split}_ensemble_examples.npz',ensemble,splits[split],arrays)
            per_design_variance=stacked.var(axis=0).mean(axis=1)
            valid=~arrays['mask'][splits[split]]
            settings[split]={'single_sample_runs':summaries,
                'single_sample_mean':{k:float(np.mean([s[k]['mean'] for s in summaries]))
                                      for k in ['relative_l2','edge_relative_l2','gradient_relative_l2','masked_mse']},
                'single_sample_std_across_seeds':{k:float(np.std([s[k]['mean'] for s in summaries]))
                                      for k in ['relative_l2','edge_relative_l2','gradient_relative_l2','masked_mse']},
                'ensemble':summarize(ensemble_records),
                'valid_pixel_sample_variance':float(per_design_variance[valid].mean()),
                'evaluation_wall_seconds':time.perf_counter()-begin}
            # Fixed examples retain all three draws to inspect variation directly.
            np.savez_compressed(output/f'{split}_sample_variation.npz',row_indices=splits[split][:4],
                                sample_seeds=np.array(result['sample_seeds']),prediction=stacked[:,:4],
                                truth=arrays['u'][splits[split][:4]],mask=arrays['mask'][splits[split][:4]])
        one=torch.as_tensor(splits['val'][:1],device=data['x'].device)
        g=torch.Generator(device=one.device).manual_seed(6180)
        settings['batch_one_prediction_seconds']=latency(
            lambda:sample_batch(model,data,one,condition,baseline,mean,std,alpha,steps,g),one.device,warmups=2,repeats=5)
        settings['ensemble_prediction_seconds']=settings['batch_one_prediction_seconds']*3
        result['step_counts'][str(steps)]=settings;dump(output/'metrics.json',settings)
        dump(args.output/'metrics.json',result)
        print('DDIM_EVALUATED',args.mode,steps,json.dumps(settings),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default=DEFAULT_DATA);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--regressor',type=Path,required=True);ap.add_argument('--mode',choices=['direct','residual'],required=True)
    ap.add_argument('--epochs',type=int,default=200);ap.add_argument('--max-train-hours',type=float,default=2)
    ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--base',type=int,default=36)
    ap.add_argument('--seed',type=int,default=6180);ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    device=setup(args.seed,args.device);args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'complete.json').exists():return
    arrays,data,splits,source_ids=reg.load_data(args.data,device)
    if args.smoke:splits={k:v[:2] for k,v in splits.items()};args.epochs=1;args.batch_size=2
    frozen,_=load_regressor(args.regressor,device)
    baseline=condition=None
    if args.mode=='residual':
        # Frozen comparator predictions use design parameters and coordinates only, including on train.
        pred=regression_predictions(frozen,data,np.arange(len(arrays['x']))) if not args.smoke else None
        if args.smoke:
            pred=np.zeros_like(arrays['u'])
            for ids in splits.values():pred[ids]=regression_predictions(frozen,data,ids)
        baseline=torch.as_tensor(pred,device=device);del pred
        baseline=baseline.masked_fill(data['mask'][:,None],0)
        fm,fs=train_statistics(data['u'],data['mask'],splits['train'])
        condition=((baseline-fm)/fs).masked_fill(data['mask'][:,None],0)
    del frozen
    raw=data['u'] if baseline is None else data['u']-baseline
    mean,std=train_statistics(raw,data['mask'],splits['train'])
    target=((raw-mean)/std).masked_fill(data['mask'][:,None],0);del raw
    model=ConditionalUNet(residual=args.mode=='residual',base=args.base).to(device)
    parameters=sum(p.numel() for p in model.parameters())
    assert args.smoke or 5_000_000<=parameters<=20_000_000
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0)
    alpha=schedule(device);amp=device.type=='cuda' and torch.cuda.get_device_capability(device)[0]>=7
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update({'experiment':[4,5] if args.mode=='direct' else [6], 'parameters':parameters,
        'timesteps':1000,'beta_schedule':'linear 0.0001 to 0.02','learning_rate':2e-4,'ema_decay':0.999,
        'mixed_precision':amp,'target_mean':mean.flatten().tolist(),'target_std':std.flatten().tolist(),
        'normalization_fit':'training valid pixels only','regressor_sha256':sha(args.regressor),
        'conditioning':'x embedding added to time embedding; residual model also receives two frozen prediction channels',
        'sampling_constraint':'mask zero after every step; clean field component clipped to known peak-normalized [-1,1] bound',
        **metadata(device)})
    dump(args.output/'config.json',config);print('DIFFUSION_CONFIG',json.dumps(config),flush=True)
    best=float('inf');best_epoch=0;stale=0;elapsed=0;start_epoch=1;stop_reason='epochs'
    if (args.output/'last.pt').exists():
        last=torch.load(args.output/'last.pt',map_location=device,weights_only=False)
        for key in ['mode','base','seed','regressor_sha256','source_sha256']:
            if last['config'][key]!=config[key]:raise ValueError(f'Resume config mismatch: {key}')
        model.load_state_dict(last['model']);ema.load_state_dict(last['ema']);optimizer.load_state_dict(last['optimizer'])
        scaler.load_state_dict(last['scaler']);start_epoch=last['epoch']+1;best=last['best'];best_epoch=last['best_epoch'];stale=last['stale'];elapsed=last['elapsed']
        np.random.set_state(last['numpy_rng']);torch.set_rng_state(last['torch_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in last['cuda_rng']])
    begin=time.perf_counter()
    with (args.output/'history.jsonl').open('a',buffering=1) as history:
        for epoch in range(start_epoch,args.epochs+1):
            if elapsed+time.perf_counter()-begin>=args.max_train_hours*3600:stop_reason='training_budget';break
            model.train();loss_sum=0;batches=0;order=np.random.permutation(splits['train']);epoch_start=time.perf_counter();budget_exhausted=False
            for start in range(0,len(order),args.batch_size):
                ids=torch.as_tensor(order[start:start+args.batch_size],device=device)
                t=torch.randint(1000,(len(ids),),device=device);valid=(~data['mask'][ids])[:,None]
                noise=torch.randn_like(target[ids])*valid;a=alpha[t,None,None,None]
                noisy=(a.sqrt()*target[ids]+(1-a).sqrt()*noise)*valid
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                    predicted=model(noisy,t,data['x'][ids],None if condition is None else condition[ids])
                    loss=masked_noise_loss(predicted,noise,data['mask'][ids])
                if not torch.isfinite(loss):raise RuntimeError('Non-finite diffusion loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                scaler.step(optimizer);scaler.update()
                with torch.no_grad():
                    for averaged,current in zip(ema.parameters(),model.parameters()):averaged.lerp_(current,0.001)
                loss_sum+=loss.item();batches+=1
                if args.smoke:break
                if elapsed+time.perf_counter()-begin>=args.max_train_hours*3600:
                    budget_exhausted=True;break
            validation=validation_noise(ema,target,data,splits['val'],condition,alpha,args.batch_size)
            if validation<best:
                best=validation;best_epoch=epoch;stale=0
                torch.save({'model':ema.state_dict(),'config':config,'epoch':epoch,'validation_noise_mse':best},args.output/'best.pt')
            else:stale+=1
            reg.sync(device)
            record={'epoch':epoch,'train_noise_mse':loss_sum/batches,'validation_noise_mse':validation,'seconds':time.perf_counter()-epoch_start}
            history.write(json.dumps(record)+'\n');print('DIFFUSION_EPOCH',args.mode,json.dumps(record),flush=True)
            torch.save({'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
                'scaler':scaler.state_dict(),'config':config,'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,
                'elapsed':elapsed+time.perf_counter()-begin,'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []},args.output/'last.pt')
            if stale>=30:stop_reason='early_stopping';break
            if budget_exhausted:stop_reason='training_budget';break
    elapsed+=time.perf_counter()-begin
    checkpoint=torch.load(args.output/'best.pt',map_location=device,weights_only=False);ema.load_state_dict(checkpoint['model'])
    result=evaluate_sampling(args,ema,data,arrays,splits,source_ids,condition,baseline,mean,std,alpha)
    result.update({'train_seconds':elapsed,'train_gpu_hours':elapsed/3600 if device.type=='cuda' else 0,
                   'best_epoch':best_epoch,'best_validation_noise_mse':best,'stop_reason':stop_reason,**metadata(device)})
    dump(args.output/'metrics.json',result);dump(args.output/'complete.json',{'status':'complete','smoke_test_only':args.smoke})
    print('DIFFUSION_COMPLETE',args.mode,flush=True)


if __name__=='__main__':main()
