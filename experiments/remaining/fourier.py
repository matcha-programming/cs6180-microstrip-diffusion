"""Experiment 3: validation-only Fourier scale selection and frozen comparator."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from common import DEFAULT_DATA, dump, metadata, setup, reg, regression_predictions, score_predictions, summarize, examples, latency, sha


class FourierDeepONet(nn.Module):
    def __init__(self,width=128,rank=64,scale=1.0):
        super().__init__(); self.rank=rank
        generator=torch.Generator().manual_seed(6180)
        self.register_buffer('B',torch.randn(128,2,generator=generator)*scale)
        def mlp(n):
            return nn.Sequential(nn.Linear(n,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,2*rank))
        self.branch=mlp(4); self.trunk=mlp(256); self.bias=nn.Parameter(torch.zeros(2))

    def forward(self,x,coords):
        angles=2*torch.pi*torch.matmul(coords/8.0,self.B.T)
        features=torch.cat((angles.sin(),angles.cos()),dim=-1)
        b=self.branch(x).reshape(-1,2,self.rank)
        t=self.trunk(features).reshape(len(x),-1,2,self.rank)
        return torch.einsum('bcp,bkcp->bck',b,t)+self.bias[None,:,None]


def train_candidate(args,scale,arrays,data,splits,device):
    output=args.output/f'scale_{scale:g}'
    if (output/'complete.json').exists():
        return json.loads((output/'complete.json').read_text())
    output.mkdir(parents=True,exist_ok=True)
    setup(args.seed,args.device)
    model=FourierDeepONet(scale=scale).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-6)
    cfg={'kind':'fourier','width':128,'rank':64,'scale':scale,'B_shape':[128,2],
         'coordinate_rescaling':'(y/h,z/h)/8, identical to plain run',
         'seed':args.seed,'parameters':sum(p.numel() for p in model.parameters()),
         'epochs':args.epochs,'patience':15,'batch_size':16,'pixels':1024,**metadata(device)}
    dump(output/'config.json',cfg)
    best=float('inf'); stale=0; best_epoch=0; started=time.perf_counter()
    with (output/'history.jsonl').open('w',buffering=1) as hist:
        for epoch in range(1,args.epochs+1):
            model.train(); losses=[]
            for start in range(0,len(splits['train']),16):
                if start==0: order=np.random.permutation(splits['train'])
                ids=torch.as_tensor(order[start:start+16],device=device)
                pix=torch.multinomial((~data['mask'][ids]).flatten(1).float(),1024,replacement=True)
                truth=data['u'][ids].flatten(2).gather(2,pix[:,None].expand(-1,2,-1))
                loss=(model(data['x'][ids],reg.coordinates(data,ids,pix))-truth).square().mean()
                if not torch.isfinite(loss):raise RuntimeError('Non-finite Fourier loss')
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                losses.append(loss.item())
                if args.smoke:break
            val=reg.validation_loss(model,data,splits['val'])
            row={'epoch':epoch,'train_mse':float(np.mean(losses)),'validation_masked_mse':val}
            hist.write(json.dumps(row)+'\n');print('FOURIER',scale,row,flush=True)
            if val<best:
                best=val;best_epoch=epoch;stale=0
                torch.save({'kind':'fourier','model':model.state_dict(),'config':cfg,
                            'validation_masked_mse':best,'epoch':epoch},output/'best.pt')
            else:stale+=1
            if stale>=15:break
    reg.sync(device)
    result={'scale':scale,'validation_masked_mse':best,'best_epoch':best_epoch,
            'train_seconds':time.perf_counter()-started,'checkpoint':str(output/'best.pt')}
    dump(output/'complete.json',result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default=DEFAULT_DATA);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--plain-checkpoint',type=Path,required=True)
    ap.add_argument('--device',default='cuda',choices=['cuda','cpu']);ap.add_argument('--seed',type=int,default=6180)
    ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();device=setup(args.seed,args.device);args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'complete.json').exists():return
    arrays,data,splits,source_ids=reg.load_data(args.data,device)
    if args.smoke:splits={k:v[:4] for k,v in splits.items()};args.epochs=1
    candidates=[train_candidate(args,s,arrays,data,splits,device) for s in [0.5,1.0,2.0]]
    chosen=min(candidates,key=lambda x:x['validation_masked_mse'])
    checkpoint=torch.load(chosen['checkpoint'],map_location=device,weights_only=False)
    model=FourierDeepONet(scale=chosen['scale']).to(device);model.load_state_dict(checkpoint['model']);model.eval()
    results={'experiment':3,'smoke_test_only':args.smoke,'candidates':candidates,'selected_scale':chosen['scale'],
             'best_validation_masked_mse':chosen['validation_masked_mse'],
             'total_search_train_seconds':sum(c['train_seconds'] for c in candidates),**metadata(device)}
    # No candidate accesses test targets before the validation-only choice is fixed.
    for split in ['test','test_extrap']:
        pred=regression_predictions(model,data,splits[split]);records=score_predictions(pred,splits[split],arrays,source_ids)
        results[split]=summarize(records);dump(args.output/f'{split}_per_design.json',records)
        examples(args.output/f'{split}_examples.npz',pred,splits[split],arrays)
    ids=torch.as_tensor(splits['val'][:1],device=device);coords=reg.coordinates(data,ids)
    results['batch_one_prediction_seconds']=latency(lambda:model(data['x'][ids],coords),device)
    plain=torch.load(args.plain_checkpoint,map_location=device,weights_only=False)
    # Choose one regression comparator for both direct and residual diffusion using validation only.
    if float(plain['validation_masked_mse'])<chosen['validation_masked_mse']:
        frozen={'kind':'plain','config':plain['config'],'model':plain['model'],
                'validation_masked_mse':float(plain['validation_masked_mse'])}
    else:frozen=checkpoint
    torch.save(frozen,args.output/'frozen_regressor.pt')
    torch.save(checkpoint,args.output/'best_fourier.pt')
    results['frozen_regressor_kind']=frozen['kind'];results['frozen_regressor_sha256']=sha(args.output/'frozen_regressor.pt')
    dump(args.output/'metrics.json',results);dump(args.output/'complete.json',{'status':'complete','smoke_test_only':args.smoke})
    print('FOURIER_COMPLETE',json.dumps(results),flush=True)


if __name__=='__main__':main()
