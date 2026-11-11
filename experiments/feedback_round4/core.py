"""Controlled field prediction objectives and validation for the feedback screen."""
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments-round-2'))
import support as s


def masked_loss(pred, truth, mask, kind='l2'):
    valid = (~mask)[:, None].expand_as(pred)
    error = pred.float() - truth.float()
    if kind not in ('l1', 'l2'):
        raise ValueError(kind)
    return (error.abs() if kind == 'l1' else error.square())[valid].mean()


def training_pair(target, mask, method, alpha):
    valid = (~mask)[:, None]
    n = len(target)
    if method == 'regression':
        return torch.zeros_like(target), torch.zeros(n, device=target.device), target
    noise = torch.randn_like(target) * valid
    if method == 'diffusion':
        t = torch.randint(1000, (n,), device=target.device)
        a = alpha[t, None, None, None]
        return (a.sqrt()*target + (1-a).sqrt()*noise)*valid, t, noise
    if method == 'flow':
        t = torch.rand(n, device=target.device)
        fraction = t[:, None, None, None]
        return ((1-fraction)*noise + fraction*target)*valid, t*999, (target-noise)*valid
    raise ValueError(method)


@torch.inference_mode()
def predict(model, batch, sample_ids, seed, mean, std, method, steps):
    if method == 'diffusion':
        return s.sample(model, batch, sample_ids, seed, mean, std,
                        s.legacy.schedule(batch['x'].device), steps)
    mask = batch['mask'][:, None]
    if method == 'regression':
        z = torch.zeros((len(batch['x']),2,64,128), device=batch['x'].device)
        pred = model(z, torch.zeros(len(z), device=z.device), batch['x'])
    elif method == 'flow':
        if steps < 1:
            raise ValueError('Positive Euler step count required')
        pred = s.initial_noise(sample_ids, seed, batch['x'].device).masked_fill(mask, 0)
        # Straight conditional flow: x_t = (1-t) noise + t normalized field.
        # Exactly `steps` network evaluations; no target is used for sampling.
        for i in range(steps):
            t = torch.full((len(pred),), 999*i/steps, device=pred.device)
            pred = (pred + model(pred, t, batch['x'])/steps).masked_fill(mask, 0)
    else:
        raise ValueError(method)
    return (pred*std+mean).clamp(-1,1).masked_fill(mask,0)


@torch.inference_mode()
def evaluate(model, data, rows, mean, std, device, method, steps=20, seeds=(6180,), batch_size=16):
    model.eval()
    records, error_sum, count = [], 0., 0
    for seed in seeds:
        for start in range(0,len(rows),batch_size):
            ids=rows[start:start+batch_size]
            b=data.batch(ids,device)
            p=predict(model,b,data.sample_ids[ids],seed,mean,std,method,steps)
            if not torch.isfinite(p).all():
                raise ValueError('Non-finite predictions')
            pred=p.cpu().numpy()
            for k,row in enumerate(ids):
                loc=data.local([row])[0]; a=data.arrays; valid=~a['mask'][loc]
                delta=pred[k].astype(np.float64)-a['u'][loc].astype(np.float64)
                error_sum+=float(np.square(delta[:,valid]).sum()); count+=int(valid.sum())*2
                records.append({'row_index':int(row),'sample_id':int(data.sample_ids[row]),'sampling_seed':seed,
                    **s.metrics.design_metrics(pred[k],a['u'][loc],a['mask'][loc],a['y_over_h'][loc],a['z_over_h'][loc])})
    result={key:float(np.mean([r[key] for r in records])) for key in s.METRICS}
    result.update(masked_mse_global=error_sum/count,designs=len(rows),draws=len(seeds),
                  nfe=1 if method=='regression' else steps)
    return result, records
