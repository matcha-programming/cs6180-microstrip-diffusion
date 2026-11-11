"""Independent validation checks and DDIM NFE sweep of the completed cosine model."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from core import s,predict,evaluate

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    device=s.setup(6180,'cuda');data=s.FieldData(s.DEFAULT_DATA)
    model,mean,std,ckpt,_=s.load_weights(args.checkpoint,device); cfg=ckpt['config']
    if cfg['data_sha256']!=s.sha(s.DEFAULT_DATA) or cfg['split_hashes']!=data.split_hashes():raise ValueError('Dataset/splits changed')
    rows=data.splits['val'][:16];ids=data.sample_ids[rows];b=data.batch(rows,device)
    a=predict(model,b,ids,6180,mean,std,'diffusion',20)
    c=torch.cat([predict(model,data.batch(rows[i:i+4],device),ids[i:i+4],6180,mean,std,'diffusion',20) for i in range(0,16,4)])
    torch.testing.assert_close(a,c,atol=2e-5,rtol=2e-4)
    b['u']=torch.randn_like(b['u'])*100
    independent=predict(model,b,ids,6180,mean,std,'diffusion',20)
    torch.testing.assert_close(a,independent,atol=0,rtol=0)
    order=np.arange(16)[::-1].copy()
    reversed_result=predict(model,data.batch(rows[order],device),ids[order],6180,mean,std,'diffusion',20)
    torch.testing.assert_close(a,reversed_result.flip(0),atol=2e-5,rtol=2e-4)
    m,rec=evaluate(model,data,rows,mean,std,device,'diffusion',20,batch_size=8)
    old=s.field_validation(model,data,rows,mean,std,device,20,batch_size=8)
    np.testing.assert_allclose(m['masked_mse_global'],old['masked_mse_global'],rtol=1e-5,atol=1e-10)
    np.testing.assert_allclose(m['relative_l2'],old['relative_l2'],rtol=1e-6)
    zero=[];permuted=[]
    for i,row in enumerate(rows):
        loc=data.local([row])[0]; ar=data.arrays; truth=ar['u'][loc]; mask=ar['mask'][loc]
        zero.append(s.metrics.design_metrics(np.zeros_like(truth),truth,mask,ar['y_over_h'][loc],ar['z_over_h'][loc])['relative_l2'])
        permuted.append(s.metrics.design_metrics(a[(i+1)%len(rows)].cpu().numpy(),truth,mask,ar['y_over_h'][loc],ar['z_over_h'][loc])['relative_l2'])
    np.testing.assert_allclose(zero,1.,atol=1e-7)
    if np.mean(permuted)<=m['relative_l2']:raise ValueError('Shuffled-design control did not worsen error')
    checks=dict(status='passed',checkpoint_sha256=s.sha(args.checkpoint),target_independence=True,
      batch_max_abs_difference=float((a-c).abs().max()),row_order_invariance=True,
      independent_global_mse=m['masked_mse_global'],legacy_global_mse=old['masked_mse_global'],
      zero_prediction_relative_l2=float(np.mean(zero)),shuffled_prediction_relative_l2=float(np.mean(permuted)),
      matched_prediction_relative_l2=m['relative_l2'],rows=rows.tolist(),
      noise_validation_note='Legacy noise loss uses a fixed seed and EMA; smoothness alone is not evidence of leakage. Its RNG stream depends on batching, so do not compare that diagnostic across batch sizes.',
      scope='Checks concern the executed sampler, masks and metric aggregation; not a proof of every possible validation error.')
    s.dump(args.output/'checks.json',checks); print(json.dumps(checks),flush=True)
    scores={}
    for steps in [20,50,100]:
        start=time.perf_counter();score,recs=evaluate(model,data,data.splits['val'],mean,std,device,'diffusion',steps,[6180,6181,6182],16)
        s.sync(device);score['elapsed_seconds']=time.perf_counter()-start
        scores[str(steps)]=score;s.dump(args.output/f'per-design-{steps}.json',recs);print(steps,score,flush=True)
    s.dump(args.output/'validation.json',dict(checkpoint_sha256=s.sha(args.checkpoint),split='val',steps=scores))
    s.dump(args.output/'complete.json',{'status':'complete'})
if __name__=='__main__':main()
