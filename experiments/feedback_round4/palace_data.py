"""New Palace designs and a training-only reconstruction/refinement pilot.

This is a separately versioned synthetic source, not the staff generator.
No fitted scaling or discarded error pixels are used for acceptance.
"""
import argparse,hashlib,json,sys,time,os
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'palace_regen'))
import regen,staffpipe
DATA='/courses/CS6180.202710/shared/final_proj/em2d_task.h5'

def dump(p,x):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def pixel_mask(raw,y,z):
    h,wh,t,er=raw
    # Cell-centre exclusion: conductor intersects each pixel's local half-width.
    dy=np.gradient(y);dz=np.gradient(z)
    return ((abs(y)[None,:]<=wh/2+dy[None,:]/2)&
            (z[:,None]+dz[:,None]/2>=1)&(z[:,None]-dz[:,None]/2<=1+t/h))

def prepare(root,count):
    root.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(6180)
    with h5py.File(DATA) as f:
        train=f['split/train'][:]; raw=f['analysis/params_raw'][:];z=f['z_over_h'][int(train[0])].astype(float)
        differences=[]
        for row in train:
            y=f['y_over_h'][row].astype(float); zz=f['z_over_h'][row].astype(float)
            if not np.array_equal(zz,z):raise ValueError('Nonconstant normalized z grid')
            candidate=pixel_mask(raw[row],y,zz)
            if not np.array_equal(candidate,f['mask'][row]):differences.append(int(row))
        dump(root/'mask-audit.json',{'train_designs':len(train),'mismatching_rows':differences,'definition':'half local spacing in y,z'})
        if differences:raise ValueError('Mask reconstruction differs; do not generate guessed targets')
        lo=np.asarray(f.attrs['x_lo']);hi=np.asarray(f.attrs['x_hi']);threshold=float(f['split'].attrs['extrap_h_threshold_mm'])
        hi_sampling=hi.copy();hi_sampling[0]=np.nextafter(threshold,lo[0])
        v=qmc.scale(qmc.LatinHypercube(d=4,seed=6180).random(count),lo,hi_sampling)
        newraw=v.copy();newraw[:,1]=np.exp(v[:,1]);x=2*(v-lo)/(hi-lo)-1
        ratio=raw[train,2]/raw[train,0];bounds=np.quantile(ratio,[0,1/3,2/3,1]);refs=[]
        for k in range(3):
            choices=train[(ratio>=bounds[k])&(ratio<=bounds[k+1])]
            refs.extend(rng.choice(choices,8,replace=False).tolist())
        cases=[]
        for i,r in enumerate(newraw):
            # Reject exact existing designs (including held-out design parameters).
            if np.any(np.max(abs(raw-r),axis=1)<1e-9):raise ValueError('Duplicate design')
            cases.append(dict(id=f'new-{i:04}',kind='new',raw=r.tolist(),x=x[i].tolist(),uniform=1.,sample_id=10000000+i))
        for i,row in enumerate(refs):
            cases.append(dict(id=f'ref-{row:05}',kind='reference',row=row,raw=raw[row].tolist(),uniform=1.))
            if i%3==0:cases.append(dict(id=f'refine-{row:05}',kind='refinement',row=row,raw=raw[row].tolist(),uniform=.75))
        np.save(root/'z_over_h.npy',z)
        dump(root/'cases.json',dict(seed=6180,count=count,cases=cases,source_dataset_sha256=digest(DATA),
          reference_split='train',mask_audit_passed=True,new_h_upper_exclusive=threshold,
          generator='palace-rebuild f8d7df46 plus feedback_round4; inferred staff convention',
          training_eligible=False,requires_gate=True))

def run(root,index,palace):
    manifest=json.loads((root/'cases.json').read_text());cases=manifest['cases'];z=np.load(root/'z_over_h.npy')
    for case in cases[index::16]:
        case_dir=root/'cases'/case['id'];case_dir.mkdir(parents=True,exist_ok=True)
        if (case_dir/'complete.json').exists():continue
        raw=np.array(case['raw']);h,wh,t,er=raw
        if case['kind']=='new':
            y=np.linspace(-(wh/2+6),wh/2+6,128)
            mask=pixel_mask(raw,y,z)
            d=dict(row=case['sample_id'],sid=case['sample_id'],h=h,w=wh*h,t=t,er=er,y=y*h,z=z*h,
                   u=np.zeros((2,64,128)),mask=mask)
        else:d=regen.load_row(DATA,case['row'])
        args=SimpleNamespace(out=str(case_dir),palace=str(palace),np=1)
        try:
            info=regen.run_one(d,args,1,1.,'es_p1',save=1,uniform=case['uniform'])
            u,E,scale=staffpipe.staff_u(info['dir'],d)
            # The mesh coordinates above are millimetres; convert gradient units to V/m.
            E,scale=E*1000,scale*1000
            if not np.isfinite(u).all() or not np.isfinite(scale) or scale<=0:raise ValueError('Nonfinite/zero field')
            if np.max(abs(u[:,d['mask']]))!=0:raise ValueError('Mask not zero')
            np.savez_compressed(case_dir/'sample.npz',u=u.astype('f4'),E=E.astype('f4'),s=scale,mask=d['mask'],
              y_over_h=d['y']/h,z_over_h=d['z']/h,params_raw=raw)
            result=dict(case=case,solver=info,sample_sha256=digest(case_dir/'sample.npz'))
            if case['kind']!='new':
                valid=~d['mask'];result['raw_relative_l2']=float(np.linalg.norm((u-d['u'])[:,valid])/np.linalg.norm(d['u'][:,valid]))
            dump(case_dir/'complete.json',result);print(case['id'],result.get('raw_relative_l2'),flush=True)
        except Exception as exc:
            dump(case_dir/'failed.json',dict(case=case,error=str(exc)));raise

def pack(root):
    manifest=json.loads((root/'cases.json').read_text());cases=manifest['cases'];refs=[];refine=[]
    for case in cases:
        p=root/'cases'/case['id']/'complete.json'
        if not p.exists():raise ValueError(f'Case incomplete: {case["id"]}')
        record=json.loads(p.read_text())
        if digest(p.parent/'sample.npz')!=record['sample_sha256']:raise ValueError('Sample changed')
        if case['kind']=='reference':refs.append(record['raw_relative_l2'])
        if case['kind']=='refinement':
            a=np.load(p.parent/'sample.npz');b=np.load(root/'cases'/f'ref-{case["row"]:05}'/'sample.npz');valid=~a['mask']
            refine.append(float(np.linalg.norm((a['u']-b['u'])[:,valid])/np.linalg.norm(a['u'][:,valid])))
    gate=dict(reference_mean=float(np.mean(refs)),reference_p95=float(np.quantile(refs,.95)),
              refinement_max=float(max(refine)),thresholds={'reference_mean':.05,'reference_p95':.10,'refinement_max':.02})
    gate['passed']=gate['reference_mean']<=.05 and gate['reference_p95']<=.10 and gate['refinement_max']<=.02
    gate['note']='No fitted scale, no removed spikes. Passing allows a separate augmentation experiment; it does not prove staff-source equivalence.'
    dump(root/'quality-gate.json',gate)
    fresh=[c for c in cases if c['kind']=='new'];dest=root/'palace-pilot.h5'
    with h5py.File(str(dest)+'.tmp','w') as f:
        f.attrs['source']='Palace v0.19.0; reconstructed electrostatic pipeline';f.attrs['course_compatible_gate']=gate['passed']
        f.attrs['source_dataset_sha256']=manifest['source_dataset_sha256'];f.attrs['seed']=6180;f.attrs['s_units']='V/m'
        f.attrs['normalization']='E from sampled V chord differences; divide by peak vector magnitude over unmasked pixels'
        f.attrs['provenance']='See cases.json, per-case solver/config/mesh/log, quality-gate.json and SHA256 manifest'
        for key,shape,dtype in [('u',(len(fresh),2,64,128),'f4'),('mask',(len(fresh),64,128),'?'),('y_over_h',(len(fresh),128),'f4'),('z_over_h',(len(fresh),64),'f4'),('analysis/params_raw',(len(fresh),4),'f8'),('x',(len(fresh),4),'f4'),('s',(len(fresh),),'f8'),('analysis/sample_id',(len(fresh),),'i8')]:
            f.create_dataset(key,shape,dtype=dtype,compression='gzip')
        for i,c in enumerate(fresh):
            with np.load(root/'cases'/c['id']/'sample.npz') as a:
                for key in ['u','mask','y_over_h','z_over_h','s']:f[key][i]=a[key]
                f['analysis/params_raw'][i]=a['params_raw'];f['x'][i]=c['x'];f['analysis/sample_id'][i]=c['sample_id']
    Path(str(dest)+'.tmp').replace(dest)
    dump(root/'complete.json',dict(status='complete',new_designs=len(fresh),reference_designs=len(refs),refined_designs=len(refine),
      training_eligible=gate['passed'],h5_sha256=digest(dest)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run','pack']);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--count',type=int,default=256);ap.add_argument('--index',type=int,default=0);ap.add_argument('--palace',type=Path)
    a=ap.parse_args()
    if a.action=='prepare':prepare(a.root,a.count)
    elif a.action=='run':run(a.root,a.index,a.palace)
    else:pack(a.root)
