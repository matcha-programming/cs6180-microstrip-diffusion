"""Run 4R and the matched position-conditioned Experiment 4 on one GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from paired_model import PositionUNet,paired_metadata
from common import dump,setup


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True);ap.add_argument('--regressor',required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    device=setup(6180,'cpu' if args.smoke else 'cuda');base=4 if args.smoke else 36
    # Assert the two runs start from exactly the same shared backbone initialization.
    setup(6180,'cpu');a=PositionUNet(base=base)
    setup(6180,'cpu');b=PositionUNet(base=base)
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
    pair={'experiments':['4R','4-position-paired'],'parameters_each':sum(p.numel() for p in a.parameters()),
          'same_initial_weights':True,'same_position_planes':torch.equal(a.position,b.position),
          'smoke_test_only':args.smoke,**paired_metadata(device)}
    dump(args.output/'pair-config.json',pair);del a,b
    here=Path(__file__).resolve().parent
    commands=[('regression',[sys.executable,str(here/'train.py'),'--data',args.data,'--output',str(args.output/'regression')]),
              ('diffusion',[sys.executable,str(here/'run_diffusion.py'),'--mode','direct','--data',args.data,'--regressor',args.regressor,'--output',str(args.output/'diffusion')])]
    state={'status':'running','smoke_test_only':args.smoke,'stages':{}}
    for name,command in commands:
        if args.smoke:command+=['--smoke','--base','4','--device','cpu']
        state['stages'][name]={'status':'running','command':command};dump(args.output/'status.json',state)
        print('PAIR_STAGE_START',name,flush=True);start=time.perf_counter()
        try:
            with (args.output/f'{name}.log').open('a',buffering=1) as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            actual=json.loads((args.output/name/'config.json').read_text())
            if actual['initial_model_sha256']!=pair['initial_model_sha256']:
                raise RuntimeError(f'{name}: actual initialization differs from paired initialization')
            if actual['parameters']!=pair['parameters_each']:
                raise RuntimeError(f'{name}: paired parameter-count mismatch')
        except Exception as e:
            state['status']='failed';state['stages'][name].update(status='failed',exit_code=getattr(e,'returncode',None),error=str(e))
            dump(args.output/'status.json',state);raise
        state['stages'][name].update(status='complete',wall_seconds=time.perf_counter()-start,initialization_verified=True)
        dump(args.output/'status.json',state);print('PAIR_STAGE_COMPLETE',name,flush=True)
    state['status']='complete';dump(args.output/'status.json',state);print('PAIR_COMPLETE',flush=True)


if __name__=='__main__':main()
