"""Run Experiments 3–6 sequentially and record each stage's actual outcome."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from common import dump,metadata,setup


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True);ap.add_argument('--plain-checkpoint',required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    python=sys.executable;here=Path(__file__).resolve().parent
    stages=[('fourier',[python,str(here/'fourier.py'),'--data',args.data,'--plain-checkpoint',args.plain_checkpoint,'--output',str(args.output/'fourier')]),
            ('direct_ddpm_ddim',[python,str(here/'diffusion.py'),'--mode','direct','--data',args.data,'--regressor',str(args.output/'fourier/frozen_regressor.pt'),'--output',str(args.output/'direct')]),
            ('residual_ddpm_ddim',[python,str(here/'diffusion.py'),'--mode','residual','--data',args.data,'--regressor',str(args.output/'fourier/frozen_regressor.pt'),'--output',str(args.output/'residual')])]
    state={'status':'running','smoke_test_only':args.smoke,'stages':{}}
    for name,command in stages:
        if args.smoke:
            command+=['--smoke','--device','cpu']
            if name!='fourier':command+=['--base','4']
        started=time.perf_counter();state['stages'][name]={'status':'running','command':command};dump(args.output/'status.json',state)
        print('STAGE_START',name,flush=True)
        try:
            with (args.output/f'{name}.log').open('a',buffering=1) as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        except subprocess.CalledProcessError as e:
            state['status']='failed';state['stages'][name].update(status='failed',exit_code=e.returncode)
            dump(args.output/'status.json',state);raise
        state['stages'][name].update(status='complete',wall_seconds=time.perf_counter()-started)
        dump(args.output/'status.json',state);print('STAGE_COMPLETE',name,flush=True)
    state['status']='complete';dump(args.output/'status.json',state)
    print('ALL_EXPERIMENTS_COMPLETE',flush=True)


if __name__=='__main__':main()
