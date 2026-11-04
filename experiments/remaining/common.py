"""Shared data, metrics, and reproducibility helpers for Experiments 3–6."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('plain_regression', ROOT/'experiments/deeponet_plain/train.py')
reg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg)
DEFAULT_DATA = '/courses/CS6180.202710/shared/final_proj/em2d_task.h5'


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=str)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(seed, device):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing a CPU fallback for the GPU experiment')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == 'cuda': torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(int(os.getenv('SLURM_CPUS_PER_TASK', '4')), 8))
    return torch.device(device)


def metadata(device):
    return {'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
            'hostname': socket.gethostname(), 'slurm_job_id': os.getenv('SLURM_JOB_ID'),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'source_sha256': {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
            'metric_convention': 'Identical to run 11597463: per-design valid-pixel norms, Euclidean 3-pixel edge band, forward coordinate gradients with valid endpoints'}


def summarize(records):
    keys = ('relative_l2','edge_relative_l2','gradient_relative_l2','masked_mse')
    result = {k: {'mean': float(np.mean([r[k] for r in records])),
                  'median': float(np.median([r[k] for r in records])),
                  'std': float(np.std([r[k] for r in records]))} for k in keys}
    result['designs'] = len(records)
    return result


def score_predictions(predictions, rows, arrays, source_ids):
    return [{'row_index': int(row), 'sample_id': int(source_ids[row]),
             **reg.design_metrics(predictions[j], arrays['u'][row], arrays['mask'][row],
                                  arrays['y_over_h'][row], arrays['z_over_h'][row])}
            for j,row in enumerate(rows)]


def examples(path, predictions, rows, arrays):
    rows = np.asarray(rows[:4]); predictions = np.asarray(predictions[:4])
    np.savez_compressed(path, row_indices=rows, prediction=predictions, truth=arrays['u'][rows],
                        mask=arrays['mask'][rows], y_over_h=arrays['y_over_h'][rows],
                        z_over_h=arrays['z_over_h'][rows])


@torch.inference_mode()
def regression_predictions(model, data, rows, batch=4):
    result=[]
    for start in range(0,len(rows),batch):
        ids=torch.as_tensor(rows[start:start+batch],device=data['x'].device)
        pred=model(data['x'][ids],reg.coordinates(data,ids)).reshape(-1,2,64,128)
        result.append(pred.cpu().numpy())
    return np.concatenate(result)


def load_regressor(path, device):
    from fourier import FourierDeepONet
    ckpt=torch.load(path,map_location=device,weights_only=False)
    cfg=ckpt.get('config',{})
    kind=ckpt.get('kind','plain')
    model=(FourierDeepONet(cfg.get('width',128),cfg.get('rank',64),cfg.get('scale',1.0))
           if kind=='fourier' else reg.DeepONet(cfg.get('width',128),cfg.get('rank',64)))
    model.load_state_dict(ckpt['model']); model.to(device).eval().requires_grad_(False)
    return model,ckpt


def latency(call, device, warmups=5, repeats=20):
    with torch.inference_mode():
        for _ in range(warmups): call()
        reg.sync(device); start=time.perf_counter()
        for _ in range(repeats): call()
        reg.sync(device)
    return (time.perf_counter()-start)/repeats
