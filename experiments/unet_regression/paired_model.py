"""Shared U-Net for Experiment 4R and its position-conditioned DDPM control."""
from pathlib import Path
import hashlib
import sys
import torch

REMAINING = Path(__file__).resolve().parents[1] / 'remaining'
if str(REMAINING) not in sys.path: sys.path.insert(0, str(REMAINING))
from models import ConditionalUNet
from common import metadata as original_metadata, sha
INITIAL_MODEL_SHA256 = None


class PositionUNet(ConditionalUNet):
    def __init__(self, residual=False, base=36):
        if residual: raise ValueError('The 4/4R paired backbone is not residual diffusion')
        # The inherited four-channel input concatenates two field planes and two fixed positions.
        super().__init__(residual=True, base=base)
        row=torch.linspace(-1,1,64)[:,None].expand(64,128)
        column=torch.linspace(-1,1,128)[None,:].expand(64,128)
        self.register_buffer('position',torch.stack((row,column))[None],persistent=False)
        digest=hashlib.sha256()
        for name,value in sorted(self.state_dict().items()):
            digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
        self.initial_state_sha256=digest.hexdigest()
        global INITIAL_MODEL_SHA256
        INITIAL_MODEL_SHA256=self.initial_state_sha256

    def forward(self,field,t,design,condition=None):
        if condition is not None:raise ValueError('Only fixed position conditioning is allowed in this pair')
        positions=self.position.to(field).expand(len(field),-1,-1,-1)
        return super().forward(field,t,design,positions)


def paired_metadata(device):
    value=original_metadata(device)
    value.update({'paired_source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
                  'initial_model_sha256':INITIAL_MODEL_SHA256,
                  'plan_commit':'e5419728dc08888c17624ea9452dce53b65e99cc',
                  'position_conditioning':'fixed row and column index planes in [-1,1]; identical for 4R and paired 4; not physical coordinate planes'})
    return value
