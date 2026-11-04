"""Run the unchanged DDPM protocol with the exact 4R position-conditioned backbone."""
from paired_model import PositionUNet, paired_metadata
import argparse
import common
import models

# Substitute only the backbone and provenance before the original training module imports them.
def make_backbone(*args,**kwargs):
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--seed',type=int,default=6180)
    seed=parser.parse_known_args()[0].seed
    # The original loader constructs a frozen regressor first, consuming random numbers.
    # Reset here so the actual DDPM initialization matches 4R, not only its nominal seed.
    common.setup(seed,'cpu')
    return PositionUNet(*args,**kwargs)

models.ConditionalUNet=make_backbone
common.metadata=paired_metadata
import diffusion

if __name__=='__main__':diffusion.main()
