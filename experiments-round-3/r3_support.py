"""Reuse executed round-two numerical helpers while recording round-three source."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ROUND2 = ROOT / "experiments-round-2"
if str(ROUND2) not in sys.path:
    sys.path.append(str(ROUND2))

from support import (DEFAULT_DATA, FieldData, dump, field_validation, legacy, make_model,
                     masked_epsilon_objective, save_checkpoint, setup, sha, sync, tensors,
                     validation_rows, provenance as round2_provenance)


def provenance(device):
    result = round2_provenance(device)
    folder = Path(__file__).parent
    result["source_sha256"].update({str(p.relative_to(ROOT)): sha(p)
                                  for p in list(folder.glob("*.py")) + [folder / "campaign.json"]})
    return result
