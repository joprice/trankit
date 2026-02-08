from ..utils.base_utils import *
from collections import namedtuple
from ..utils.tbinfo import langwithner
from ..utils.mwt_lemma_utils.mwt_utils import get_mwt_expansions
from ..utils.posdep_utils import *
from ..utils.tokenizer_utils import *
from ..utils.ner_utils import *


def batch_to_device(batch, device, non_blocking=False):
    """Move all tensor fields in a namedtuple batch to the specified device."""
    updates = {}
    for field in batch._fields:
        val = getattr(batch, field)
        if isinstance(val, torch.Tensor):
            updates[field] = val.to(device, non_blocking=non_blocking)
    return batch._replace(**updates) if updates else batch
