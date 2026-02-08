from ..utils.base_utils import *
from collections import namedtuple
from ..utils.tbinfo import langwithner
from ..utils.mwt_lemma_utils.mwt_utils import get_mwt_expansions
from ..utils.posdep_utils import *
from ..utils.tokenizer_utils import *
from ..utils.ner_utils import *


def encode_pieces(tokenizer, pieces, max_length):
    """Encode pre-tokenized wordpieces to IDs with special tokens.

    The slow XLMRobertaTokenizer accepts a list of token strings in .encode(),
    but the fast tokenizer does not.  This helper works with both.
    """
    ids = [tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(pieces) + [tokenizer.eos_token_id]
    if len(ids) > max_length:
        ids = ids[:max_length - 1] + [ids[-1]]  # keep EOS
    return ids


def batched_tokenize_words(tokenizer, words):
    """Tokenize words in a single batched call. Requires a fast tokenizer."""
    if not words:
        return []
    if not tokenizer.is_fast:
        raise TypeError(
            f"batched_tokenize_words requires a fast tokenizer "
            f"(got {type(tokenizer).__name__})"
        )
    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False, verbose=False)
    wids = enc.word_ids()
    if wids is None:
        raise RuntimeError(
            "word_ids() returned None — is_split_into_words may not be supported"
        )
    ids = enc.input_ids
    # Filter out standalone ▁ tokens (ID for '▁') to match the old
    # `if p != '▁'` filtering in the per-word tokenize path.
    spiece_underline_id = tokenizer.convert_tokens_to_ids('▁')
    n = len(words)
    pieces = [[] for _ in range(n)]
    for pos, wid in enumerate(wids):
        if wid is not None and ids[pos] != spiece_underline_id:
            pieces[wid].append(ids[pos])
    # Placeholder for words that produce zero pieces (e.g. whitespace-only)
    placeholder_id = tokenizer.convert_tokens_to_ids('-')
    if placeholder_id is None or placeholder_id < 0:
        placeholder_id = tokenizer.unk_token_id
    if placeholder_id is None or placeholder_id < 0:
        raise ValueError("Cannot determine a valid placeholder token ID")
    for i in range(n):
        if not pieces[i]:
            pieces[i] = [placeholder_id]
    return pieces


def encode_pieces_from_ids(tokenizer, flat_ids, max_length):
    """Wrap pre-computed piece IDs with BOS/EOS and truncate."""
    ids = [tokenizer.bos_token_id] + list(flat_ids) + [tokenizer.eos_token_id]
    if len(ids) > max_length:
        ids = ids[:max_length - 1] + [ids[-1]]  # keep EOS
    return ids


def batch_to_device(batch, device, non_blocking=False):
    """Move all tensor fields in a namedtuple batch to the specified device."""
    updates = {}
    for field in batch._fields:
        val = getattr(batch, field)
        if isinstance(val, torch.Tensor):
            updates[field] = val.to(device, non_blocking=non_blocking)
    return batch._replace(**updates) if updates else batch
