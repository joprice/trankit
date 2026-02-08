"""Parity tests for batched_tokenize_words vs per-word tokenize path."""

import pytest
from transformers import XLMRobertaTokenizerFast

from trankit.iterators import batched_tokenize_words, encode_pieces_from_ids, encode_pieces


@pytest.fixture(scope="module")
def tokenizer():
    return XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")


def _old_tokenize_words(tokenizer, words):
    """Reference: the old per-word tokenize path."""
    pieces_str = [[p for p in tokenizer.tokenize(w) if p != '▁'] for w in words]
    for ps in pieces_str:
        if len(ps) == 0:
            ps += ['-']
    return pieces_str


def _old_encode(tokenizer, pieces_str, max_length):
    """Reference: old encode_pieces (string-based)."""
    flat = [p for ps in pieces_str for p in ps]
    return encode_pieces(tokenizer, flat, max_length)


# --- Word lists for parametrized tests ---

WORD_LISTS = [
    pytest.param(["Hello", "world"], id="basic"),
    pytest.param(["running", "quickly", "downstream"], id="subwords"),
    pytest.param(["café", "naïve", "résumé"], id="accented"),
    pytest.param(["你好", "世界"], id="CJK"),
    pytest.param(["a"], id="single-word"),
    pytest.param(["antidisestablishmentarianism"], id="very-long-word"),
    pytest.param(["Hello", " ", "world"], id="whitespace-word"),
    pytest.param(["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"], id="sentence"),
]


@pytest.mark.parametrize("words", WORD_LISTS)
def test_piece_count_parity(tokenizer, words):
    """Batched path produces same number of pieces per word as old path."""
    old = _old_tokenize_words(tokenizer, words)
    new = batched_tokenize_words(tokenizer, words)

    old_lens = [len(ps) for ps in old]
    new_lens = [len(ps) for ps in new]
    assert old_lens == new_lens, f"Word lens differ: old={old_lens}, new={new_lens}"


@pytest.mark.parametrize("words", WORD_LISTS)
def test_token_id_parity(tokenizer, words):
    """Batched path produces same token IDs as old path."""
    old_str = _old_tokenize_words(tokenizer, words)
    old_ids = [tokenizer.convert_tokens_to_ids(ps) for ps in old_str]

    new_ids = batched_tokenize_words(tokenizer, words)

    assert old_ids == new_ids, f"Token IDs differ:\nold={old_ids}\nnew={new_ids}"


@pytest.mark.parametrize("words", WORD_LISTS)
def test_encode_parity(tokenizer, words):
    """encode_pieces_from_ids matches old encode_pieces output."""
    max_length = 512

    old_str = _old_tokenize_words(tokenizer, words)
    old_encoded = _old_encode(tokenizer, old_str, max_length)

    new_ids = batched_tokenize_words(tokenizer, words)
    flat_ids = [p for ps in new_ids for p in ps]
    new_encoded = encode_pieces_from_ids(tokenizer, flat_ids, max_length)

    assert old_encoded == new_encoded, (
        f"Encoded IDs differ:\nold={old_encoded}\nnew={new_encoded}"
    )


def test_encode_truncation(tokenizer):
    """Truncation preserves BOS and EOS tokens correctly."""
    # Generate enough pieces to exceed max_length
    words = ["antidisestablishmentarianism"] * 100
    max_length = 20

    new_ids = batched_tokenize_words(tokenizer, words)
    flat_ids = [p for ps in new_ids for p in ps]
    encoded = encode_pieces_from_ids(tokenizer, flat_ids, max_length)

    assert len(encoded) <= max_length
    assert encoded[0] == tokenizer.bos_token_id
    assert encoded[-1] == tokenizer.eos_token_id


def test_empty_input(tokenizer):
    """Empty word list returns empty result."""
    assert batched_tokenize_words(tokenizer, []) == []


def test_non_fast_tokenizer_raises():
    """Should raise TypeError for non-fast tokenizers."""
    from transformers import XLMRobertaTokenizer
    slow = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    with pytest.raises(TypeError, match="fast tokenizer"):
        batched_tokenize_words(slow, ["hello"])
