"""Parity tests for batched tokenizer_utils hot paths."""

import pytest
from transformers import XLMRobertaTokenizerFast

from trankit.utils.tokenizer_batching import batched_tokenize_pseudo_tokens
from trankit.utils.tokenizer_utils import get_mapping_wp_character_to_or_character


@pytest.fixture(scope="module")
def tokenizer():
    return XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")


# --- Reference implementations (old per-token loops) ---

def _old_tokenize_pseudo_tokens(tokenizer, pseudo_tokens):
    """Old per-token loop from wordpiece_tokenize_from_raw_text line 95."""
    return [
        [p for p in tokenizer.tokenize(t) if p != '▁']
        for t in pseudo_tokens
    ]


def _old_get_mapping(tokenizer, wp_single_string, or_single_string):
    """Old per-character loop from get_mapping_wp_character_to_or_character."""
    wp_char_to_or_char = {}
    converted_text = ''
    for char_id, char in enumerate(or_single_string):
        converted_chars = ''.join(
            [c if not c.startswith('▁') else c[1:]
             for c in tokenizer.tokenize(char) if c != '▁'])
        for converted_c in converted_chars:
            c_id = len(converted_text)
            wp_char_to_or_char[c_id] = char_id
            converted_text += converted_c
    return wp_char_to_or_char


# --- Test inputs ---

PSEUDO_TOKEN_LISTS = [
    pytest.param(["Hello", "world"], id="basic"),
    pytest.param(["running", "quickly"], id="subwords"),
    pytest.param(["café", "naïve", "résumé"], id="accented"),
    pytest.param(["你", "好", "世", "界"], id="CJK-chars"),
    pytest.param([".", ",", "!", "?", ";"], id="punctuation"),
    pytest.param(["Hello", ".", "world", "!"], id="mixed-punct"),
    pytest.param(["e\u0301"], id="combining-mark"),  # é as e + combining acute
    pytest.param(["\ufb01"], id="ligature-fi"),
    pytest.param(["مرحبا"], id="Arabic"),
    pytest.param(["สวัสดี"], id="Thai"),
    pytest.param(["नमस्ते"], id="Devanagari"),
    pytest.param(["a"], id="single-char"),
    pytest.param(["antidisestablishmentarianism"], id="very-long-word"),
]

ORIGINAL_STRINGS = [
    pytest.param("Hello world", id="basic"),
    pytest.param("café naïve", id="accented"),
    pytest.param("你好世界", id="CJK"),
    pytest.param("Hello,world!", id="punctuation"),
    pytest.param("e\u0301", id="combining-mark"),
    pytest.param("\ufb01", id="ligature-fi"),
    pytest.param("مرحبا", id="Arabic"),
    pytest.param("สวัสดี", id="Thai"),
    pytest.param("नमस्ते", id="Devanagari"),
    pytest.param("a", id="single-char"),
    pytest.param("Thequickbrownfox", id="no-spaces"),
]


# --- batched_tokenize_pseudo_tokens tests ---

@pytest.mark.parametrize("pseudo_tokens", PSEUDO_TOKEN_LISTS)
def test_pseudo_tokens_parity(tokenizer, pseudo_tokens):
    """Batched pseudo-token tokenization matches old per-token loop."""
    old = _old_tokenize_pseudo_tokens(tokenizer, pseudo_tokens)
    new = batched_tokenize_pseudo_tokens(tokenizer, pseudo_tokens)
    assert old == new, f"Pieces differ:\nold={old}\nnew={new}"


def test_pseudo_tokens_empty(tokenizer):
    """Empty input returns empty result."""
    assert batched_tokenize_pseudo_tokens(tokenizer, []) == []


def test_pseudo_tokens_non_fast_raises():
    """Should raise TypeError for non-fast tokenizers."""
    from transformers import XLMRobertaTokenizer
    slow = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    with pytest.raises(TypeError, match="fast tokenizer"):
        batched_tokenize_pseudo_tokens(slow, ["hello"])


# --- get_mapping_wp_character_to_or_character tests ---

@pytest.mark.parametrize("or_string", ORIGINAL_STRINGS)
def test_mapping_parity(tokenizer, or_string):
    """Batched character mapping matches old per-character loop."""
    # wp_single_string is unused in new implementation but kept for compat
    wp_string = "dummy"
    old = _old_get_mapping(tokenizer, wp_string, or_string)
    new = get_mapping_wp_character_to_or_character(tokenizer, wp_string, or_string)
    assert old == new, f"Mapping differs for '{or_string}':\nold={old}\nnew={new}"


def test_mapping_empty(tokenizer):
    """Empty input returns empty mapping."""
    assert get_mapping_wp_character_to_or_character(tokenizer, "", "") == {}


def test_mapping_non_fast_raises():
    """Should raise TypeError for non-fast tokenizers."""
    from transformers import XLMRobertaTokenizer
    slow = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    with pytest.raises(TypeError, match="fast tokenizer"):
        get_mapping_wp_character_to_or_character(slow, "test", "test")
