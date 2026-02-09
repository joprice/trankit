"""
Batch parity tests for trankit.

Validates that batched pipeline paths produce identical results to
sequential single-doc processing.

Run with:
    python trankit/tests/test_batch_parity.py
"""

import os
import sys
import math
import torch
import trankit
from trankit.batch_pipeline import batch_process

_PARAGRAPH = (
    "The United Nations Secretary-General called on world leaders to take immediate "
    "action on climate change during a summit in New York. Scientists from the "
    "Intergovernmental Panel on Climate Change presented new findings showing that "
    "global temperatures have risen faster than previously predicted. Several nations, "
    "including France, Germany, and Japan, pledged to reduce carbon emissions by fifty "
    "percent over the next decade. Environmental groups welcomed the commitments but "
    "warned that concrete policy changes are needed to meet the targets. Meanwhile, "
    "the European Central Bank announced new green finance initiatives to support "
    "sustainable development across member states.\n\n"
)
_WORDS_PER_PARA = len(_PARAGRAPH.split())


def make_document(target_words):
    reps = max(1, math.ceil(target_words / _WORDS_PER_PARA))
    doc = _PARAGRAPH * reps
    words = doc.split()
    if len(words) > target_words:
        doc = " ".join(words[:target_words]) + "."
    return doc


SHORT_DOC = "John Donovan from Apple Inc. announced a new product today."
MEDIUM_DOC = make_document(100)
LONG_DOC = make_document(300)

DOCS = [SHORT_DOC, MEDIUM_DOC, LONG_DOC]


def normalize(obj):
    """Normalize tuples to lists for comparison."""
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize(x) for x in obj]
    return obj


def compare(a, b, path=""):
    """Recursively compare two structures, return list of differences."""
    a, b = normalize(a), normalize(b)
    diffs = []
    if type(a) != type(b):
        diffs.append(f"{path}: type mismatch {type(a).__name__} vs {type(b).__name__}")
        return diffs
    if isinstance(a, dict):
        for key in set(list(a.keys()) + list(b.keys())):
            if key not in a:
                diffs.append(f"{path}.{key}: missing in batch result")
            elif key not in b:
                diffs.append(f"{path}.{key}: missing in sequential result")
            else:
                diffs.extend(compare(a[key], b[key], f"{path}.{key}"))
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
        for i in range(min(len(a), len(b))):
            diffs.extend(compare(a[i], b[i], f"{path}[{i}]"))
    else:
        if a != b:
            diffs.append(f"{path}: {repr(a)} vs {repr(b)}")
    return diffs


def test_tokenize_batch_parity(p):
    """tokenize_batch(docs) must equal [tokenize(d) for d in docs]."""
    with torch.no_grad():
        batch_results = p.tokenize_batch(DOCS)
        seq_results = [p.tokenize(d) for d in DOCS]

    if len(batch_results) != len(seq_results):
        raise RuntimeError(
            f"tokenize_batch returned {len(batch_results)} results, "
            f"expected {len(seq_results)}"
        )

    for i, (br, sr) in enumerate(zip(batch_results, seq_results)):
        diffs = compare(br, sr)
        if diffs:
            raise RuntimeError(
                f"tokenize_batch parity failure for doc {i}: "
                f"{len(diffs)} differences:\n" +
                "\n".join(f"  {d}" for d in diffs[:10])
            )
    print("PASS: tokenize_batch parity")


def test_whitespace_docs(p):
    """Empty/whitespace docs return [] in batch results."""
    ws_docs = ["   ", "\n\n", "\t  \n"]
    with torch.no_grad():
        batch_results = p.tokenize_batch(ws_docs)

    if len(batch_results) != len(ws_docs):
        raise RuntimeError(
            f"whitespace batch returned {len(batch_results)} results, "
            f"expected {len(ws_docs)}"
        )

    for i, br in enumerate(batch_results):
        if br != []:
            raise RuntimeError(
                f"whitespace doc {i} should return [], got {br!r}"
            )
    print("PASS: whitespace docs")


def test_batch_process_parity(p):
    """batch_process(p, docs, batch_tokenize=True) must equal [p(d) for d in docs]."""
    with torch.no_grad():
        batch_results = batch_process(p, DOCS, batch_tokenize=True)
        seq_results = [p(d) for d in DOCS]

    if len(batch_results) != len(seq_results):
        raise RuntimeError(
            f"batch_process returned {len(batch_results)} results, "
            f"expected {len(seq_results)}"
        )

    for i, (br, sr) in enumerate(zip(batch_results, seq_results)):
        diffs = compare(br, sr)
        if diffs:
            raise RuntimeError(
                f"batch_process parity failure for doc {i}: "
                f"{len(diffs)} differences:\n" +
                "\n".join(f"  {d}" for d in diffs[:10])
            )
    print("PASS: batch_process parity")


def test_fused_vs_unfused(p):
    """Fused tagger+NER loop must produce identical results to unfused two-loop path."""
    import trankit.batch_pipeline as bp
    saved = bp._FUSE_TAGGER_NER

    with torch.no_grad():
        bp._FUSE_TAGGER_NER = True
        fused_results = batch_process(p, DOCS, batch_tokenize=True)

        bp._FUSE_TAGGER_NER = False
        unfused_results = batch_process(p, DOCS, batch_tokenize=True)

    bp._FUSE_TAGGER_NER = saved

    if len(fused_results) != len(unfused_results):
        raise RuntimeError(
            f"fused returned {len(fused_results)} results, "
            f"unfused returned {len(unfused_results)}"
        )

    for i, (fr, ur) in enumerate(zip(fused_results, unfused_results)):
        diffs = compare(fr, ur)
        if diffs:
            raise RuntimeError(
                f"fused vs unfused parity failure for doc {i}: "
                f"{len(diffs)} differences:\n" +
                "\n".join(f"  {d}" for d in diffs[:10])
            )
    print("PASS: fused vs unfused parity")


def test_dual_vs_sequential(p):
    """Dual adapter single-pass must produce identical results to two-pass.

    Requires stacked_adapters=True on the pipeline. If not, the test is skipped.
    """
    if not getattr(p, '_stacked_adapters', False):
        print("SKIP: dual vs sequential (requires stacked_adapters=True)")
        return

    import trankit.batch_pipeline as bp
    saved_dual = bp._DUAL_ADAPTER
    saved_fuse = bp._FUSE_TAGGER_NER

    try:
        with torch.no_grad():
            bp._DUAL_ADAPTER = True
            bp._FUSE_TAGGER_NER = True
            dual_results = batch_process(p, DOCS, batch_tokenize=True)

            bp._DUAL_ADAPTER = False
            bp._FUSE_TAGGER_NER = True
            sequential_results = batch_process(p, DOCS, batch_tokenize=True)
    finally:
        bp._DUAL_ADAPTER = saved_dual
        bp._FUSE_TAGGER_NER = saved_fuse

    if len(dual_results) != len(sequential_results):
        raise RuntimeError(
            f"dual returned {len(dual_results)} results, "
            f"sequential returned {len(sequential_results)}"
        )

    for i, (dr, sr) in enumerate(zip(dual_results, sequential_results)):
        diffs = compare(dr, sr)
        if diffs:
            raise RuntimeError(
                f"dual vs sequential parity failure for doc {i}: "
                f"{len(diffs)} differences:\n" +
                "\n".join(f"  {d}" for d in diffs[:10])
            )
    print("PASS: dual vs sequential parity")


def run_parity_tests():
    p = trankit.Pipeline("english", embedding="xlm-roberta-base")
    all_passed = True

    for name, fn in [
        ("tokenize_batch parity", test_tokenize_batch_parity),
        ("whitespace docs", test_whitespace_docs),
        ("batch_process parity", test_batch_process_parity),
        ("fused vs unfused", test_fused_vs_unfused),
    ]:
        try:
            fn(p)
        except RuntimeError as e:
            print(f"FAIL: {name}\n  {e}")
            all_passed = False

    # Dual-adapter test requires stacked_adapters=True
    p_stacked = trankit.Pipeline("english", embedding="xlm-roberta-base",
                                 stacked_adapters=True)
    try:
        test_dual_vs_sequential(p_stacked)
    except RuntimeError as e:
        print(f"FAIL: dual vs sequential\n  {e}")
        all_passed = False

    if all_passed:
        print("\nAll batch parity tests passed.")
    else:
        print("\nSome parity tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    run_parity_tests()
