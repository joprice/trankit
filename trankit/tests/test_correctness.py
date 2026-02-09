"""
Correctness validation for trankit inference.

Compares current __call__ output against a saved baseline.
Run with:
    python trankit/tests/test_correctness.py
"""

import os
import sys
import json
import torch
import trankit

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "correctness_baseline.json")

LONG_TEXT = (
    "The United Nations Secretary-General called on world leaders to take immediate "
    "action on climate change during a summit in New York. Scientists from the "
    "Intergovernmental Panel on Climate Change presented new findings showing that "
    "global temperatures have risen faster than previously predicted. Several nations, "
    "including France, Germany, and Japan, pledged to reduce carbon emissions by fifty "
    "percent over the next decade. Environmental groups welcomed the commitments but "
    "warned that concrete policy changes are needed to meet the targets. Meanwhile, "
    "the European Central Bank announced new green finance initiatives to support "
    "sustainable development across member states.\n\n"
) * 5

SHORT_TEXT = (
    "John Donovan from Apple Inc. announced a new product today in San Francisco. "
    "The device will be available next month."
)


def normalize(obj):
    """Normalize tuples to lists for JSON round-trip comparison."""
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
                diffs.append(f"{path}.{key}: missing in current")
            elif key not in b:
                diffs.append(f"{path}.{key}: missing in baseline")
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


def _run_cases(p, baseline, label=""):
    """Run test cases against baseline, return True if all pass."""
    prefix = f"[{label}] " if label else ""
    cases = {
        "full_doc": lambda: p(LONG_TEXT),
        "full_doc_short": lambda: p(SHORT_TEXT),
        "full_sent": lambda: p(SHORT_TEXT, is_sent=True),
        "full_pretok": lambda: p([["John", "likes", "cats"], ["Mary", "likes", "dogs"]]),
        "full_pretok_sent": lambda: p(["John", "likes", "cats"], is_sent=True),
    }

    all_passed = True
    with torch.no_grad():
        for name, fn in cases.items():
            result = fn()
            diffs = compare(result, baseline[name])
            if diffs:
                print(f"FAIL: {prefix}{name} - {len(diffs)} differences:")
                for d in diffs[:10]:
                    print(f"  {d}")
                if len(diffs) > 10:
                    print(f"  ... and {len(diffs) - 10} more")
                all_passed = False
            else:
                print(f"PASS: {prefix}{name}")
    return all_passed


def run_validation():
    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    p = trankit.Pipeline("english", embedding="xlm-roberta-base")
    all_passed = _run_cases(p, baseline, label="cached")

    # Second pass: stacked_adapters mode
    p_stacked = trankit.Pipeline("english", embedding="xlm-roberta-base", stacked_adapters=True)
    if not _run_cases(p_stacked, baseline, label="stacked"):
        all_passed = False

    if all_passed:
        print("\nAll correctness checks passed.")
    else:
        print("\nSome checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    run_validation()
