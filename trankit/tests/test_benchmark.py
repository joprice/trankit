"""
Benchmark test for trankit inference throughput.

Measures tokens/sec and sentences/sec across individual tasks
and the full pipeline. Run with:

    python trankit/tests/test_benchmark.py [embedding]

embedding defaults to xlm-roberta-base.
"""

import os
import sys
import time
import statistics
import json
import torch
import trankit

WARMUP_RUNS = 2
BENCHMARK_RUNS = 10

# A short document for per-call latency measurement
SHORT_TEXT = (
    "John Donovan from Apple Inc. announced a new product today in San Francisco. "
    "The device will be available next month."
)

# A longer document for throughput measurement (repeated paragraphs)
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
) * 5  # ~5 paragraphs, ~500 words


def count_tokens(result):
    """Count total tokens in a pipeline result."""
    if "sentences" in result:
        return sum(len(s["tokens"]) for s in result["sentences"])
    elif "tokens" in result:
        return len(result["tokens"])
    return 0


def count_sentences(result):
    """Count sentences in a pipeline result."""
    if "sentences" in result:
        return len(result["sentences"])
    return 1


def benchmark_task(fn, text, label, runs=BENCHMARK_RUNS, warmup=WARMUP_RUNS):
    """Benchmark a single task function, returning timing stats and throughput."""
    with torch.inference_mode():
        # Warmup
        for _ in range(warmup):
            result = fn(text)

        # Count tokens/sentences from last warmup result
        num_tokens = count_tokens(result)
        num_sentences = count_sentences(result)

        # Timed runs
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            fn(text)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    mean_t = statistics.mean(times)
    stdev_t = statistics.stdev(times) if len(times) > 1 else 0.0
    median_t = statistics.median(times)
    min_t = min(times)
    max_t = max(times)
    tokens_per_sec = num_tokens / mean_t if mean_t > 0 else 0
    sents_per_sec = num_sentences / mean_t if mean_t > 0 else 0

    return {
        "task": label,
        "runs": runs,
        "num_tokens": num_tokens,
        "num_sentences": num_sentences,
        "mean_sec": round(mean_t, 4),
        "median_sec": round(median_t, 4),
        "stdev_sec": round(stdev_t, 4),
        "min_sec": round(min_t, 4),
        "max_sec": round(max_t, 4),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "sents_per_sec": round(sents_per_sec, 2),
    }


def format_row(r):
    return (
        f"  {r['task']:<25s} "
        f"{r['mean_sec']:>8.4f}s  "
        f"(+/- {r['stdev_sec']:.4f})  "
        f"{r['tokens_per_sec']:>8.1f} tok/s  "
        f"{r['sents_per_sec']:>6.2f} sent/s  "
        f"[{r['num_tokens']} tokens, {r['num_sentences']} sents]"
    )


def run_benchmarks(embedding):
    print(f"\n{'=' * 70}")
    print(f"Trankit Inference Benchmark")
    print(f"Embedding: {embedding}")
    print(f"Warmup runs: {WARMUP_RUNS}  |  Benchmark runs: {BENCHMARK_RUNS}")
    print(f"{'=' * 70}\n")

    print("Initializing pipeline...")
    t0 = time.perf_counter()
    p = trankit.Pipeline("english", embedding=embedding)
    init_time = time.perf_counter() - t0
    print(f"Pipeline initialized in {init_time:.2f}s\n")

    results = []

    # --- Short text benchmarks (latency-focused) ---
    print(f"--- Short text ({len(SHORT_TEXT)} chars) ---")
    for label, fn in [
        ("tokenize (short)", p.tokenize),
        ("posdep (short)", p.posdep),
        ("lemmatize (short)", p.lemmatize),
        ("ner (short)", p.ner),
        ("full pipeline (short)", p),
    ]:
        r = benchmark_task(fn, SHORT_TEXT, label)
        results.append(r)
        print(format_row(r))

    print()

    # --- Long text benchmarks (throughput-focused) ---
    print(f"--- Long text ({len(LONG_TEXT)} chars) ---")
    for label, fn in [
        ("tokenize (long)", p.tokenize),
        ("posdep (long)", p.posdep),
        ("lemmatize (long)", p.lemmatize),
        ("ner (long)", p.ner),
        ("full pipeline (long)", p),
    ]:
        r = benchmark_task(fn, LONG_TEXT, label)
        results.append(r)
        print(format_row(r))

    print(f"\n{'=' * 70}")
    print("Done.\n")

    # Write JSON results for programmatic comparison
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, f"benchmark_results_{embedding}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "embedding": embedding,
                "warmup_runs": WARMUP_RUNS,
                "benchmark_runs": BENCHMARK_RUNS,
                "init_time_sec": round(init_time, 2),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    embedding = sys.argv[1] if len(sys.argv) > 1 else "xlm-roberta-base"
    run_benchmarks(embedding)
