"""
Benchmark test for trankit inference throughput.

Measures tokens/sec and sentences/sec across individual tasks
and the full pipeline. Run with:

    python trankit/tests/test_benchmark.py [embedding]

embedding defaults to xlm-roberta-base.
"""

import cProfile
import math
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

# Single paragraph text for building throughput documents
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
    """Repeat paragraph text to hit approximately target_words."""
    reps = math.ceil(target_words / _WORDS_PER_PARA)
    return _PARAGRAPH * reps


def percentile(sorted_data, p):
    """Linear interpolation percentile on pre-sorted data (p in 0-100)."""
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (p / 100) * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


def format_latency_stats(times, num_tokens, num_sentences):
    """Compute aggregate stats from per-document wall times."""
    sorted_t = sorted(times)
    mean_t = statistics.mean(times)
    total_t = sum(times)
    n = len(times)
    return {
        "num_docs": n,
        "total_sec": round(total_t, 2),
        "mean_ms": round(mean_t * 1000, 1),
        "median_ms": round(percentile(sorted_t, 50) * 1000, 1),
        "p95_ms": round(percentile(sorted_t, 95) * 1000, 1),
        "p99_ms": round(percentile(sorted_t, 99) * 1000, 1),
        "min_ms": round(min(times) * 1000, 1),
        "max_ms": round(max(times) * 1000, 1),
        "stdev_ms": round(statistics.stdev(times) * 1000, 1) if n > 1 else 0.0,
        "docs_per_sec": round(n / total_t, 2) if total_t > 0 else 0,
        "tokens_per_sec": round(num_tokens * n / total_t, 1) if total_t > 0 else 0,
    }


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


def benchmark_task(fn, text, label, runs=BENCHMARK_RUNS, warmup=WARMUP_RUNS, profiler=None):
    """Benchmark a single task function, returning timing stats and throughput."""
    with torch.inference_mode():
        # Warmup (not profiled)
        for _ in range(warmup):
            result = fn(text)

        # Count tokens/sentences from last warmup result
        num_tokens = count_tokens(result)
        num_sentences = count_sentences(result)

        # Timed runs (profiled if profiler provided)
        if profiler is not None:
            profiler.enable()
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            fn(text)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        if profiler is not None:
            profiler.disable()

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


def run_benchmarks(embedding, gpu=True, profile_path=None, cache_adapters=True):
    profiler = None
    if profile_path is not None:
        if profile_path == "":
            # default path
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
            os.makedirs(out_dir, exist_ok=True)
            device_label = "cpu" if not gpu else "gpu"
            profile_path = os.path.join(out_dir, f"bench_{embedding}_{device_label}.prof")
        profiler = cProfile.Profile()

    print(f"\n{'=' * 70}")
    print(f"Trankit Inference Benchmark")
    print(f"Embedding: {embedding}")
    print(f"Warmup runs: {WARMUP_RUNS}  |  Benchmark runs: {BENCHMARK_RUNS}")
    print(f"cache_adapters: {cache_adapters}")
    if profile_path:
        print(f"Profiling: ON (warmup/init excluded)")
    print(f"{'=' * 70}\n")

    print("Initializing pipeline...")
    t0 = time.perf_counter()
    p = trankit.Pipeline("english", embedding=embedding, gpu=gpu, cache_adapters=cache_adapters)
    init_time = time.perf_counter() - t0
    device_type = str(p._config.device.type)
    print(f"Pipeline initialized in {init_time:.2f}s (device: {device_type})\n")

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
        r = benchmark_task(fn, SHORT_TEXT, label, profiler=profiler)
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
        r = benchmark_task(fn, LONG_TEXT, label, profiler=profiler)
        results.append(r)
        print(format_row(r))

    print(f"\n{'=' * 70}")
    print("Done.\n")

    # Write JSON results for programmatic comparison
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, f"benchmark_results_{embedding}_{device_type}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "embedding": embedding,
                "device": device_type,
                "warmup_runs": WARMUP_RUNS,
                "benchmark_runs": BENCHMARK_RUNS,
                "init_time_sec": round(init_time, 2),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {out_path}")

    if profiler is not None:
        profiler.dump_stats(profile_path)
        print(f"Profile written to {profile_path}")


def run_throughput_benchmark(embedding, gpu=True, cache_adapters=True,
                             num_docs=1000, target_words=300,
                             task="full", langs=None,
                             warmup=5, profile_path=None):
    profiler = None
    if profile_path is not None:
        if profile_path == "":
            out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
            os.makedirs(out_dir, exist_ok=True)
            device_label = "cpu" if not gpu else "gpu"
            profile_path = os.path.join(out_dir, f"throughput_{embedding}_{device_label}.prof")
        profiler = cProfile.Profile()

    lang_list = langs if langs else ["english"]
    first_lang = lang_list[0]

    print(f"\n{'=' * 70}")
    print(f"Trankit Throughput Benchmark")
    print(f"Embedding: {embedding} | cache_adapters: {cache_adapters}")
    print(f"Task: {task} pipeline | Documents: {num_docs} | ~{target_words} words/doc")
    if len(lang_list) > 1:
        print(f"Languages: {', '.join(lang_list)} (round-robin)")
    else:
        print(f"Language: {first_lang}")
    print(f"{'=' * 70}\n")

    # 1. Initialize pipeline
    print("Initializing pipeline...")
    t0 = time.perf_counter()
    p = trankit.Pipeline(first_lang, embedding=embedding, gpu=gpu, cache_adapters=cache_adapters)
    init_time = time.perf_counter() - t0
    device_type = str(p._config.device.type)
    print(f"Pipeline initialized in {init_time:.2f}s (device: {device_type})")

    # 2. Add extra languages
    if len(lang_list) > 1:
        for lang in lang_list[1:]:
            p.add(lang)
        print(f"Added: {', '.join(lang_list[1:])}")

    # 3. Generate document text
    doc_text = make_document(target_words)

    # 4. Build doc schedule: [(text, lang), ...]
    schedule = [(doc_text, lang_list[i % len(lang_list)]) for i in range(num_docs)]

    # 5. Map task string to callable
    task_fns = {
        "full": lambda text: p(text),
        "tokenize": lambda text: p.tokenize(text),
        "posdep": lambda text: p.posdep(text),
        "lemmatize": lambda text: p.lemmatize(text),
        "ner": lambda text: p.ner(text),
    }
    if task not in task_fns:
        print(f"Unknown task: {task}. Valid: {', '.join(task_fns)}")
        sys.exit(1)
    base_fn = task_fns[task]
    multi_lang = len(lang_list) > 1

    def run_doc(text, lang):
        if multi_lang:
            p.set_active(lang)
        return base_fn(text)

    # 6. Warmup
    print(f"\nWarming up ({warmup} docs/lang)...")
    num_tokens = 0
    num_sentences = 0
    with torch.inference_mode():
        for lang in lang_list:
            for _ in range(warmup):
                result = run_doc(doc_text, lang)
        # Count from last warmup result
        num_tokens = count_tokens(result)
        num_sentences = count_sentences(result)
    actual_words = len(doc_text.split())
    print(f"Document: {actual_words} words, {num_sentences} sentences, ~{num_tokens} tokens")

    # 7. Timed run
    print(f"\nProcessing {num_docs} documents...")
    times = []
    per_lang_times = {lang: [] for lang in lang_list}
    with torch.inference_mode():
        if profiler is not None:
            profiler.enable()
        for text, lang in schedule:
            start = time.perf_counter()
            run_doc(text, lang)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            per_lang_times[lang].append(elapsed)
        if profiler is not None:
            profiler.disable()

    total_time = sum(times)
    print(f"Done in {total_time:.1f}s")

    # 8. Report
    stats = format_latency_stats(times, num_tokens, num_sentences)

    print(f"\nAggregate:")
    print(f"  Documents/sec:   {stats['docs_per_sec']}")
    print(f"  Tokens/sec:      {stats['tokens_per_sec']}")
    print(f"\nPer-document latency:")
    print(f"  Mean:    {stats['mean_ms']:.1f}ms")
    print(f"  Median:  {stats['median_ms']:.1f}ms")
    print(f"  P95:     {stats['p95_ms']:.1f}ms")
    print(f"  P99:     {stats['p99_ms']:.1f}ms")
    print(f"  Min:     {stats['min_ms']:.1f}ms")
    print(f"  Max:     {stats['max_ms']:.1f}ms")
    print(f"  Stdev:   {stats['stdev_ms']:.1f}ms")

    # Per-language breakdown
    lang_stats = {}
    if multi_lang:
        print(f"\nPer-language breakdown:")
        for lang in lang_list:
            ls = format_latency_stats(per_lang_times[lang], num_tokens, num_sentences)
            lang_stats[lang] = ls
            n = len(per_lang_times[lang])
            print(f"  {lang} ({n} docs): mean={ls['mean_ms']:.1f}ms  "
                  f"p95={ls['p95_ms']:.1f}ms  tok/s={ls['tokens_per_sec']}")

    print(f"\n{'=' * 70}")
    print("Done.\n")

    # 9. Save JSON
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, f"throughput_{embedding}_{device_type}.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "embedding": embedding,
                "device": device_type,
                "cache_adapters": cache_adapters,
                "task": task,
                "num_docs": num_docs,
                "target_words": target_words,
                "actual_words": actual_words,
                "num_tokens": num_tokens,
                "num_sentences": num_sentences,
                "languages": lang_list,
                "warmup_per_lang": warmup,
                "init_time_sec": round(init_time, 2),
                "aggregate": stats,
                "per_language": lang_stats,
            },
            f,
            indent=2,
        )
    print(f"Results saved to {out_path}")

    if profiler is not None:
        profiler.dump_stats(profile_path)
        print(f"Profile written to {profile_path}")


def _get_flag_value(flags, key, default=None):
    """Extract --key=value from flags list, return value or default."""
    prefix = f"--{key}="
    for f in flags:
        if f.startswith(prefix):
            return f.split("=", 1)[1]
    return default


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    embedding = args[0] if args else "xlm-roberta-base"
    gpu = "--cpu" not in flags
    cache_adapters = "--no-cache-adapters" not in flags
    profile_path = None
    for f in flags:
        if f.startswith("--profile="):
            profile_path = f.split("=", 1)[1]
        elif f == "--profile":
            profile_path = ""  # sentinel: use default path

    if "--throughput" in flags:
        num_docs = int(_get_flag_value(flags, "docs", "1000"))
        target_words = int(_get_flag_value(flags, "words", "300"))
        task = _get_flag_value(flags, "task", "full")
        langs_str = _get_flag_value(flags, "langs", None)
        langs = langs_str.split(",") if langs_str else None
        warmup = int(_get_flag_value(flags, "warmup", "5"))
        run_throughput_benchmark(
            embedding, gpu=gpu, cache_adapters=cache_adapters,
            num_docs=num_docs, target_words=target_words,
            task=task, langs=langs, warmup=warmup,
            profile_path=profile_path,
        )
    else:
        run_benchmarks(embedding, gpu=gpu, profile_path=profile_path, cache_adapters=cache_adapters)
