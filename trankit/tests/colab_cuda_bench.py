# Trankit CUDA Benchmark for Colab T4 — adapter-caching branch
# Runs the same per-task + full-pipeline benchmark as test_benchmark.py
# Copy this entire cell into Colab
# First: Runtime > Change runtime type > GPU (T4)

# ── Config ──────────────────────────────────────────────────
EMBEDDING = "xlm-roberta-base"
CACHE_ADAPTERS = True
WARMUP_RUNS = 2
BENCHMARK_RUNS = 10

# ── 1. Check CUDA ────────────────────────────────────────────
import torch
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. Select GPU runtime:")
    print("Runtime > Change runtime type > T4 GPU")
    raise SystemExit(1)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# ── 2. Install ───────────────────────────────────────────────
!pip uninstall -y trankit adapters 2>/dev/null
!pip install --no-cache-dir -q --no-deps --force-reinstall git+https://github.com/joprice/trankit.git@adapter-caching
!pip install --no-cache-dir -q adapters psutil langid filelock tqdm requests protobuf sentencepiece sacremoses regex packaging

# ── 3. Setup ─────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*adapters available but none.*")

import logging
logging.getLogger("adapters").setLevel(logging.ERROR)

import math
import sys
import time
import statistics
import json
from trankit import Pipeline

REPORT_PATH = "cuda_benchmark_report.txt"

class Tee:
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
    def flush(self):
        self.stream.flush()
        self.file.flush()

_report_file = open(REPORT_PATH, "w")
_orig_stdout = sys.stdout
sys.stdout = Tee(_report_file, _orig_stdout)

SHORT_TEXT = (
    "John Donovan from Apple Inc. announced a new product today in San Francisco. "
    "The device will be available next month."
)

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


def count_tokens(result):
    if "sentences" in result:
        return sum(len(s["tokens"]) for s in result["sentences"])
    return len(result.get("tokens", []))


def count_sentences(result):
    return len(result["sentences"]) if "sentences" in result else 1


def gpu_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def benchmark_task(fn, text, label, runs=BENCHMARK_RUNS, warmup=WARMUP_RUNS):
    with torch.inference_mode():
        for _ in range(warmup):
            result = fn(text)

        num_tokens = count_tokens(result)
        num_sentences = count_sentences(result)

        times = []
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn(text)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    mean_t = statistics.mean(times)
    stdev_t = statistics.stdev(times) if len(times) > 1 else 0.0
    tokens_per_sec = num_tokens / mean_t if mean_t > 0 else 0
    sents_per_sec = num_sentences / mean_t if mean_t > 0 else 0

    return {
        "task": label,
        "runs": runs,
        "num_tokens": num_tokens,
        "num_sentences": num_sentences,
        "mean_sec": round(mean_t, 4),
        "stdev_sec": round(stdev_t, 4),
        "min_sec": round(min(times), 4),
        "max_sec": round(max(times), 4),
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


# ── 4. Initialize ────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"Trankit CUDA Benchmark — {EMBEDDING} (adapter-caching)")
print(f"Warmup: {WARMUP_RUNS} | Runs: {BENCHMARK_RUNS}")
print(f"cache_adapters: {CACHE_ADAPTERS}")
print(f"{'=' * 70}\n")

torch.cuda.empty_cache()
t0 = time.perf_counter()
p = Pipeline("english", gpu=True, cache_dir="./cache", embedding=EMBEDDING, cache_adapters=CACHE_ADAPTERS)
init_time = time.perf_counter() - t0
device_type = str(p._config.device.type)
print(f"Device: {device_type}")
print(f"Pipeline initialized in {init_time:.2f}s")
print(f"VRAM after init: {gpu_mb():.0f}MB\n")

results = []

# ── 5. Short text ─────────────────────────────────────────────
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

# ── 6. Long text ──────────────────────────────────────────────
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

# ── 7. Save ──────────────────────────────────────────────────
gpu_name = torch.cuda.get_device_name(0).replace(" ", "-")
json_path = f"benchmark_results_{EMBEDDING.replace('/', '-')}_{gpu_name}.json"
with open(json_path, "w") as f:
    json.dump({
        "embedding": EMBEDDING,
        "branch": "adapter-caching",
        "device": device_type,
        "gpu": torch.cuda.get_device_name(0),
        "cache_adapters": CACHE_ADAPTERS,
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
        "init_time_sec": round(init_time, 2),
        "results": results,
    }, f, indent=2)

sys.stdout = _orig_stdout
_report_file.close()
print(f"Report: {REPORT_PATH}")
print(f"JSON: {json_path}")

try:
    from google.colab import files
    files.download(REPORT_PATH)
    files.download(json_path)
except ImportError:
    pass
