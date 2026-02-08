# Trankit Throughput Benchmark for Colab T4
# Copy this entire cell into Colab
# First: Runtime > Change runtime type > GPU (T4)

# ── Config ──────────────────────────────────────────────────
EMBEDDING = "xlm-roberta-large"  # or "xlm-roberta-base"
NUM_DOCS = 100        # enough for ~30-40s sustained GPU load with large
TARGET_WORDS = 300
TASK = "full"         # full, tokenize, posdep, lemmatize, ner
WARMUP = 5
CACHE_ADAPTERS = True

# ── 1. Check CUDA ────────────────────────────────────────────
import torch
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. Select GPU runtime:")
    print("Runtime > Change runtime type > T4 GPU")
    raise SystemExit(1)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# ── 2. Install ───────────────────────────────────────────────
!pip uninstall -y trankit adapters 2>/dev/null
!pip install --no-cache-dir -q --no-deps --force-reinstall git+https://github.com/joprice/trankit.git@adapter-caching
!pip install --no-cache-dir -q adapters psutil langid filelock tqdm requests protobuf sentencepiece sacremoses regex packaging

import importlib.metadata, pathlib, json as _json
_direct_url = pathlib.Path(importlib.metadata.distribution("trankit")._path) / "direct_url.json"
if _direct_url.exists():
    _commit = _json.loads(_direct_url.read_text()).get("vcs_info", {}).get("commit_id", "unknown")
    print(f"trankit commit: {_commit[:12]}")
else:
    _commit = "unknown"
    print("WARNING: could not determine installed commit")

# ── 3. Setup (suppress warnings, tee output to file) ─────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*adapters available but none.*")

import logging
logging.getLogger("adapters").setLevel(logging.ERROR)

import io
import math
import os
import sys
import time
import statistics
import json
import subprocess
import threading

REPORT_PATH = "throughput_report.txt"

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

from trankit import Pipeline

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
    reps = math.ceil(target_words / _WORDS_PER_PARA)
    return _PARAGRAPH * reps


def gpu_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def gpu_utilization():
    """Sample GPU utilization % via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        gpu_pct, mem_pct, mem_used, mem_total = [x.strip() for x in out.split(",")]
        return {
            "gpu_util_pct": int(gpu_pct),
            "mem_util_pct": int(mem_pct),
            "mem_used_mb": int(mem_used),
            "mem_total_mb": int(mem_total),
        }
    except Exception:
        return None


class GpuMonitor:
    """Background thread that samples nvidia-smi at ~1Hz during the timed run."""

    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            s = gpu_utilization()
            if s:
                self.samples.append(s)
            self._stop.wait(1.0)


def percentile(sorted_data, p):
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (p / 100) * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


def count_tokens(result):
    if "sentences" in result:
        return sum(len(s["tokens"]) for s in result["sentences"])
    return len(result.get("tokens", []))


def count_sentences(result):
    if "sentences" in result:
        return len(result["sentences"])
    return 1


# ── 4. Initialize ────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"Trankit Throughput Benchmark")
print(f"Embedding: {EMBEDDING} | cache_adapters: {CACHE_ADAPTERS}")
print(f"Task: {TASK} | Documents: {NUM_DOCS} | ~{TARGET_WORDS} words/doc")
print("=" * 70)

torch.cuda.empty_cache()
print(f"\nVRAM before init: {gpu_mb():.0f}MB")

t0 = time.perf_counter()
p = Pipeline("english", gpu=True, cache_dir="./cache", embedding=EMBEDDING, cache_adapters=CACHE_ADAPTERS)
init_time = time.perf_counter() - t0
device_type = str(p._config.device.type)
print(f"Pipeline initialized in {init_time:.1f}s (device: {device_type})")
print(f"VRAM after init: {gpu_mb():.0f}MB")

assert device_type == "cuda", f"Expected cuda, got {device_type}"

init_util = gpu_utilization()
if init_util:
    print(f"GPU util: {init_util['gpu_util_pct']}% | VRAM: {init_util['mem_used_mb']}MB / {init_util['mem_total_mb']}MB")

# ── 5. Build task function ───────────────────────────────────
task_fns = {
    "full": lambda text: p(text),
    "tokenize": lambda text: p.tokenize(text),
    "posdep": lambda text: p.posdep(text),
    "lemmatize": lambda text: p.lemmatize(text),
    "ner": lambda text: p.ner(text),
}
assert TASK in task_fns, f"Unknown task: {TASK}. Valid: {list(task_fns)}"
run_fn = task_fns[TASK]

doc_text = make_document(TARGET_WORDS)
actual_words = len(doc_text.split())

# ── 6. Warmup ────────────────────────────────────────────────
print(f"\nWarming up ({WARMUP} docs)...")
with torch.inference_mode():
    for _ in range(WARMUP):
        result = run_fn(doc_text)
    torch.cuda.synchronize()

num_tokens = count_tokens(result)
num_sentences = count_sentences(result)
print(f"Document: {actual_words} words, {num_sentences} sentences, ~{num_tokens} tokens")
print(f"VRAM after warmup: {gpu_mb():.0f}MB")

warmup_util = gpu_utilization()
if warmup_util:
    print(f"GPU util: {warmup_util['gpu_util_pct']}% | VRAM: {warmup_util['mem_used_mb']}MB / {warmup_util['mem_total_mb']}MB")

# ── 7. Timed run ─────────────────────────────────────────────
print(f"\nProcessing {NUM_DOCS} documents...")
monitor = GpuMonitor()
times = []

with torch.inference_mode():
    monitor.start()
    for i in range(NUM_DOCS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run_fn(doc_text)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    monitor.stop()

total_time = sum(times)
print(f"Done in {total_time:.1f}s")

# ── 8. Report ────────────────────────────────────────────────
sorted_t = sorted(times)
mean_t = statistics.mean(times)

stats = {
    "num_docs": NUM_DOCS,
    "total_sec": round(total_time, 2),
    "mean_ms": round(mean_t * 1000, 1),
    "median_ms": round(percentile(sorted_t, 50) * 1000, 1),
    "p95_ms": round(percentile(sorted_t, 95) * 1000, 1),
    "p99_ms": round(percentile(sorted_t, 99) * 1000, 1),
    "min_ms": round(min(times) * 1000, 1),
    "max_ms": round(max(times) * 1000, 1),
    "stdev_ms": round(statistics.stdev(times) * 1000, 1) if len(times) > 1 else 0.0,
    "docs_per_sec": round(NUM_DOCS / total_time, 2),
    "tokens_per_sec": round(num_tokens * NUM_DOCS / total_time, 1),
}

# GPU utilization stats
gpu_stats = {}
if monitor.samples:
    gpu_utils = [s["gpu_util_pct"] for s in monitor.samples]
    mem_utils = [s["mem_used_mb"] for s in monitor.samples]
    gpu_stats = {
        "samples": len(monitor.samples),
        "gpu_util_mean_pct": round(statistics.mean(gpu_utils), 1),
        "gpu_util_max_pct": max(gpu_utils),
        "gpu_util_min_pct": min(gpu_utils),
        "vram_mean_mb": round(statistics.mean(mem_utils)),
        "vram_max_mb": max(mem_utils),
        "vram_total_mb": monitor.samples[0]["mem_total_mb"],
    }

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

if gpu_stats:
    print(f"\nGPU utilization (sampled at ~1Hz during timed run):")
    print(f"  GPU compute:  mean={gpu_stats['gpu_util_mean_pct']}%  "
          f"min={gpu_stats['gpu_util_min_pct']}%  max={gpu_stats['gpu_util_max_pct']}%")
    print(f"  VRAM:         mean={gpu_stats['vram_mean_mb']}MB  "
          f"max={gpu_stats['vram_max_mb']}MB  / {gpu_stats['vram_total_mb']}MB  "
          f"({round(gpu_stats['vram_max_mb'] / gpu_stats['vram_total_mb'] * 100)}% peak)")

# ── 9. Save JSON ─────────────────────────────────────────────
gpu_name = torch.cuda.get_device_name(0).replace(" ", "-")
embedding_name = p._config.embedding_name
out_path = f"throughput_{embedding_name}_{device_type}_{gpu_name}.json"

with open(out_path, "w") as f:
    json.dump(
        {
            "embedding": EMBEDDING,
            "commit": _commit,
            "device": device_type,
            "gpu_name": torch.cuda.get_device_name(0),
            "cache_adapters": CACHE_ADAPTERS,
            "task": TASK,
            "num_docs": NUM_DOCS,
            "target_words": TARGET_WORDS,
            "actual_words": actual_words,
            "num_tokens": num_tokens,
            "num_sentences": num_sentences,
            "warmup": WARMUP,
            "init_time_sec": round(init_time, 2),
            "vram_after_init_mb": round(gpu_mb()),
            "aggregate": stats,
            "gpu_utilization": gpu_stats,
        },
        f,
        indent=2,
    )
print(f"\nJSON saved to {out_path}")

# ── 10. Summary ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Embedding: {EMBEDDING}")
print(f"Task: {TASK} | {NUM_DOCS} docs x ~{num_tokens} tokens")
print(f"Throughput: {stats['docs_per_sec']} docs/s | {stats['tokens_per_sec']} tok/s")
print(f"Latency: {stats['median_ms']:.0f}ms median | {stats['p95_ms']:.0f}ms p95")
if gpu_stats:
    headroom = gpu_stats['vram_total_mb'] - gpu_stats['vram_max_mb']
    print(f"GPU util: {gpu_stats['gpu_util_mean_pct']}% mean")
    print(f"VRAM headroom: {headroom}MB free of {gpu_stats['vram_total_mb']}MB")
print("=" * 70)

# Close report file and download
sys.stdout = _orig_stdout
_report_file.close()
print(f"Report saved to {REPORT_PATH}")

try:
    from google.colab import files
    files.download(REPORT_PATH)
    files.download(out_path)
except ImportError:
    pass
