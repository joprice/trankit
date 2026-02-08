# Trankit Stage-Level Batching Benchmark for Colab T4
# Compares serial vs batched throughput to measure GPU utilization gains
# Copy this entire cell into Colab
# First: Runtime > Change runtime type > GPU (T4)

# ── Config ──────────────────────────────────────────────────
EMBEDDING = "xlm-roberta-large"
CACHE_ADAPTERS = True
NUM_DOCS = 100
TARGET_WORDS = 300
WARMUP = 5
BATCH_SIZES = [1, 5, 10, 25, 50]  # 1 = serial baseline

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
import subprocess
import threading
from trankit import Pipeline
from trankit.batch_pipeline import batch_process

REPORT_PATH = "batch_benchmark_report.txt"

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


def count_tokens(result):
    if "sentences" in result:
        return sum(len(s["tokens"]) for s in result["sentences"])
    return len(result.get("tokens", []))


def count_sentences(result):
    return len(result["sentences"]) if "sentences" in result else 1


def gpu_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


class GpuMonitor:
    def __init__(self):
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"], text=True
                ).strip()
                gpu_pct, mem_used, mem_total = [x.strip() for x in out.split(",")]
                self.samples.append({
                    "gpu_pct": int(gpu_pct),
                    "mem_used_mb": int(mem_used),
                    "mem_total_mb": int(mem_total),
                })
            except Exception:
                pass
            self._stop.wait(0.5)

    def summary(self):
        if not self.samples:
            return {}
        gpu = [s["gpu_pct"] for s in self.samples]
        mem = [s["mem_used_mb"] for s in self.samples]
        return {
            "samples": len(self.samples),
            "gpu_mean_pct": round(statistics.mean(gpu), 1),
            "gpu_max_pct": max(gpu),
            "mem_max_mb": max(mem),
            "mem_total_mb": self.samples[0]["mem_total_mb"],
        }


def percentile(sorted_data, p):
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (p / 100) * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


# ── 4. Initialize ────────────────────────────────────────────
print(f"{'=' * 70}")
print(f"Trankit Batch Benchmark — {EMBEDDING}")
print(f"Documents: {NUM_DOCS} | ~{TARGET_WORDS} words/doc")
print(f"Batch sizes: {BATCH_SIZES}")
print(f"{'=' * 70}\n")

torch.cuda.empty_cache()
p = Pipeline("english", gpu=True, cache_dir="./cache", embedding=EMBEDDING, cache_adapters=CACHE_ADAPTERS)
device_type = str(p._config.device.type)
print(f"Device: {device_type}")
print(f"VRAM after init: {gpu_mb():.0f}MB\n")

doc_text = make_document(TARGET_WORDS)
actual_words = len(doc_text.split())

# Get token/sentence counts from a single run
with torch.inference_mode():
    sample = p(doc_text)
num_tokens = count_tokens(sample)
num_sentences = count_sentences(sample)
print(f"Document: {actual_words} words, {num_sentences} sentences, ~{num_tokens} tokens\n")

all_results = []

# ── 5. Run each batch size ───────────────────────────────────
print(f"{'Batch':>6s} {'docs/s':>7s} {'tok/s':>8s} {'mean':>8s} {'p95':>8s} {'GPU%':>6s} {'GPU max':>8s} {'VRAM':>7s}")
print("-" * 72)

for bs in BATCH_SIZES:
    # Warmup
    with torch.inference_mode():
        for _ in range(WARMUP):
            if bs == 1:
                p(doc_text)
            else:
                p.batch([doc_text] * min(bs, 5))
        torch.cuda.synchronize()

    # Timed run
    monitor = GpuMonitor()
    times = []  # per-doc times

    with torch.inference_mode():
        monitor.start()
        if bs == 1:
            # Serial baseline
            for _ in range(NUM_DOCS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                p(doc_text)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        else:
            # Batched
            docs_list = [doc_text] * NUM_DOCS
            for i in range(0, NUM_DOCS, bs):
                chunk = docs_list[i:i + bs]
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                batch_process(p, chunk)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                per_doc = elapsed / len(chunk)
                times.extend([per_doc] * len(chunk))
        monitor.stop()

    total = sum(times)
    sorted_t = sorted(times)
    mean_t = statistics.mean(times)
    gs = monitor.summary()

    result = {
        "batch_size": bs,
        "num_docs": NUM_DOCS,
        "total_sec": round(total, 2),
        "docs_per_sec": round(NUM_DOCS / total, 2),
        "tokens_per_sec": round(num_tokens * NUM_DOCS / total, 1),
        "mean_ms": round(mean_t * 1000, 1),
        "p95_ms": round(percentile(sorted_t, 95) * 1000, 1),
        "min_ms": round(min(times) * 1000, 1),
        "max_ms": round(max(times) * 1000, 1),
        "gpu_mean_pct": gs.get("gpu_mean_pct", 0),
        "gpu_max_pct": gs.get("gpu_max_pct", 0),
        "mem_max_mb": gs.get("mem_max_mb", 0),
    }
    all_results.append(result)

    print(f"  {bs:>4d} {result['docs_per_sec']:>7.1f} {result['tokens_per_sec']:>8.0f} "
          f"{result['mean_ms']:>7.0f}ms {result['p95_ms']:>7.0f}ms "
          f"{result['gpu_mean_pct']:>5.0f}% {result['gpu_max_pct']:>7.0f}% "
          f"{result['mem_max_mb']:>6d}MB")

# ── 6. Summary ───────────────────────────────────────────────
baseline = all_results[0]
print(f"\n{'=' * 70}")
print("SPEEDUP vs serial (batch=1)")
print("=" * 70)
for r in all_results:
    speedup = r["docs_per_sec"] / baseline["docs_per_sec"] if baseline["docs_per_sec"] > 0 else 0
    print(f"  batch={r['batch_size']:<4d}  {r['docs_per_sec']:>5.1f} docs/s  "
          f"{r['tokens_per_sec']:>6.0f} tok/s  GPU={r['gpu_mean_pct']:.0f}%  "
          f"speedup={speedup:.2f}x")

print(f"\n{'=' * 70}")

# ── 7. Save ──────────────────────────────────────────────────
gpu_name = torch.cuda.get_device_name(0).replace(" ", "-")
json_path = f"batch_benchmark_{EMBEDDING.replace('/', '-')}_{gpu_name}.json"
with open(json_path, "w") as f:
    json.dump({
        "embedding": EMBEDDING,
        "gpu": torch.cuda.get_device_name(0),
        "num_docs": NUM_DOCS,
        "target_words": TARGET_WORDS,
        "actual_words": actual_words,
        "num_tokens": num_tokens,
        "num_sentences": num_sentences,
        "results": all_results,
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
