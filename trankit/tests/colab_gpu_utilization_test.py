# Trankit GPU Utilization Diagnostic for Colab T4
# Answers: where is the GPU idle time? Is it between tasks or within tasks?
# Copy this entire cell into Colab
# First: Runtime > Change runtime type > GPU (T4)

# ── Config ──────────────────────────────────────────────────
EMBEDDING = "xlm-roberta-large"
CACHE_ADAPTERS = True
DOCS_PER_TEST = 50  # enough for ~15-20s sustained load per test
WARMUP = 3

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
import math
import time
import statistics
import json
import subprocess
import threading
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


def count_tokens(result):
    if "sentences" in result:
        return sum(len(s["tokens"]) for s in result["sentences"])
    return len(result.get("tokens", []))


def count_sentences(result):
    return len(result["sentences"]) if "sentences" in result else 1


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
            self._stop.wait(0.5)  # sample at 2Hz for finer granularity

    def summary(self):
        if not self.samples:
            return {"gpu_mean": 0, "gpu_max": 0, "mem_max_mb": 0}
        gpu = [s["gpu_pct"] for s in self.samples]
        mem = [s["mem_used_mb"] for s in self.samples]
        return {
            "samples": len(self.samples),
            "gpu_mean_pct": round(statistics.mean(gpu), 1),
            "gpu_max_pct": max(gpu),
            "gpu_min_pct": min(gpu),
            "mem_max_mb": max(mem),
            "mem_total_mb": self.samples[0]["mem_total_mb"],
        }


def run_test(label, fn, doc_text, num_docs):
    """Run fn(doc_text) num_docs times, return timing + GPU stats."""
    # warmup
    with torch.inference_mode():
        for _ in range(WARMUP):
            result = fn(doc_text)
        torch.cuda.synchronize()

    ntok = count_tokens(result)
    nsent = count_sentences(result)

    monitor = GpuMonitor()
    times = []
    with torch.inference_mode():
        monitor.start()
        for _ in range(num_docs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(doc_text)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        monitor.stop()

    total = sum(times)
    mean_t = statistics.mean(times)
    gs = monitor.summary()
    return {
        "label": label,
        "num_docs": num_docs,
        "tokens_per_doc": ntok,
        "sentences_per_doc": nsent,
        "total_sec": round(total, 1),
        "mean_ms": round(mean_t * 1000, 1),
        "docs_per_sec": round(num_docs / total, 2),
        "tokens_per_sec": round(ntok * num_docs / total, 1),
        "gpu_mean_pct": gs.get("gpu_mean_pct", 0),
        "gpu_max_pct": gs.get("gpu_max_pct", 0),
        "mem_max_mb": gs.get("mem_max_mb", 0),
    }


# ── 4. Initialize ────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"GPU Utilization Diagnostic — {EMBEDDING}")
print("=" * 70)

torch.cuda.empty_cache()
p = Pipeline("english", gpu=True, cache_dir="./cache", embedding=EMBEDDING, cache_adapters=CACHE_ADAPTERS)
print(f"Device: {p._config.device.type}\n")

all_results = []

# ═══════════════════════════════════════════════════════════════
# TEST 1: Per-task breakdown at fixed doc size (300 words)
# Which task leaves the most GPU idle?
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: Per-task GPU utilization (~300 word docs)")
print("=" * 70)

doc_300 = make_document(300)
words_300 = len(doc_300.split())
print(f"Document: {words_300} words\n")

print(f"{'Task':<20s} {'docs/s':>7s} {'tok/s':>8s} {'mean':>8s} {'GPU%':>6s} {'GPU max':>8s}")
print("-" * 62)

for task_name, fn in [
    ("tokenize", p.tokenize),
    ("posdep", p.posdep),
    ("lemmatize", p.lemmatize),
    ("ner", p.ner),
    ("full pipeline", p),
]:
    r = run_test(f"task:{task_name}:300w", fn, doc_300, DOCS_PER_TEST)
    all_results.append(r)
    print(f"  {task_name:<18s} {r['docs_per_sec']:>7.1f} {r['tokens_per_sec']:>8.0f} "
          f"{r['mean_ms']:>7.0f}ms {r['gpu_mean_pct']:>5.0f}% {r['gpu_max_pct']:>7.0f}%")

# ═══════════════════════════════════════════════════════════════
# TEST 2: Vary document size (full pipeline)
# Does the GPU want bigger batches?
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("TEST 2: Full pipeline — varying document size")
print("=" * 70)

print(f"\n{'Words':<8s} {'Sents':>6s} {'Tokens':>7s} {'docs/s':>7s} {'tok/s':>8s} {'mean':>8s} {'GPU%':>6s} {'GPU max':>8s}")
print("-" * 70)

for target_words in [50, 150, 300, 600, 1200, 2400]:
    doc = make_document(target_words)
    actual_words = len(doc.split())
    # fewer docs for larger sizes to keep total time reasonable
    n = max(10, DOCS_PER_TEST * 300 // target_words)
    r = run_test(f"full:{target_words}w", p, doc, n)
    all_results.append(r)
    print(f"  {actual_words:<7d} {r['sentences_per_doc']:>6d} {r['tokens_per_doc']:>7d} "
          f"{r['docs_per_sec']:>7.1f} {r['tokens_per_sec']:>8.0f} "
          f"{r['mean_ms']:>7.0f}ms {r['gpu_mean_pct']:>5.0f}% {r['gpu_max_pct']:>7.0f}%")

# ═══════════════════════════════════════════════════════════════
# TEST 3: Tokenize-only at varying sizes
# Isolate the XLM-R forward pass scaling
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("TEST 3: Tokenize only — varying document size")
print("=" * 70)

print(f"\n{'Words':<8s} {'Sents':>6s} {'Tokens':>7s} {'docs/s':>7s} {'tok/s':>8s} {'mean':>8s} {'GPU%':>6s} {'GPU max':>8s}")
print("-" * 70)

for target_words in [50, 150, 300, 600, 1200, 2400]:
    doc = make_document(target_words)
    actual_words = len(doc.split())
    n = max(10, DOCS_PER_TEST * 300 // target_words)
    r = run_test(f"tokenize:{target_words}w", p.tokenize, doc, n)
    all_results.append(r)
    print(f"  {actual_words:<7d} {r['sentences_per_doc']:>6d} {r['tokens_per_doc']:>7d} "
          f"{r['docs_per_sec']:>7.1f} {r['tokens_per_sec']:>8.0f} "
          f"{r['mean_ms']:>7.0f}ms {r['gpu_mean_pct']:>5.0f}% {r['gpu_max_pct']:>7.0f}%")

# ── Save ─────────────────────────────────────────────────────
gpu_name = torch.cuda.get_device_name(0).replace(" ", "-")
out_path = f"gpu_utilization_diagnostic_{EMBEDDING.replace('/', '-')}_{gpu_name}.json"
with open(out_path, "w") as f:
    json.dump({"embedding": EMBEDDING, "gpu": torch.cuda.get_device_name(0), "results": all_results}, f, indent=2)
print(f"\nResults saved to {out_path}")

try:
    from google.colab import files
    files.download(out_path)
except ImportError:
    pass

print(f"\n{'=' * 70}")
print("INTERPRETATION GUIDE")
print("=" * 70)
print("""
If GPU% goes UP with larger docs:
  → GPU is starved for work. Batching multiple docs per call would help.

If GPU% stays LOW even with large docs:
  → CPU-side overhead (Python, data prep) is the bottleneck.
  → Need to overlap CPU prep with GPU compute (async pipeline).

If individual tasks show higher GPU% than full pipeline:
  → The dead time is between tasks (result unpacking, re-encoding).
  → Fusing tasks or pipelining stages would help.

If tokenize alone saturates GPU:
  → The XLM-R forward pass is efficient; overhead is in task heads.
""")
