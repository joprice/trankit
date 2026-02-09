# Trankit Batch Tokenize Benchmark for Colab T4
# Compares merged vs per-doc tokenizer GPU passes
# Copy this entire cell into Colab
# First: Runtime > Change runtime type > GPU (T4)

# ── Config ──────────────────────────────────────────────────
EMBEDDING = "xlm-roberta-large"
CACHE_ADAPTERS = True
STACKED_ADAPTERS = True
NUM_DOCS = 100
WARMUP = 5
BATCH_SIZE = 20

# Test matrix: (label, target_words, task)
# Short docs maximize overhead-to-compute ratio;
# tokenize-only isolates the stage we're optimizing
SCENARIOS = [
    ("short/tokenize", 30, "tokenize"),
    ("short/full",     30, "full"),
    ("med/tokenize",  100, "tokenize"),
    ("med/full",      100, "full"),
    ("long/tokenize", 300, "tokenize"),
    ("long/full",     300, "full"),
]

# ── 1. Check CUDA ────────────────────────────────────────────
import torch
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. Select GPU runtime:")
    print("Runtime > Change runtime type > T4 GPU")
    raise SystemExit(1)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# ── 2. Install ───────────────────────────────────────────────
!apt-get install -y -qq git > /dev/null 2>&1
!pip uninstall -y trankit adapters 2>/dev/null
!pip install --no-cache-dir -q --no-deps --force-reinstall git+https://github.com/joprice/trankit.git@adapter-caching
!pip install --no-cache-dir -q adapters psutil langid filelock tqdm requests protobuf sentencepiece sacremoses regex packaging

# ── 2b. Print installed commit hash ──────────────────────────
import importlib.metadata, pathlib, json as _json
try:
    _dist = importlib.metadata.distribution("trankit")
    _du = pathlib.Path(str(_dist._path)) / "direct_url.json"
    if _du.exists():
        _info = _json.loads(_du.read_text()).get("vcs_info", {})
        print(f"trankit commit: {_info.get('commit_id', '?')[:10]}")
        print(f"trankit branch: {_info.get('requested_revision', '?')}")
    else:
        print(f"trankit version: {_dist.version}")
except Exception as e:
    print(f"trankit version: unknown ({e})")

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

REPORT_PATH = "batch_tokenize_benchmark_report.txt"


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
    reps = max(1, math.ceil(target_words / _WORDS_PER_PARA))
    doc = _PARAGRAPH * reps
    # Trim to roughly target_words
    words = doc.split()
    if len(words) > target_words:
        doc = " ".join(words[:target_words]) + "."
    return doc


def count_tokens(result):
    if isinstance(result, list):
        return 0
    if "sentences" in result:
        return sum(len(s["tokens"]) for s in result["sentences"])
    return len(result.get("tokens", []))


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
            self._stop.wait(0.25)

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


def run_scenario(p, doc_text, task, mode, num_docs, batch_size, warmup):
    """Run a single scenario and return timing dict."""
    docs_list = [doc_text] * num_docs

    def run_batch(chunk):
        if mode == "serial":
            if task == "tokenize":
                return [p.tokenize(d) for d in chunk]
            else:
                return [p(d) for d in chunk]
        elif mode == "merged":
            if task == "tokenize":
                return p.tokenize_batch(chunk)
            else:
                return p.batch(chunk, batch_tokenize=True)
        else:  # per-doc
            if task == "tokenize":
                return [p.tokenize(d) for d in chunk]
            else:
                return p.batch(chunk, batch_tokenize=False)

    # Warmup
    with torch.inference_mode():
        for _ in range(warmup):
            run_batch(docs_list[:min(batch_size, 5)])
        torch.cuda.synchronize()

    # Timed run
    monitor = GpuMonitor()
    times = []

    with torch.inference_mode():
        monitor.start()
        for i in range(0, num_docs, batch_size):
            chunk = docs_list[i:i + batch_size]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            results = run_batch(chunk)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            per_doc = elapsed / len(chunk)
            times.extend([per_doc] * len(chunk))
        monitor.stop()

    # Count tokens from a single result
    with torch.inference_mode():
        if task == "tokenize":
            sample = p.tokenize(doc_text)
        else:
            sample = p(doc_text)
    num_tokens = count_tokens(sample)

    total = sum(times)
    sorted_t = sorted(times)
    gs = monitor.summary()

    return {
        "total_sec": round(total, 2),
        "docs_per_sec": round(num_docs / total, 2),
        "tokens_per_sec": round(num_tokens * num_docs / total, 1),
        "mean_ms": round(statistics.mean(times) * 1000, 1),
        "p95_ms": round(percentile(sorted_t, 95) * 1000, 1),
        "min_ms": round(min(times) * 1000, 1),
        "max_ms": round(max(times) * 1000, 1),
        "gpu_mean_pct": gs.get("gpu_mean_pct", 0),
        "gpu_max_pct": gs.get("gpu_max_pct", 0),
        "mem_max_mb": gs.get("mem_max_mb", 0),
        "num_tokens": num_tokens,
    }


# ── 4. Initialize ────────────────────────────────────────────
print(f"{'=' * 70}")
print(f"Trankit Batch Tokenize Benchmark")
print(f"Embedding: {EMBEDDING}")
print(f"cache_adapters: {CACHE_ADAPTERS}  stacked_adapters: {STACKED_ADAPTERS}")
print(f"Documents: {NUM_DOCS} | Batch size: {BATCH_SIZE} | Warmup: {WARMUP}")
print(f"Scenarios: {len(SCENARIOS)}")
print(f"{'=' * 70}\n")

torch.cuda.empty_cache()
p = Pipeline("english", gpu=True, cache_dir="./cache", embedding=EMBEDDING,
             cache_adapters=CACHE_ADAPTERS, stacked_adapters=STACKED_ADAPTERS)
print(f"Device: {p._config.device}")
print(f"VRAM after init: {gpu_mb():.0f}MB\n")

# ── 5. Run scenarios ─────────────────────────────────────────
all_results = []

header = (f"{'Scenario':<20s} {'Mode':<10s} {'docs/s':>7s} {'tok/s':>8s} "
          f"{'mean':>8s} {'p95':>8s} {'GPU%':>6s} {'VRAM':>7s}")
print(header)
print("-" * len(header))

ALL_MODES = ["merged", "per-doc", "serial"]

for label, target_words, task in SCENARIOS:
    doc_text = make_document(target_words)
    actual_words = len(doc_text.split())

    # "per-doc" is identical to "serial" for tokenize-only tasks — skip it
    modes = [m for m in ALL_MODES if not (task == "tokenize" and m == "per-doc")]

    for mode in modes:
        r = run_scenario(p, doc_text, task, mode,
                         NUM_DOCS, BATCH_SIZE, WARMUP)
        r["scenario"] = label
        r["mode"] = mode
        r["target_words"] = target_words
        r["actual_words"] = actual_words
        r["task"] = task
        all_results.append(r)

        print(f"{label:<20s} {mode:<10s} {r['docs_per_sec']:>7.1f} {r['tokens_per_sec']:>8.0f} "
              f"{r['mean_ms']:>7.0f}ms {r['p95_ms']:>7.0f}ms "
              f"{r['gpu_mean_pct']:>5.0f}% {r['mem_max_mb']:>6d}MB")

# ── 6. Summary ───────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("SPEEDUP: merged vs serial")
print("=" * 70)

for label, target_words, task in SCENARIOS:
    merged = next(r for r in all_results if r["scenario"] == label and r["mode"] == "merged")
    serial = next(r for r in all_results if r["scenario"] == label and r["mode"] == "serial")
    if serial["docs_per_sec"] > 0:
        speedup = merged["docs_per_sec"] / serial["docs_per_sec"]
    else:
        speedup = 0
    delta_ms = serial["mean_ms"] - merged["mean_ms"]
    print(f"  {label:<20s}  {merged['docs_per_sec']:>5.1f} vs {serial['docs_per_sec']:>5.1f} docs/s  "
          f"speedup={speedup:.2f}x  delta={delta_ms:+.1f}ms/doc")

print(f"\n{'=' * 70}")

# ── 6b. Stage-level profiling for long/full ──────────────────
print("\nSTAGE PROFILING: long/full merged (wall-clock time per stage)")
print("=" * 70)

profile_doc = make_document(300)
profile_docs = [profile_doc] * NUM_DOCS

# Warmup
with torch.inference_mode():
    for _ in range(3):
        batch_process(p, profile_docs[:5], batch_tokenize=True)
    torch.cuda.synchronize()

# Timed runs — collect stage times across batches
stage_accum = {}
profile_times = []

with torch.inference_mode():
    for i in range(0, NUM_DOCS, BATCH_SIZE):
        chunk = profile_docs[i:i + BATCH_SIZE]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch_process(p, chunk, batch_tokenize=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        profile_times.append(elapsed)

        # Accumulate stage times from the pipeline
        if hasattr(p, '_last_batch_stage_times'):
            for stage, dt in p._last_batch_stage_times.items():
                stage_accum[stage] = stage_accum.get(stage, 0.0) + dt

total_wall = sum(profile_times)
total_docs = NUM_DOCS

print(f"  Total: {total_wall:.2f}s  ({total_docs / total_wall:.1f} docs/s)")
print(f"  Per doc: {total_wall / total_docs * 1000:.1f}ms")
print()

if stage_accum:
    for stage in ['tokenize', 'tagger', 'lemma', 'ner']:
        if stage in stage_accum:
            st = stage_accum[stage]
            pct = st / total_wall * 100
            per_doc = st / total_docs * 1000
            print(f"  {stage:<12s}  {st:>6.2f}s  {pct:>5.1f}%  {per_doc:>6.1f}ms/doc")
    overhead = total_wall - sum(stage_accum.values())
    if overhead > 0.01:
        print(f"  {'overhead':<12s}  {overhead:>6.2f}s  {overhead / total_wall * 100:>5.1f}%  {overhead / total_docs * 1000:>6.1f}ms/doc")

print(f"\n{'=' * 70}")

# ── 6c. Batch size sweep for long/full merged ─────────────────
print("\nBATCH SIZE SWEEP: long/full merged (TRANKIT_EVAL_BATCH_SIZE)")
print("=" * 70)

sweep_doc = make_document(300)
sweep_docs = [sweep_doc] * NUM_DOCS
SWEEP_BATCH_SIZES = [8, 16, 24, 32]

sweep_header = (f"{'EvalBS':>7s} {'docs/s':>7s} {'mean':>8s} {'p95':>8s} "
                f"{'GPU%':>6s} {'VRAM':>7s}")
print(sweep_header)
print("-" * len(sweep_header))

sweep_results = []

for ebs in SWEEP_BATCH_SIZES:
    os.environ['TRANKIT_EVAL_BATCH_SIZE'] = str(ebs)
    # Reload the module-level override
    import trankit.batch_pipeline as _bp
    _bp._EVAL_BATCH_SIZE_OVERRIDE = ebs

    # Warmup
    with torch.inference_mode():
        for _ in range(3):
            _bp.batch_process(p, sweep_docs[:5], batch_tokenize=True)
        torch.cuda.synchronize()

    monitor = GpuMonitor()
    times = []

    with torch.inference_mode():
        monitor.start()
        for i in range(0, NUM_DOCS, BATCH_SIZE):
            chunk = sweep_docs[i:i + BATCH_SIZE]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _bp.batch_process(p, chunk, batch_tokenize=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            per_doc = elapsed / len(chunk)
            times.extend([per_doc] * len(chunk))
        monitor.stop()

    total = sum(times)
    sorted_t = sorted(times)
    gs = monitor.summary()

    r = {
        "eval_batch_size": ebs,
        "docs_per_sec": round(NUM_DOCS / total, 2),
        "mean_ms": round(statistics.mean(times) * 1000, 1),
        "p95_ms": round(percentile(sorted_t, 95) * 1000, 1),
        "gpu_mean_pct": gs.get("gpu_mean_pct", 0),
        "mem_max_mb": gs.get("mem_max_mb", 0),
    }
    sweep_results.append(r)

    print(f"{ebs:>7d} {r['docs_per_sec']:>7.1f} {r['mean_ms']:>7.0f}ms "
          f"{r['p95_ms']:>7.0f}ms {r['gpu_mean_pct']:>5.0f}% {r['mem_max_mb']:>6d}MB")

# Restore default
os.environ.pop('TRANKIT_EVAL_BATCH_SIZE', None)
_bp._EVAL_BATCH_SIZE_OVERRIDE = None

print(f"\n{'=' * 70}")

# ── 7. Save ──────────────────────────────────────────────────
gpu_name = torch.cuda.get_device_name(0).replace(" ", "-")
json_path = f"batch_tokenize_benchmark_{EMBEDDING.replace('/', '-')}_{gpu_name}.json"
with open(json_path, "w") as f:
    json.dump({
        "embedding": EMBEDDING,
        "gpu": torch.cuda.get_device_name(0),
        "cache_adapters": CACHE_ADAPTERS,
        "stacked_adapters": STACKED_ADAPTERS,
        "num_docs": NUM_DOCS,
        "batch_size": BATCH_SIZE,
        "warmup": WARMUP,
        "results": all_results,
        "batch_size_sweep": sweep_results,
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
