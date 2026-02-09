# Trankit CUDA Benchmark for Colab T4 — adapter-caching branch
# Runs both cached and stacked-adapter modes, then prints a comparison table.
# Copy this entire cell into Colab
# First: Runtime > Change runtime type > GPU (T4)

# ── Config ──────────────────────────────────────────────────
EMBEDDING = "xlm-roberta-base"
FP16 = False         # Set to True to enable autocast (experimental)
CPU_LEMMA = False    # Set to True to move seq2seq lemma decoder to CPU
WARMUP_RUNS = 2
BENCHMARK_RUNS = 10
PROFILE = False      # Set to True to collect cProfile stats
TORCH_PROFILE = False  # Set to True to collect torch.profiler traces
TORCH_PROFILE_TASKS = ["full pipeline (long)"]  # Task labels to profile; [] means all
TORCH_PROFILE_RECORD_SHAPES = True
TORCH_PROFILE_WITH_STACK = False
TORCH_PROFILE_WAIT = 1
TORCH_PROFILE_WARMUP = 1
TORCH_PROFILE_ACTIVE = 2
TORCH_PROFILE_REPEAT = 1
BYPASS_ADAPTER_RESET = True  # Set to False to benchmark without the bypass
STRIP_LORA = True            # Set to False to keep LoRA wrappers (no-op overhead)
PATCH_ADAPTER_OVERHEAD = True  # Set to False to skip adapter composition monkey-patches
PIN_MEMORY = None              # None = auto (True on CUDA), set False to test without

import os
os.environ['TRANKIT_BYPASS_ADAPTER_RESET'] = '1' if BYPASS_ADAPTER_RESET else '0'
os.environ['TRANKIT_STRIP_LORA'] = '1' if STRIP_LORA else '0'
os.environ['TRANKIT_PATCH_ADAPTER_OVERHEAD'] = '1' if PATCH_ADAPTER_OVERHEAD else '0'

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
# Flush stale trankit modules from previous Colab cell runs so
# Python reimports from the freshly pip-installed files on disk.
import sys
for _mod in list(sys.modules):
    if _mod == 'trankit' or _mod.startswith('trankit.'):
        del sys.modules[_mod]

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*adapters available but none.*")

import logging
logging.getLogger("adapters").setLevel(logging.ERROR)

import cProfile
import io
import math
import pstats
import time
import statistics
import json
from torch.profiler import profile as torch_profile, ProfilerActivity
from torch.profiler import schedule as torch_profiler_schedule
from torch.autograd.profiler import record_function
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


def benchmark_task(fn, text, label, runs=BENCHMARK_RUNS, warmup=WARMUP_RUNS,
                   profiler=None):
    with torch.inference_mode():
        for _ in range(warmup):
            result = fn(text)

        num_tokens = count_tokens(result)
        num_sentences = count_sentences(result)

        times = []
        if profiler is not None:
            profiler.enable()
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn(text)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        if profiler is not None:
            profiler.disable()

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


TASKS = [
    ("tokenize (short)", "tokenize", SHORT_TEXT),
    ("posdep (short)", "posdep", SHORT_TEXT),
    ("lemmatize (short)", "lemmatize", SHORT_TEXT),
    ("ner (short)", "ner", SHORT_TEXT),
    ("full pipeline (short)", None, SHORT_TEXT),
    ("tokenize (long)", "tokenize", LONG_TEXT),
    ("posdep (long)", "posdep", LONG_TEXT),
    ("lemmatize (long)", "lemmatize", LONG_TEXT),
    ("ner (long)", "ner", LONG_TEXT),
    ("full pipeline (long)", None, LONG_TEXT),
]


def run_mode(p, mode_label, profiler=None):
    """Run all benchmark tasks for a pipeline, return list of result dicts."""
    results = []
    print(f"\n--- {mode_label}: Short text ({len(SHORT_TEXT)} chars) ---")
    for label, method, text in TASKS:
        if text is not SHORT_TEXT:
            continue
        fn = getattr(p, method) if method else p
        r = benchmark_task(fn, text, label, profiler=profiler)
        results.append(r)
        print(format_row(r))

    print(f"\n--- {mode_label}: Long text ({len(LONG_TEXT)} chars) ---")
    for label, method, text in TASKS:
        if text is not LONG_TEXT:
            continue
        fn = getattr(p, method) if method else p
        r = benchmark_task(fn, text, label, profiler=profiler)
        results.append(r)
        print(format_row(r))

    return results


# ── 4. Header ───────────────────────────────────────────────
print(f"\n{'=' * 70}")
import importlib.metadata as _meta
_trankit_ver = _meta.version("trankit")
try:
    _direct_url = json.loads(_meta.distribution("trankit").read_text("direct_url.json"))
    _commit = _direct_url.get("vcs_info", {}).get("commit_id", "unknown")[:10]
except Exception:
    _commit = "unknown"
print(f"Trankit CUDA Benchmark — {EMBEDDING} (adapter-caching)")
print(f"trankit: {_trankit_ver} commit {_commit}")
print(f"Warmup: {WARMUP_RUNS} | Runs: {BENCHMARK_RUNS}")
print(f"fp16: {FP16} | cpu_lemma: {CPU_LEMMA}")
print(f"bypass_adapter_reset: {BYPASS_ADAPTER_RESET} | strip_lora: {STRIP_LORA} | patch_adapter_overhead: {PATCH_ADAPTER_OVERHEAD}")
print(f"pin_memory: {PIN_MEMORY}")
print(f"profile: {PROFILE}")
print(f"{'=' * 70}")

# ── 5. Run cached mode ──────────────────────────────────────
print(f"\n{'=' * 70}")
print("Mode: cache_adapters (baseline)")
print(f"{'=' * 70}")

torch.cuda.empty_cache()
t0 = time.perf_counter()
p_cached = Pipeline("english", gpu=True, cache_dir="./cache", embedding=EMBEDDING,
                     fp16=FP16, cpu_lemma=CPU_LEMMA, cache_adapters=True,
                     pin_memory=PIN_MEMORY)
init_cached = time.perf_counter() - t0
print(f"Device: {p_cached._config.device.type}")
print(f"Pipeline initialized in {init_cached:.2f}s")
print(f"VRAM after init: {gpu_mb():.0f}MB")

profiler_cached = cProfile.Profile() if PROFILE else None
results_cached = run_mode(p_cached, "cached", profiler=profiler_cached)

# Free cached pipeline
del p_cached
torch.cuda.empty_cache()
import gc; gc.collect()

# ── 6. Run stacked mode ─────────────────────────────────────
print(f"\n{'=' * 70}")
print("Mode: stacked_adapters")
print(f"{'=' * 70}")

torch.cuda.empty_cache()
t0 = time.perf_counter()
p_stacked = Pipeline("english", gpu=True, cache_dir="./cache", embedding=EMBEDDING,
                      fp16=FP16, cpu_lemma=CPU_LEMMA, stacked_adapters=True,
                      pin_memory=PIN_MEMORY)
init_stacked = time.perf_counter() - t0
print(f"Device: {p_stacked._config.device.type}")
print(f"Pipeline initialized in {init_stacked:.2f}s")
print(f"VRAM after init: {gpu_mb():.0f}MB")

profiler_stacked = cProfile.Profile() if PROFILE else None
results_stacked = run_mode(p_stacked, "stacked", profiler=profiler_stacked)

del p_stacked
torch.cuda.empty_cache()
gc.collect()

# ── 7. Comparison table ──────────────────────────────────────
print(f"\n{'=' * 70}")
print("Comparison: stacked vs cached")
print(f"{'=' * 70}\n")

header = (
    f"  {'Task':<25s} "
    f"{'Cached':>9s}  "
    f"{'Stacked':>9s}  "
    f"{'Delta':>8s}  "
    f"{'Cached tok/s':>12s}  "
    f"{'Stacked tok/s':>13s}  "
    f"{'Delta':>8s}"
)
print(header)
print("  " + "-" * (len(header) - 2))

for rc, rs in zip(results_cached, results_stacked):
    time_delta_pct = ((rs["mean_sec"] - rc["mean_sec"]) / rc["mean_sec"] * 100) if rc["mean_sec"] > 0 else 0
    tps_delta_pct = ((rs["tokens_per_sec"] - rc["tokens_per_sec"]) / rc["tokens_per_sec"] * 100) if rc["tokens_per_sec"] > 0 else 0
    time_sign = "+" if time_delta_pct >= 0 else ""
    tps_sign = "+" if tps_delta_pct >= 0 else ""
    print(
        f"  {rc['task']:<25s} "
        f"{rc['mean_sec']:>8.4f}s  "
        f"{rs['mean_sec']:>8.4f}s  "
        f"{time_sign}{time_delta_pct:>6.1f}%  "
        f"{rc['tokens_per_sec']:>11.1f}  "
        f"{rs['tokens_per_sec']:>12.1f}  "
        f"{tps_sign}{tps_delta_pct:>6.1f}%"
    )

print()

# ── 8. Profile output ───────────────────────────────────────
for label, prof in [("cached", profiler_cached), ("stacked", profiler_stacked)]:
    if prof is not None:
        print(f"\n{'=' * 70}")
        print(f"cProfile ({label}) — top 40 by cumulative time")
        print(f"{'=' * 70}\n")
        stream = io.StringIO()
        ps = pstats.Stats(prof, stream=stream)
        ps.strip_dirs().sort_stats("cumtime").print_stats(40)
        print(stream.getvalue())

# ── 9. Save ──────────────────────────────────────────────────
gpu_name = torch.cuda.get_device_name(0).replace(" ", "-")
suffix = "_cpulemma" if CPU_LEMMA else ""
json_path = f"benchmark_results_{EMBEDDING.replace('/', '-')}_{gpu_name}{suffix}.json"
with open(json_path, "w") as f:
    json.dump({
        "embedding": EMBEDDING,
        "branch": "adapter-caching",
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "fp16": FP16,
        "cpu_lemma": CPU_LEMMA,
        "bypass_adapter_reset": BYPASS_ADAPTER_RESET,
        "strip_lora": STRIP_LORA,
        "patch_adapter_overhead": PATCH_ADAPTER_OVERHEAD,
        "pin_memory": PIN_MEMORY,
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
        "init_time_cached_sec": round(init_cached, 2),
        "init_time_stacked_sec": round(init_stacked, 2),
        "results_cached": results_cached,
        "results_stacked": results_stacked,
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
