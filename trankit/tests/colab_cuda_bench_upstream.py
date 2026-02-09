# Trankit CUDA Benchmark — upstream master vs adapter-caching branch
# Runs adapter-caching modes FIRST (stacked then cached) so upstream gets the warmed-up GPU advantage.
# Three modes: stacked_adapters → cache_adapters → upstream
# Copy this entire cell into Colab
# First: Runtime > Change runtime type > GPU (T4)

# ── Config ──────────────────────────────────────────────────
EMBEDDING = "xlm-roberta-base"
WARMUP_RUNS = 2
BENCHMARK_RUNS = 10
SWITCHING_LANGUAGES = ["english", "french", "german"]
SWITCHING_WARMUP_ROUNDS = 1   # full round-robin cycles for warmup
SWITCHING_BENCHMARK_ROUNDS = 3  # full round-robin cycles to measure
SWITCHING_ALL_WARMUP_ROUNDS = 1
SWITCHING_ALL_BENCHMARK_ROUNDS = 2  # fewer rounds — 21 langs × rounds gets long on upstream

# ── 1. Check CUDA ────────────────────────────────────────────
import torch
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. Select GPU runtime:")
    print("Runtime > Change runtime type > T4 GPU")
    raise SystemExit(1)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# ── 2. Common setup ──────────────────────────────────────────
import gc
import os
import subprocess
import sys
import time
import statistics
import json
import warnings
import logging


def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", *args])

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*adapters available but none.*")
logging.getLogger("adapters").setLevel(logging.ERROR)

REPORT_PATH = "cuda_upstream_benchmark_report.txt"

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

VALIDATION_TEXT = "John Donovan from Apple Inc. announced a new product today in San Francisco."

SWITCHING_TEXTS = {
    "english": (
        "John Donovan from Apple Inc. announced a new product today in San Francisco. "
        "The device will be available next month."
    ),
    "french": (
        "Le président Emmanuel Macron a annoncé de nouvelles mesures économiques à Paris. "
        "Les changements entreront en vigueur le mois prochain."
    ),
    "german": (
        "Bundeskanzler Olaf Scholz kündigte neue Wirtschaftsreformen in Berlin an. "
        "Die Maßnahmen sollen ab nächstem Monat gelten."
    ),
}

SWITCHING_ALL_LANGUAGES = [
    "catalan", "danish", "german", "greek", "english", "spanish",
    "finnish", "french", "hungarian", "italian", "japanese", "korean",
    "dutch", "polish", "portuguese", "romanian", "russian", "swedish",
    "turkish", "ukrainian", "chinese",
]

SWITCHING_ALL_TEXTS = {
    "catalan": "El president del govern va anunciar noves mesures econòmiques a Barcelona avui.",
    "danish": "Statsministeren præsenterede nye økonomiske reformer i København i dag.",
    "german": "Bundeskanzler Olaf Scholz kündigte neue Wirtschaftsreformen in Berlin an.",
    "greek": "Ο πρωθυπουργός ανακοίνωσε νέα οικονομικά μέτρα στην Αθήνα σήμερα.",
    "english": "John Donovan from Apple Inc. announced a new product today in San Francisco.",
    "spanish": "El presidente del gobierno anunció nuevas medidas económicas en Madrid hoy.",
    "finnish": "Pääministeri esitteli uusia talousuudistuksia Helsingissä tänään.",
    "french": "Le président Emmanuel Macron a annoncé de nouvelles mesures économiques à Paris.",
    "hungarian": "A miniszterelnök új gazdasági reformokat jelentett be Budapesten ma.",
    "italian": "Il presidente del consiglio ha annunciato nuove riforme economiche a Roma oggi.",
    "japanese": "首相は本日東京で新たな経済改革を発表した。",
    "korean": "총리는 오늘 서울에서 새로운 경제 개혁을 발표했다.",
    "dutch": "De premier kondigde vandaag nieuwe economische hervormingen aan in Den Haag.",
    "polish": "Premier ogłosił dziś nowe reformy gospodarcze w Warszawie.",
    "portuguese": "O primeiro-ministro anunciou novas reformas econômicas em Lisboa hoje.",
    "romanian": "Prim-ministrul a anunțat astăzi noi reforme economice la București.",
    "russian": "Премьер-министр объявил о новых экономических реформах в Москве сегодня.",
    "swedish": "Statsministern presenterade nya ekonomiska reformer i Stockholm idag.",
    "turkish": "Başbakan bugün Ankara'da yeni ekonomik reformları açıkladı.",
    "ukrainian": "Прем'єр-міністр оголосив про нові економічні реформи у Києві сьогодні.",
    "chinese": "总理今天在北京宣布了新的经济改革措施。",
}


def count_tokens(result):
    if "sentences" in result:
        return sum(len(s["tokens"]) for s in result["sentences"])
    return len(result.get("tokens", []))


def count_sentences(result):
    return len(result["sentences"]) if "sentences" in result else 1


def gpu_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def flush_trankit_modules():
    for mod in list(sys.modules):
        if mod == 'trankit' or mod.startswith('trankit.'):
            del sys.modules[mod]


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


def run_mode(p, mode_label):
    """Run all benchmark tasks for a pipeline, return list of result dicts."""
    results = []
    print(f"\n--- {mode_label}: Short text ({len(SHORT_TEXT)} chars) ---")
    for label, method, text in TASKS:
        if text is not SHORT_TEXT:
            continue
        fn = getattr(p, method) if method else p
        r = benchmark_task(fn, text, label)
        results.append(r)
        print(format_row(r))

    print(f"\n--- {mode_label}: Long text ({len(LONG_TEXT)} chars) ---")
    for label, method, text in TASKS:
        if text is not LONG_TEXT:
            continue
        fn = getattr(p, method) if method else p
        r = benchmark_task(fn, text, label)
        results.append(r)
        print(format_row(r))

    return results


def run_switching_benchmark(p, languages, texts, task_name, rounds, warmup_rounds):
    """Run round-robin language switching benchmark.

    Each round cycles through supported languages, calling the task once per language.
    Languages that don't support the task (e.g., NER for catalan) are auto-detected
    during the first warmup pass and excluded from timing.
    """
    if task_name == "full pipeline":
        fn = p
    else:
        fn = getattr(p, task_name)

    # First pass: detect which languages support this task (doubles as warmup)
    supported = []
    with torch.inference_mode():
        for lang in languages:
            try:
                p.set_active(lang)
                fn(texts[lang])
                supported.append(lang)
            except (AssertionError, RuntimeError):
                pass

    if not supported:
        return {
            "task": task_name,
            "skipped": True,
            "rounds": rounds,
            "languages": languages,
            "supported_languages": [],
            "round_mean_sec": 0,
            "round_stdev_sec": 0,
            "total_tokens_per_round": 0,
            "tokens_per_sec": 0,
        }

    # Additional warmup rounds with supported languages
    with torch.inference_mode():
        for _ in range(warmup_rounds):
            for lang in supported:
                p.set_active(lang)
                fn(texts[lang])

    # Benchmark
    round_times = []
    total_tokens = 0

    with torch.inference_mode():
        # Count tokens once
        for lang in supported:
            p.set_active(lang)
            result = fn(texts[lang])
            total_tokens += count_tokens(result)

        for _ in range(rounds):
            torch.cuda.synchronize()
            round_start = time.perf_counter()
            for lang in supported:
                p.set_active(lang)
                fn(texts[lang])
            torch.cuda.synchronize()
            round_elapsed = time.perf_counter() - round_start
            round_times.append(round_elapsed)

    mean_round = statistics.mean(round_times)
    return {
        "task": task_name,
        "rounds": rounds,
        "languages": languages,
        "supported_languages": supported,
        "round_mean_sec": round(mean_round, 4),
        "round_stdev_sec": round(statistics.stdev(round_times) if len(round_times) > 1 else 0.0, 4),
        "total_tokens_per_round": total_tokens,
        "tokens_per_sec": round(total_tokens / mean_round if mean_round > 0 else 0, 1),
    }


def run_switching_mode(p, languages, texts, mode_label, rounds, warmup_rounds):
    """Run switching benchmarks for all tasks."""
    results = []
    print(f"\n--- {mode_label}: Language switching ({len(languages)} langs) ---")
    for task_name in ["tokenize", "posdep", "ner", "full pipeline"]:
        r = run_switching_benchmark(
            p, languages, texts, task_name,
            rounds=rounds,
            warmup_rounds=warmup_rounds,
        )
        results.append(r)
        if r.get("skipped"):
            print(f"  {task_name:<20s}  SKIPPED (not supported for any language)")
        else:
            n_sup = len(r.get("supported_languages", languages))
            lang_note = f" ({n_sup}/{len(languages)} langs)" if n_sup < len(languages) else ""
            print(
                f"  {task_name:<20s} "
                f"{r['round_mean_sec']:>8.4f}s/round  "
                f"(+/- {r['round_stdev_sec']:.4f})  "
                f"{r['tokens_per_sec']:>8.1f} tok/s  "
                f"[{r['total_tokens_per_round']} tokens{lang_note}]"
            )
    return results


def add_languages(p, languages):
    """Add languages to pipeline, skipping those already present."""
    for lang in languages:
        if lang != "english":
            p.add(lang)


# ── 3. Header ───────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"Trankit CUDA Benchmark — upstream vs adapter-caching — {EMBEDDING}")
print(f"Warmup: {WARMUP_RUNS} | Runs: {BENCHMARK_RUNS}")
print(f"Note: adapter-caching runs FIRST (stacked, cached); upstream runs last (warmed GPU)")
print(f"{'=' * 70}")

# ── 4a. Install & run adapter-caching (stacked) FIRST ────────
print(f"\n{'=' * 70}")
print("Installing adapter-caching branch...")
print(f"{'=' * 70}")

pip("uninstall", "-y", "trankit", "adapters")
pip("install", "--no-cache-dir", "-q", "--no-deps", "--force-reinstall", "git+https://github.com/joprice/trankit.git@adapter-caching")
pip("install", "--no-cache-dir", "-q", "adapters", "psutil", "langid", "filelock", "tqdm", "requests", "protobuf", "sentencepiece", "sacremoses", "regex", "packaging")

flush_trankit_modules()

os.environ['TRANKIT_BYPASS_ADAPTER_RESET'] = '1'
os.environ['TRANKIT_STRIP_LORA'] = '1'
os.environ['TRANKIT_PATCH_ADAPTER_OVERHEAD'] = '1'

import importlib
import importlib.metadata as _meta
from trankit import Pipeline as StackedPipeline

_opt_ver = _meta.version("trankit")
try:
    _direct_url = json.loads(_meta.distribution("trankit").read_text("direct_url.json"))
    _opt_commit = _direct_url.get("vcs_info", {}).get("commit_id", "unknown")[:10]
except Exception:
    _opt_commit = "unknown"

print(f"\n{'=' * 70}")
print(f"Mode: adapter-caching (stacked_adapters) — runs first")
print(f"trankit: {_opt_ver} commit {_opt_commit}")
print(f"{'=' * 70}")

torch.cuda.empty_cache()
t0 = time.perf_counter()
p_stacked = StackedPipeline("english", gpu=True, cache_dir="./cache", embedding=EMBEDDING,
                             stacked_adapters=True)
init_stacked = time.perf_counter() - t0
print(f"Device: {p_stacked._config.device.type}")
print(f"Pipeline initialized in {init_stacked:.2f}s")
print(f"VRAM after init: {gpu_mb():.0f}MB")

results_stacked = run_mode(p_stacked, "stacked")

# 3-language switching
add_languages(p_stacked, SWITCHING_LANGUAGES)
switching_stacked = run_switching_mode(
    p_stacked, SWITCHING_LANGUAGES, SWITCHING_TEXTS, "stacked",
    rounds=SWITCHING_BENCHMARK_ROUNDS, warmup_rounds=SWITCHING_WARMUP_ROUNDS)

# 21-language switching
add_languages(p_stacked, SWITCHING_ALL_LANGUAGES)
switching_all_stacked = run_switching_mode(
    p_stacked, SWITCHING_ALL_LANGUAGES, SWITCHING_ALL_TEXTS, "stacked (21 langs)",
    rounds=SWITCHING_ALL_BENCHMARK_ROUNDS, warmup_rounds=SWITCHING_ALL_WARMUP_ROUNDS)
p_stacked.set_active("english")

# Save stacked outputs for correctness validation
with torch.inference_mode():
    stacked_outputs = {
        "tokenize": p_stacked.tokenize(VALIDATION_TEXT),
        "posdep": p_stacked.posdep(VALIDATION_TEXT),
        "ner": p_stacked.ner(VALIDATION_TEXT),
        "lemmatize": p_stacked.lemmatize(VALIDATION_TEXT),
        "full": p_stacked(VALIDATION_TEXT),
    }

del p_stacked
del StackedPipeline
torch.cuda.empty_cache()
gc.collect()
flush_trankit_modules()

# ── 4b. Run adapter-caching with cache_adapters (no stacked) ──
print(f"\n{'=' * 70}")
print(f"Mode: adapter-caching (cache_adapters only) — runs second")
print(f"trankit: {_opt_ver} commit {_opt_commit}")
print(f"{'=' * 70}")

from trankit import Pipeline as CachedPipeline

torch.cuda.empty_cache()
t0 = time.perf_counter()
p_cached = CachedPipeline("english", gpu=True, cache_dir="./cache", embedding=EMBEDDING,
                           cache_adapters=True)
init_cached = time.perf_counter() - t0
print(f"Device: {p_cached._config.device.type}")
print(f"Pipeline initialized in {init_cached:.2f}s")
print(f"VRAM after init: {gpu_mb():.0f}MB")

results_cached = run_mode(p_cached, "cached")

# 3-language switching
add_languages(p_cached, SWITCHING_LANGUAGES)
switching_cached = run_switching_mode(
    p_cached, SWITCHING_LANGUAGES, SWITCHING_TEXTS, "cached",
    rounds=SWITCHING_BENCHMARK_ROUNDS, warmup_rounds=SWITCHING_WARMUP_ROUNDS)

# 21-language switching
add_languages(p_cached, SWITCHING_ALL_LANGUAGES)
switching_all_cached = run_switching_mode(
    p_cached, SWITCHING_ALL_LANGUAGES, SWITCHING_ALL_TEXTS, "cached (21 langs)",
    rounds=SWITCHING_ALL_BENCHMARK_ROUNDS, warmup_rounds=SWITCHING_ALL_WARMUP_ROUNDS)
p_cached.set_active("english")

# Save cached outputs for validation
with torch.inference_mode():
    cached_outputs = {
        "tokenize": p_cached.tokenize(VALIDATION_TEXT),
        "posdep": p_cached.posdep(VALIDATION_TEXT),
        "ner": p_cached.ner(VALIDATION_TEXT),
        "lemmatize": p_cached.lemmatize(VALIDATION_TEXT),
        "full": p_cached(VALIDATION_TEXT),
    }

del p_cached
del CachedPipeline
torch.cuda.empty_cache()
gc.collect()
flush_trankit_modules()

# ── 5. Install & run upstream THIRD (warmed GPU) ──────────────
print(f"\n{'=' * 70}")
print("Installing upstream trankit (master)...")
print(f"{'=' * 70}")

pip("uninstall", "-y", "trankit")
pip("install", "--no-cache-dir", "-q", "--no-deps", "git+https://github.com/nlp-uoregon/trankit.git@master")
# six is needed by upstream but not by adapter-caching
pip("install", "--no-cache-dir", "-q", "six")

flush_trankit_modules()

# Patch __init__.py BEFORE importing — upstream's TPipeline uses AdamW
# which was removed from newer transformers. Find the file on disk without importing.
import importlib.util
_spec = importlib.util.find_spec("trankit")
_init_file = _spec.origin  # path to trankit/__init__.py
with open(_init_file, "r") as f:
    _src = f.read()
if "from .tpipeline import TPipeline" in _src:
    _src = _src.replace(
        "from .tpipeline import TPipeline",
        "# TPipeline skipped for benchmarking"
    )
    with open(_init_file, "w") as f:
        f.write(_src)
    print("Patched: skipped TPipeline import")

_upstream_ver = _meta.version("trankit")
try:
    _direct_url = json.loads(_meta.distribution("trankit").read_text("direct_url.json"))
    _upstream_commit = _direct_url.get("vcs_info", {}).get("commit_id", "unknown")[:10]
except Exception:
    _upstream_commit = "unknown"

print(f"\n{'=' * 70}")
print(f"Mode: upstream (master) — runs third (warmed GPU)")
print(f"trankit: {_upstream_ver} commit {_upstream_commit}")
print(f"{'=' * 70}")

torch.cuda.empty_cache()
from trankit import Pipeline as UpstreamPipeline
t0 = time.perf_counter()
p_upstream = UpstreamPipeline("english", gpu=True, cache_dir="./cache", embedding=EMBEDDING)
init_upstream = time.perf_counter() - t0
print(f"Device: cuda")
print(f"Pipeline initialized in {init_upstream:.2f}s")
print(f"VRAM after init: {gpu_mb():.0f}MB")

results_upstream = run_mode(p_upstream, "upstream")

# 3-language switching
add_languages(p_upstream, SWITCHING_LANGUAGES)
switching_upstream = run_switching_mode(
    p_upstream, SWITCHING_LANGUAGES, SWITCHING_TEXTS, "upstream",
    rounds=SWITCHING_BENCHMARK_ROUNDS, warmup_rounds=SWITCHING_WARMUP_ROUNDS)

# 21-language switching
add_languages(p_upstream, SWITCHING_ALL_LANGUAGES)
switching_all_upstream = run_switching_mode(
    p_upstream, SWITCHING_ALL_LANGUAGES, SWITCHING_ALL_TEXTS, "upstream (21 langs)",
    rounds=SWITCHING_ALL_BENCHMARK_ROUNDS, warmup_rounds=SWITCHING_ALL_WARMUP_ROUNDS)
p_upstream.set_active("english")

# Validate outputs match
print(f"\n{'=' * 70}")
print("Output validation")
print(f"{'=' * 70}\n")

with torch.inference_mode():
    upstream_outputs = {
        "tokenize": p_upstream.tokenize(VALIDATION_TEXT),
        "posdep": p_upstream.posdep(VALIDATION_TEXT),
        "ner": p_upstream.ner(VALIDATION_TEXT),
        "lemmatize": p_upstream.lemmatize(VALIDATION_TEXT),
        "full": p_upstream(VALIDATION_TEXT),
    }

all_match = True
print("  stacked vs upstream:")
for task_name in stacked_outputs:
    if stacked_outputs[task_name] == upstream_outputs[task_name]:
        print(f"    {task_name:<12s} PASS")
    else:
        all_match = False
        print(f"    {task_name:<12s} MISMATCH")

cached_match = True
print("  cached vs stacked:")
for task_name in cached_outputs:
    if cached_outputs[task_name] == stacked_outputs[task_name]:
        print(f"    {task_name:<12s} PASS")
    else:
        cached_match = False
        print(f"    {task_name:<12s} MISMATCH")

if all_match and cached_match:
    print("\n  All outputs match across all three modes.")
elif all_match:
    print("\n  Stacked matches upstream. Cached has differences.")
else:
    print("\n  WARNING: Some outputs differ!")

del p_upstream
del UpstreamPipeline
torch.cuda.empty_cache()
gc.collect()

# ── 6. Comparison tables ─────────────────────────────────────
print(f"\n{'=' * 70}")
print("Comparison: upstream vs cached vs stacked (time)")
print(f"{'=' * 70}\n")

time_header = (
    f"  {'Task':<25s} "
    f"{'Upstream':>9s}  "
    f"{'Cached':>9s}  "
    f"{'Stacked':>9s}  "
    f"{'Cache Δ':>8s}  "
    f"{'Stack Δ':>8s}"
)
print(time_header)
print("  " + "-" * (len(time_header) - 2))

for ru, rc, rs in zip(results_upstream, results_cached, results_stacked):
    cache_delta = ((rc["mean_sec"] - ru["mean_sec"]) / ru["mean_sec"] * 100) if ru["mean_sec"] > 0 else 0
    stack_delta = ((rs["mean_sec"] - ru["mean_sec"]) / ru["mean_sec"] * 100) if ru["mean_sec"] > 0 else 0
    print(
        f"  {ru['task']:<25s} "
        f"{ru['mean_sec']:>8.4f}s  "
        f"{rc['mean_sec']:>8.4f}s  "
        f"{rs['mean_sec']:>8.4f}s  "
        f"{'+' if cache_delta >= 0 else ''}{cache_delta:>6.1f}%  "
        f"{'+' if stack_delta >= 0 else ''}{stack_delta:>6.1f}%"
    )

print(f"\n{'=' * 70}")
print("Comparison: upstream vs cached vs stacked (throughput)")
print(f"{'=' * 70}\n")

tps_header = (
    f"  {'Task':<25s} "
    f"{'Upstream':>9s}  "
    f"{'Cached':>9s}  "
    f"{'Stacked':>9s}  "
    f"{'Cache Δ':>8s}  "
    f"{'Stack Δ':>8s}"
)
print(tps_header)
print("  " + "-" * (len(tps_header) - 2))

for ru, rc, rs in zip(results_upstream, results_cached, results_stacked):
    cache_delta = ((rc["tokens_per_sec"] - ru["tokens_per_sec"]) / ru["tokens_per_sec"] * 100) if ru["tokens_per_sec"] > 0 else 0
    stack_delta = ((rs["tokens_per_sec"] - ru["tokens_per_sec"]) / ru["tokens_per_sec"] * 100) if ru["tokens_per_sec"] > 0 else 0
    print(
        f"  {ru['task']:<25s} "
        f"{ru['tokens_per_sec']:>8.1f}  "
        f"{rc['tokens_per_sec']:>8.1f}  "
        f"{rs['tokens_per_sec']:>8.1f}  "
        f"{'+' if cache_delta >= 0 else ''}{cache_delta:>6.1f}%  "
        f"{'+' if stack_delta >= 0 else ''}{stack_delta:>6.1f}%"
    )

print(f"\n  Init time: upstream {init_upstream:.2f}s | cached {init_cached:.2f}s | stacked {init_stacked:.2f}s")

# ── 3-language switching comparison ──
print(f"\n{'=' * 70}")
print(f"Language switching (3 langs): upstream vs cached vs stacked")
print(f"{'=' * 70}\n")

sw_header = (
    f"  {'Task':<20s} "
    f"{'Upstream':>10s}  "
    f"{'Cached':>10s}  "
    f"{'Stacked':>10s}  "
    f"{'Cache Δ':>8s}  "
    f"{'Stack Δ':>8s}"
)
print(sw_header)
print("  " + "-" * (len(sw_header) - 2))

for su, sc, ss in zip(switching_upstream, switching_cached, switching_stacked):
    if su.get("skipped") or sc.get("skipped") or ss.get("skipped"):
        print(f"  {su['task']:<20s}  SKIPPED")
        continue
    cache_delta = ((sc["round_mean_sec"] - su["round_mean_sec"]) / su["round_mean_sec"] * 100) if su["round_mean_sec"] > 0 else 0
    stack_delta = ((ss["round_mean_sec"] - su["round_mean_sec"]) / su["round_mean_sec"] * 100) if su["round_mean_sec"] > 0 else 0
    print(
        f"  {su['task']:<20s} "
        f"{su['round_mean_sec']:>9.4f}s  "
        f"{sc['round_mean_sec']:>9.4f}s  "
        f"{ss['round_mean_sec']:>9.4f}s  "
        f"{'+' if cache_delta >= 0 else ''}{cache_delta:>6.1f}%  "
        f"{'+' if stack_delta >= 0 else ''}{stack_delta:>6.1f}%"
    )

# ── 21-language switching comparison ──
print(f"\n{'=' * 70}")
print(f"Language switching (21 langs): upstream vs cached vs stacked")
print(f"{'=' * 70}\n")

print(sw_header)
print("  " + "-" * (len(sw_header) - 2))

for su, sc, ss in zip(switching_all_upstream, switching_all_cached, switching_all_stacked):
    if su.get("skipped") or sc.get("skipped") or ss.get("skipped"):
        n_sup = len(su.get("supported_languages", []))
        print(f"  {su['task']:<20s}  SKIPPED (0/{len(su.get('languages', []))} langs supported)")
        continue
    n_sup = len(su.get("supported_languages", su.get("languages", [])))
    n_total = len(su.get("languages", []))
    lang_note = f"  [{n_sup}/{n_total} langs]" if n_sup < n_total else ""
    cache_delta = ((sc["round_mean_sec"] - su["round_mean_sec"]) / su["round_mean_sec"] * 100) if su["round_mean_sec"] > 0 else 0
    stack_delta = ((ss["round_mean_sec"] - su["round_mean_sec"]) / su["round_mean_sec"] * 100) if su["round_mean_sec"] > 0 else 0
    print(
        f"  {su['task']:<20s} "
        f"{su['round_mean_sec']:>9.4f}s  "
        f"{sc['round_mean_sec']:>9.4f}s  "
        f"{ss['round_mean_sec']:>9.4f}s  "
        f"{'+' if cache_delta >= 0 else ''}{cache_delta:>6.1f}%  "
        f"{'+' if stack_delta >= 0 else ''}{stack_delta:>6.1f}%"
        f"{lang_note}"
    )

print()

# ── 7. Save ──────────────────────────────────────────────────
gpu_name = torch.cuda.get_device_name(0).replace(" ", "-")
json_path = f"benchmark_upstream_vs_opt_{EMBEDDING.replace('/', '-')}_{gpu_name}.json"
with open(json_path, "w") as f:
    json.dump({
        "embedding": EMBEDDING,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
        "run_order": "stacked first, cached second, upstream third (warmed GPU)",
        "upstream": {
            "version": _upstream_ver,
            "commit": _upstream_commit,
            "init_time_sec": round(init_upstream, 2),
            "results": results_upstream,
        },
        "cached": {
            "version": _opt_ver,
            "commit": _opt_commit,
            "branch": "adapter-caching",
            "cache_adapters": True,
            "stacked_adapters": False,
            "init_time_sec": round(init_cached, 2),
            "results": results_cached,
        },
        "stacked": {
            "version": _opt_ver,
            "commit": _opt_commit,
            "branch": "adapter-caching",
            "stacked_adapters": True,
            "init_time_sec": round(init_stacked, 2),
            "results": results_stacked,
        },
        "switching_3lang": {
            "languages": SWITCHING_LANGUAGES,
            "warmup_rounds": SWITCHING_WARMUP_ROUNDS,
            "benchmark_rounds": SWITCHING_BENCHMARK_ROUNDS,
            "upstream": switching_upstream,
            "cached": switching_cached,
            "stacked": switching_stacked,
        },
        "switching_21lang": {
            "languages": SWITCHING_ALL_LANGUAGES,
            "warmup_rounds": SWITCHING_ALL_WARMUP_ROUNDS,
            "benchmark_rounds": SWITCHING_ALL_BENCHMARK_ROUNDS,
            "upstream": switching_all_upstream,
            "cached": switching_all_cached,
            "stacked": switching_all_stacked,
        },
        "outputs_match": all_match,
        "cached_matches_stacked": cached_match,
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
