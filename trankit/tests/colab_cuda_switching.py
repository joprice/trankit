# Trankit CUDA Multi-Language Switching Benchmark — adapter-caching branch
# Compares cached vs stacked adapter modes when switching between languages.
# Copy this entire cell into Colab
# First: Runtime > Change runtime type > GPU (T4)

# ── Config ──────────────────────────────────────────────────
EMBEDDING = "xlm-roberta-base"
LANGUAGES = ["english", "french", "german"]  # languages to cycle through
FP16 = False
CPU_LEMMA = False
WARMUP_ROUNDS = 2    # full round-robin cycles for warmup
BENCHMARK_ROUNDS = 5  # full round-robin cycles to measure
BYPASS_ADAPTER_RESET = True
STRIP_LORA = True
PATCH_ADAPTER_OVERHEAD = True
PIN_MEMORY = None

import os
import subprocess
import sys

os.environ['TRANKIT_BYPASS_ADAPTER_RESET'] = '1' if BYPASS_ADAPTER_RESET else '0'
os.environ['TRANKIT_STRIP_LORA'] = '1' if STRIP_LORA else '0'
os.environ['TRANKIT_PATCH_ADAPTER_OVERHEAD'] = '1' if PATCH_ADAPTER_OVERHEAD else '0'


def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", *args])


# ── 1. Check CUDA ────────────────────────────────────────────
import torch
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. Select GPU runtime:")
    print("Runtime > Change runtime type > T4 GPU")
    raise SystemExit(1)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# ── 2. Install ───────────────────────────────────────────────
pip("uninstall", "-y", "trankit", "adapters")
pip("install", "--no-cache-dir", "-q", "--no-deps", "--force-reinstall", "git+https://github.com/joprice/trankit.git@adapter-caching")
pip("install", "--no-cache-dir", "-q", "adapters", "psutil", "langid", "filelock", "tqdm", "requests", "protobuf", "sentencepiece", "sacremoses", "regex", "packaging")

# ── 3. Setup ─────────────────────────────────────────────────
for _mod in list(sys.modules):
    if _mod == 'trankit' or _mod.startswith('trankit.'):
        del sys.modules[_mod]

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*adapters available but none.*")

import logging
logging.getLogger("adapters").setLevel(logging.ERROR)

import gc
import time
import statistics
import json
from trankit import Pipeline

REPORT_PATH = "cuda_switching_report.txt"

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

# Sample texts per language — short and long variants
TEXTS = {
    "english": {
        "short": (
            "John Donovan from Apple Inc. announced a new product today in San Francisco. "
            "The device will be available next month."
        ),
        "long": (
            "The United Nations Secretary-General called on world leaders to take immediate "
            "action on climate change during a summit in New York. Scientists from the "
            "Intergovernmental Panel on Climate Change presented new findings showing that "
            "global temperatures have risen faster than previously predicted. Several nations, "
            "including France, Germany, and Japan, pledged to reduce carbon emissions by fifty "
            "percent over the next decade. Environmental groups welcomed the commitments but "
            "warned that concrete policy changes are needed to meet the targets. Meanwhile, "
            "the European Central Bank announced new green finance initiatives to support "
            "sustainable development across member states.\n\n"
        ) * 3,
    },
    "french": {
        "short": (
            "Le président Emmanuel Macron a annoncé de nouvelles mesures économiques à Paris. "
            "Les changements entreront en vigueur le mois prochain."
        ),
        "long": (
            "Le sommet européen de Bruxelles a réuni les dirigeants des vingt-sept États "
            "membres pour discuter de la politique énergétique commune. La Commission "
            "européenne a présenté un plan ambitieux visant à réduire la dépendance aux "
            "combustibles fossiles. L'Allemagne et la France ont proposé un fonds commun "
            "pour financer la transition énergétique. Les pays du sud de l'Europe ont "
            "exprimé des réserves quant au calendrier proposé. Le Premier ministre italien "
            "a souligné la nécessité d'une approche plus progressive pour les économies "
            "les plus fragiles.\n\n"
        ) * 3,
    },
    "german": {
        "short": (
            "Bundeskanzler Olaf Scholz kündigte neue Wirtschaftsreformen in Berlin an. "
            "Die Maßnahmen sollen ab nächstem Monat gelten."
        ),
        "long": (
            "Die Bundesregierung hat ein umfassendes Klimaschutzpaket vorgestellt, das "
            "weitreichende Änderungen in der Energiepolitik vorsieht. Bundesumweltministerin "
            "Steffi Lemke betonte die Notwendigkeit schnellen Handelns angesichts der "
            "steigenden Temperaturen. Der Verband der Automobilindustrie reagierte "
            "zurückhaltend auf die geplanten Verschärfungen der Emissionsgrenzwerte. "
            "Mehrere Bundesländer forderten zusätzliche Mittel für den Ausbau erneuerbarer "
            "Energien. Die Opposition kritisierte das Paket als unzureichend und verwies "
            "auf die Erfahrungen anderer europäischer Länder.\n\n"
        ) * 3,
    },
}


def count_tokens(result):
    if "sentences" in result:
        return sum(len(s["tokens"]) for s in result["sentences"])
    return len(result.get("tokens", []))


def gpu_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def run_switching_benchmark(p, languages, text_key, task_name, rounds,
                            warmup_rounds):
    """Run round-robin language switching benchmark.

    Each round cycles through all languages, calling the task once per language.
    Returns per-language results and aggregate switching stats.
    """
    if task_name == "full pipeline":
        fns = {lang: p for lang in languages}
    else:
        fns = {lang: getattr(p, task_name) for lang in languages}

    # Warmup: cycle through all languages
    with torch.inference_mode():
        for _ in range(warmup_rounds):
            for lang in languages:
                p.set_active(lang)
                fns[lang](TEXTS[lang][text_key])

    # Benchmark
    per_lang_times = {lang: [] for lang in languages}
    round_times = []

    with torch.inference_mode():
        for _ in range(rounds):
            torch.cuda.synchronize()
            round_start = time.perf_counter()
            for lang in languages:
                p.set_active(lang)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fns[lang](TEXTS[lang][text_key])
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                per_lang_times[lang].append(elapsed)
            round_elapsed = time.perf_counter() - round_start
            round_times.append(round_elapsed)

    # Compute per-language stats
    per_lang_results = {}
    for lang in languages:
        times = per_lang_times[lang]
        mean_t = statistics.mean(times)
        text = TEXTS[lang][text_key]
        # Run once more to get token count
        with torch.inference_mode():
            p.set_active(lang)
            result = fns[lang](text)
        num_tokens = count_tokens(result)
        per_lang_results[lang] = {
            "mean_sec": round(mean_t, 4),
            "stdev_sec": round(statistics.stdev(times) if len(times) > 1 else 0.0, 4),
            "tokens_per_sec": round(num_tokens / mean_t if mean_t > 0 else 0, 1),
            "num_tokens": num_tokens,
        }

    # Aggregate round stats (includes switching overhead)
    total_tokens = sum(r["num_tokens"] for r in per_lang_results.values())
    mean_round = statistics.mean(round_times)
    return {
        "task": task_name,
        "text_key": text_key,
        "rounds": rounds,
        "languages": languages,
        "per_lang": per_lang_results,
        "round_mean_sec": round(mean_round, 4),
        "round_stdev_sec": round(statistics.stdev(round_times) if len(round_times) > 1 else 0.0, 4),
        "total_tokens_per_round": total_tokens,
        "tokens_per_sec": round(total_tokens / mean_round if mean_round > 0 else 0, 1),
    }


def run_all_tasks(p, languages, mode_label):
    """Run switching benchmarks for all task/text combinations."""
    results = []
    tasks = ["tokenize", "posdep", "ner", "full pipeline"]

    for text_key in ["short", "long"]:
        print(f"\n--- {mode_label}: {text_key} text, {len(languages)} languages ---")
        for task_name in tasks:
            r = run_switching_benchmark(
                p, languages, text_key, task_name,
                rounds=BENCHMARK_ROUNDS, warmup_rounds=WARMUP_ROUNDS,
            )
            results.append(r)
            # Print per-round aggregate
            print(
                f"  {task_name:<20s} "
                f"{r['round_mean_sec']:>8.4f}s/round  "
                f"(+/- {r['round_stdev_sec']:.4f})  "
                f"{r['tokens_per_sec']:>8.1f} tok/s  "
                f"[{r['total_tokens_per_round']} tokens across {len(languages)} langs]"
            )
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
print(f"Trankit CUDA Switching Benchmark — {EMBEDDING} (adapter-caching)")
print(f"trankit: {_trankit_ver} commit {_commit}")
print(f"Languages: {LANGUAGES}")
print(f"Warmup rounds: {WARMUP_ROUNDS} | Benchmark rounds: {BENCHMARK_ROUNDS}")
print(f"fp16: {FP16} | cpu_lemma: {CPU_LEMMA}")
print(f"bypass_adapter_reset: {BYPASS_ADAPTER_RESET} | strip_lora: {STRIP_LORA} | patch_adapter_overhead: {PATCH_ADAPTER_OVERHEAD}")
print(f"pin_memory: {PIN_MEMORY}")
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
for lang in LANGUAGES:
    if lang != "english":
        p_cached.add(lang)
init_cached = time.perf_counter() - t0
print(f"Device: {p_cached._config.device.type}")
print(f"Pipeline initialized in {init_cached:.2f}s ({len(LANGUAGES)} languages)")
print(f"VRAM after init: {gpu_mb():.0f}MB")

results_cached = run_all_tasks(p_cached, LANGUAGES, "cached")

del p_cached
torch.cuda.empty_cache()
gc.collect()

# ── 6. Run stacked mode ─────────────────────────────────────
print(f"\n{'=' * 70}")
print("Mode: stacked_adapters")
print(f"{'=' * 70}")

torch.cuda.empty_cache()
t0 = time.perf_counter()
p_stacked = Pipeline("english", gpu=True, cache_dir="./cache", embedding=EMBEDDING,
                      fp16=FP16, cpu_lemma=CPU_LEMMA, stacked_adapters=True,
                      pin_memory=PIN_MEMORY)
for lang in LANGUAGES:
    if lang != "english":
        p_stacked.add(lang)
init_stacked = time.perf_counter() - t0
print(f"Device: {p_stacked._config.device.type}")
print(f"Pipeline initialized in {init_stacked:.2f}s ({len(LANGUAGES)} languages)")
print(f"VRAM after init: {gpu_mb():.0f}MB")

results_stacked = run_all_tasks(p_stacked, LANGUAGES, "stacked")

del p_stacked
torch.cuda.empty_cache()
gc.collect()

# ── 7. Comparison table ──────────────────────────────────────
print(f"\n{'=' * 70}")
print("Comparison: stacked vs cached (round-robin switching)")
print(f"{'=' * 70}")

for text_key in ["short", "long"]:
    rc_group = [r for r in results_cached if r["text_key"] == text_key]
    rs_group = [r for r in results_stacked if r["text_key"] == text_key]

    print(f"\n  --- {text_key} text ---")
    header = (
        f"  {'Task':<20s} "
        f"{'Cached':>10s}  "
        f"{'Stacked':>10s}  "
        f"{'Delta':>8s}  "
        f"{'Cached tok/s':>12s}  "
        f"{'Stacked tok/s':>13s}  "
        f"{'Delta':>8s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rc, rs in zip(rc_group, rs_group):
        time_delta = ((rs["round_mean_sec"] - rc["round_mean_sec"])
                      / rc["round_mean_sec"] * 100) if rc["round_mean_sec"] > 0 else 0
        tps_delta = ((rs["tokens_per_sec"] - rc["tokens_per_sec"])
                     / rc["tokens_per_sec"] * 100) if rc["tokens_per_sec"] > 0 else 0
        t_sign = "+" if time_delta >= 0 else ""
        s_sign = "+" if tps_delta >= 0 else ""
        print(
            f"  {rc['task']:<20s} "
            f"{rc['round_mean_sec']:>9.4f}s  "
            f"{rs['round_mean_sec']:>9.4f}s  "
            f"{t_sign}{time_delta:>6.1f}%  "
            f"{rc['tokens_per_sec']:>11.1f}  "
            f"{rs['tokens_per_sec']:>12.1f}  "
            f"{s_sign}{tps_delta:>6.1f}%"
        )

# Per-language breakdown for full pipeline
print(f"\n{'=' * 70}")
print("Per-language breakdown: full pipeline")
print(f"{'=' * 70}")

for text_key in ["short", "long"]:
    rc_fp = next(r for r in results_cached
                 if r["text_key"] == text_key and r["task"] == "full pipeline")
    rs_fp = next(r for r in results_stacked
                 if r["text_key"] == text_key and r["task"] == "full pipeline")

    print(f"\n  --- {text_key} text ---")
    header = (
        f"  {'Language':<12s} "
        f"{'Cached':>9s}  "
        f"{'Stacked':>9s}  "
        f"{'Delta':>8s}  "
        f"{'Cached tok/s':>12s}  "
        f"{'Stacked tok/s':>13s}  "
        f"{'Delta':>8s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for lang in LANGUAGES:
        lc = rc_fp["per_lang"][lang]
        ls = rs_fp["per_lang"][lang]
        time_delta = ((ls["mean_sec"] - lc["mean_sec"])
                      / lc["mean_sec"] * 100) if lc["mean_sec"] > 0 else 0
        tps_delta = ((ls["tokens_per_sec"] - lc["tokens_per_sec"])
                     / lc["tokens_per_sec"] * 100) if lc["tokens_per_sec"] > 0 else 0
        t_sign = "+" if time_delta >= 0 else ""
        s_sign = "+" if tps_delta >= 0 else ""
        print(
            f"  {lang:<12s} "
            f"{lc['mean_sec']:>8.4f}s  "
            f"{ls['mean_sec']:>8.4f}s  "
            f"{t_sign}{time_delta:>6.1f}%  "
            f"{lc['tokens_per_sec']:>11.1f}  "
            f"{ls['tokens_per_sec']:>12.1f}  "
            f"{s_sign}{tps_delta:>6.1f}%"
        )

print()

# ── 8. Save ──────────────────────────────────────────────────
gpu_name = torch.cuda.get_device_name(0).replace(" ", "-")
json_path = f"switching_results_{EMBEDDING.replace('/', '-')}_{gpu_name}.json"
with open(json_path, "w") as f:
    json.dump({
        "embedding": EMBEDDING,
        "branch": "adapter-caching",
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "languages": LANGUAGES,
        "fp16": FP16,
        "cpu_lemma": CPU_LEMMA,
        "bypass_adapter_reset": BYPASS_ADAPTER_RESET,
        "strip_lora": STRIP_LORA,
        "patch_adapter_overhead": PATCH_ADAPTER_OVERHEAD,
        "pin_memory": PIN_MEMORY,
        "warmup_rounds": WARMUP_ROUNDS,
        "benchmark_rounds": BENCHMARK_ROUNDS,
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
