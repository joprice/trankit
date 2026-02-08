# Trankit CUDA Verification - Copy this entire cell into Colab
# First: Runtime > Change runtime type > GPU (T4)

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
!pip install --no-cache-dir -q --no-deps git+https://github.com/joprice/trankit.git@adapter-caching
!pip install --no-cache-dir -q adapters psutil langid filelock tqdm requests protobuf sentencepiece sacremoses regex packaging

# ── 3. Setup ─────────────────────────────────────────────────
import time
import statistics
import json
from trankit import Pipeline

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

WARMUP = 2
RUNS = 10

def gpu_mb():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2

# ── 4. Initialize ────────────────────────────────────────────
print("\n" + "=" * 70)
print("INITIALIZING PIPELINE")
print("=" * 70)
torch.cuda.empty_cache()

t0 = time.perf_counter()
p = Pipeline("english", gpu=True, cache_dir="./cache")
init_time = time.perf_counter() - t0
device_type = str(p._config.device.type)
print(f"Device: {device_type}")
print(f"Init time: {init_time:.1f}s")
print(f"VRAM: {gpu_mb():.0f}MB")

assert device_type == "cuda", f"Expected cuda, got {device_type}"

# ── 5. Correctness ───────────────────────────────────────────
print("\n" + "=" * 70)
print("CORRECTNESS CHECKS")
print("=" * 70)
failures = []

def check(name, fn, validate):
    """Run fn, pass result to validate. validate should return True or raise."""
    try:
        result = fn()
        ok = validate(result)
        if ok:
            print(f"  PASS: {name}")
        else:
            print(f"  FAIL: {name} - validation returned False")
            failures.append(name)
    except Exception as e:
        print(f"  FAIL: {name} - {e}")
        failures.append(name)

with torch.inference_mode():
    # full pipeline - document
    def val_full_doc(r):
        assert "sentences" in r, "missing sentences"
        assert len(r["sentences"]) == 25, f"expected 25 sentences, got {len(r['sentences'])}"
        s0 = r["sentences"][0]
        assert "tokens" in s0, "missing tokens in sentence"
        t0 = s0["tokens"][0]
        for key in ["text", "upos", "xpos", "feats", "head", "deprel", "lemma", "ner"]:
            assert key in t0, f"missing '{key}' in token"
        return True
    check("full pipeline (doc)", lambda: p(LONG_TEXT), val_full_doc)

    # full pipeline - short doc
    def val_full_short(r):
        assert len(r["sentences"]) == 2
        tokens = [t["text"] for s in r["sentences"] for t in s["tokens"]]
        assert "John" in tokens and "Donovan" in tokens
        return True
    check("full pipeline (short)", lambda: p(SHORT_TEXT), val_full_short)

    # single sentence
    def val_sent(r):
        assert "tokens" in r
        assert len(r["tokens"]) > 0
        t0 = r["tokens"][0]
        for key in ["text", "upos", "lemma", "ner"]:
            assert key in t0, f"missing '{key}'"
        return True
    check("full pipeline (sent)", lambda: p(SHORT_TEXT, is_sent=True), val_sent)

    # pretokenized
    def val_pretok(r):
        assert len(r["sentences"]) == 2
        s0_words = [t["text"] for t in r["sentences"][0]["tokens"]]
        assert s0_words == ["John", "likes", "cats"]
        return True
    check("pretokenized", lambda: p([["John", "likes", "cats"], ["Mary", "likes", "dogs"]]), val_pretok)

    # pretokenized single sentence
    def val_pretok_sent(r):
        words = [t["text"] for t in r["tokens"]]
        assert words == ["John", "likes", "cats"]
        return True
    check("pretokenized (sent)", lambda: p(["John", "likes", "cats"], is_sent=True), val_pretok_sent)

    # individual tasks
    def val_tokenize(r):
        assert "sentences" in r
        assert len(r["sentences"]) == 2
        return True
    check("tokenize", lambda: p.tokenize(SHORT_TEXT), val_tokenize)

    def val_posdep(r):
        t0 = r["sentences"][0]["tokens"][0]
        assert "upos" in t0 and "head" in t0 and "deprel" in t0
        return True
    check("posdep", lambda: p.posdep(SHORT_TEXT), val_posdep)

    def val_lemma(r):
        t0 = r["sentences"][0]["tokens"][0]
        assert "lemma" in t0
        return True
    check("lemmatize", lambda: p.lemmatize(SHORT_TEXT), val_lemma)

    def val_ner(r):
        t0 = r["sentences"][0]["tokens"][0]
        assert "ner" in t0
        # Should find at least one named entity in "John Donovan from Apple Inc."
        ner_tags = [t["ner"] for s in r["sentences"] for t in s["tokens"]]
        assert any(tag != "O" for tag in ner_tags), "no entities found"
        return True
    check("ner", lambda: p.ner(SHORT_TEXT), val_ner)

if failures:
    print(f"\n{len(failures)} check(s) FAILED: {failures}")
else:
    print("\nAll correctness checks passed.")

# ── 6. Benchmark ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("BENCHMARK")
print("=" * 70)

def count_tokens(result):
    if "sentences" in result:
        return sum(len(s["tokens"]) for s in result["sentences"])
    return len(result.get("tokens", []))

def count_sentences(result):
    if "sentences" in result:
        return len(result["sentences"])
    return 1

def bench(fn, text, label):
    with torch.inference_mode():
        for _ in range(WARMUP):
            result = fn(text)
        torch.cuda.synchronize()

        ntok = count_tokens(result)
        nsent = count_sentences(result)

        times = []
        for _ in range(RUNS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(text)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    mean_t = statistics.mean(times)
    stdev_t = statistics.stdev(times)
    median_t = statistics.median(times)
    tok_s = ntok / mean_t
    sent_s = nsent / mean_t
    print(f"  {label:<25s} {mean_t:>8.4f}s (+/- {stdev_t:.4f})  "
          f"{tok_s:>8.1f} tok/s  {sent_s:>6.2f} sent/s  [{ntok} tok, {nsent} sent]")
    return {
        "task": label,
        "runs": RUNS,
        "num_tokens": ntok,
        "num_sentences": nsent,
        "mean_sec": round(mean_t, 4),
        "median_sec": round(median_t, 4),
        "stdev_sec": round(stdev_t, 4),
        "min_sec": round(min(times), 4),
        "max_sec": round(max(times), 4),
        "tokens_per_sec": round(tok_s, 1),
        "sents_per_sec": round(sent_s, 2),
    }

bench_results = []
for label_suffix, text in [("short", SHORT_TEXT), ("long", LONG_TEXT)]:
    print(f"\n--- {label_suffix} text ({len(text)} chars) ---")
    for task_label, fn in [
        ("tokenize", p.tokenize),
        ("posdep", p.posdep),
        ("lemmatize", p.lemmatize),
        ("ner", p.ner),
        ("full pipeline", p),
    ]:
        bench_results.append(bench(fn, text, f"{task_label} ({label_suffix})"))

# ── 7. Save results ──────────────────────────────────────────
gpu_name = torch.cuda.get_device_name(0).replace(" ", "-")
out = {
    "embedding": "xlm-roberta-base",
    "device": device_type,
    "gpu_name": torch.cuda.get_device_name(0),
    "vram_mb": round(gpu_mb()),
    "warmup_runs": WARMUP,
    "benchmark_runs": RUNS,
    "init_time_sec": round(init_time, 2),
    "correctness": {
        "passed": len(failures) == 0,
        "failures": failures,
    },
    "results": bench_results,
}

out_path = f"benchmark_results_xlm-roberta-base_{device_type}_{gpu_name}.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved to {out_path}")

# In Colab, download the file automatically
try:
    from google.colab import files
    files.download(out_path)
except ImportError:
    pass

# ── 8. Summary ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM used: {gpu_mb():.0f}MB")
print(f"Init time: {init_time:.1f}s")
full_long = next(r for r in bench_results if r["task"] == "full pipeline (long)")
print(f"Full pipeline throughput (long): {full_long['tokens_per_sec']:.0f} tok/s")
if failures:
    print(f"\nFAILED CHECKS: {failures}")
else:
    print("\nAll checks passed.")
