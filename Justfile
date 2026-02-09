python := ".venv/bin/python"
python_upstream := ".venv-upstream/bin/python"

# list available recipes
default:
    @just --list

# run correctness checks (gpu)
test:
    {{python}} trankit/tests/test_correctness.py

# run benchmark on gpu (uses mps on mac, cuda otherwise)
# pass --stacked to use stacked adapters
bench model="xlm-roberta-base" *args="":
    {{python}} trankit/tests/test_benchmark.py {{model}} {{args}}

# run benchmark on cpu
bench-cpu model="xlm-roberta-base" *args="":
    {{python}} trankit/tests/test_benchmark.py {{model}} --cpu {{args}}

# run all benchmark variants for base/large:
# - local CPU
# - local GPU (MPS on Mac, CUDA otherwise)
# - T4 results are produced separately (e.g., Colab) and copied into trankit/tests manually
bench-all:
    set -eu
    just bench-cpu xlm-roberta-base
    just bench-cpu xlm-roberta-large
    just bench xlm-roberta-base
    just bench xlm-roberta-large

bench-gpu:
    set -eu
    just bench xlm-roberta-base
    just bench xlm-roberta-large

# simulate server throughput: N docs of ~W words through the full pipeline
bench-throughput model="xlm-roberta-base" *args="":
    {{python}} trankit/tests/test_benchmark.py {{model}} --throughput {{args}}

# run benchmark under cProfile (only timed runs, excludes warmup/init)
# device: "gpu" (mps/cuda) or "cpu"
# pass out=<path> to override output path
bench-profile model="xlm-roberta-base" device="gpu" out="":
    #!/usr/bin/env bash
    set -eu
    if [ -z "{{out}}" ]; then
      profile_flag="--profile"
    else
      profile_flag="--profile={{out}}"
    fi
    if [ "{{device}}" = "cpu" ]; then
      {{python}} trankit/tests/test_benchmark.py {{model}} --cpu "$profile_flag"
    else
      {{python}} trankit/tests/test_benchmark.py {{model}} "$profile_flag"
    fi

# run cProfile benchmarks for base/large on both cpu and gpu
bench-profile-all:
    set -eu
    just bench-profile xlm-roberta-base cpu
    just bench-profile xlm-roberta-large cpu
    just bench-profile xlm-roberta-base gpu
    just bench-profile xlm-roberta-large gpu

# print top methods from a profile file
bench-profile-report profile="trankit/tests/profiles/bench_xlm-roberta-base_gpu.prof" sort="cumtime" limit="40":
    {{python}} -c "import pstats; s=pstats.Stats('{{profile}}'); s.strip_dirs().sort_stats('{{sort}}').print_stats(int('{{limit}}'))"

# run correctness then benchmark (both gpu)
verify: test (bench)

# compare current benchmark results against a git ref (default: HEAD)
bench-compare ref="HEAD" *args="":
    {{python}} trankit/tests/bench_compare.py {{ref}} {{args}}

# run batch parity tests
test-parity:
    {{python}} trankit/tests/test_batch_parity.py

# run stacked adapter tests
test-stacked:
    {{python}} -m pytest trankit/tests/test_stacked_adapters.py -v

# quick smoke test on cpu
test-cpu:
    TRANKIT_QUIET=1 {{python}} -c " \
    from trankit import Pipeline; \
    p = Pipeline('english', gpu=False, cache_dir='./cache'); \
    r = p('John Donovan from Apple Inc. announced a new product today.'); \
    assert 'sentences' in r and len(r['sentences']) == 1; \
    t = r['sentences'][0]['tokens'][0]; \
    assert all(k in t for k in ['text','upos','lemma','ner']); \
    print(f'CPU correctness: PASS (device={p._config.device})'); \
    "

# create .venv-upstream with upstream trankit master for baseline comparison
setup-upstream:
    #!/usr/bin/env bash
    set -eu
    if [ -d .venv-upstream ]; then
      echo ".venv-upstream already exists, skipping creation"
    else
      echo "Creating .venv-upstream..."
      python3.12 -m venv .venv-upstream
      echo "Installing upstream trankit (master)..."
      .venv-upstream/bin/pip install --no-cache-dir --no-deps \
        git+https://github.com/nlp-uoregon/trankit.git@master
      echo "Installing dependencies..."
      .venv-upstream/bin/pip install --no-cache-dir \
        torch adapters transformers \
        numpy protobuf requests tqdm langid filelock tokenizers \
        regex packaging sentencepiece sacremoses six huggingface_hub
      echo "Patching __init__.py to skip TPipeline import..."
      INIT_FILE=".venv-upstream/lib/python3.12/site-packages/trankit/__init__.py"
      sed -i '' 's/^from .tpipeline import TPipeline/# TPipeline skipped for inference benchmarking/' "$INIT_FILE"
      echo "Done. Verify with: just test-upstream"
    fi

# quick smoke test for upstream venv
test-upstream:
    #!/usr/bin/env bash
    set -eu
    cd /tmp
    {{python_upstream}} -c " \
    import warnings; warnings.filterwarnings('ignore'); \
    from trankit import Pipeline; \
    print('trankit', __import__('trankit').__version__); \
    p = Pipeline('english', gpu=False, cache_dir='/Users/josephprice/dev/trankit/cache'); \
    r = p('John Donovan from Apple Inc. announced a new product today.'); \
    assert 'sentences' in r and len(r['sentences']) == 1; \
    t = r['sentences'][0]['tokens'][0]; \
    assert all(k in t for k in ['text','upos','lemma','ner']); \
    print(f'Upstream CPU correctness: PASS (device={p._config.device})'); \
    "

# run cpu benchmark against upstream master (baseline comparison)
bench-upstream model="xlm-roberta-base":
    #!/usr/bin/env bash
    set -eu
    PROJ="/Users/josephprice/dev/trankit"
    # Copy benchmark script to /tmp so upstream's trankit package is used (not local source)
    cp trankit/tests/test_benchmark.py /tmp/_bench_upstream.py
    # Symlink cache dir so model weights are shared
    ln -sfn "$PROJ/cache" /tmp/cache
    cd /tmp
    "$PROJ"/{{python_upstream}} _bench_upstream.py {{model}} --cpu
    # Copy results back with _upstream suffix
    for f in /tmp/benchmark_results_*.json; do
      [ -f "$f" ] || continue
      base=$(basename "$f" .json)
      cp "$f" "$PROJ/trankit/tests/${base}_upstream.json"
      echo "Saved: trankit/tests/${base}_upstream.json"
    done
