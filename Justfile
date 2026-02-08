python := ".venv/bin/python"

# list available recipes
default:
    @just --list

# run correctness checks (gpu)
test:
    {{python}} trankit/tests/test_correctness.py

# run benchmark on gpu (uses mps on mac, cuda otherwise)
bench model="xlm-roberta-base":
    {{python}} trankit/tests/test_benchmark.py {{model}}

# run benchmark on cpu
bench-cpu model="xlm-roberta-base":
    {{python}} trankit/tests/test_benchmark.py {{model}} --cpu

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
