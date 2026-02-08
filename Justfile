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

# run correctness then benchmark (both gpu)
verify: test (bench)

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
