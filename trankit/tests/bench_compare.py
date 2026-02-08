#!/usr/bin/env python3
"""Compare current benchmark results against a git ref and print a report."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent
BENCH_PATTERN = "benchmark_results_{model}_{device}.json"


def load_json(path):
    return json.loads(path.read_text())


def load_git_json(rel_path, ref):
    """Load a JSON file from a git ref."""
    try:
        blob = subprocess.check_output(
            ["git", "show", f"{ref}:{rel_path}"],
            stderr=subprocess.DEVNULL,
        )
        return json.loads(blob)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def fmt_delta(old, new):
    if old == 0:
        return "n/a"
    pct = (new - old) / old * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def print_report(model, device, old_data, new_data):
    old_by_task = {r["task"]: r for r in old_data["results"]}
    new_by_task = {r["task"]: r for r in new_data["results"]}

    header = f"{model} / {device}"
    print(f"\n{'=' * 60}")
    print(f"  {header}")
    print(f"{'=' * 60}")

    # Init time
    old_init = old_data.get("init_time_sec", 0)
    new_init = new_data.get("init_time_sec", 0)
    print(f"  Init time: {old_init:.2f}s -> {new_init:.2f}s ({fmt_delta(old_init, new_init)})")
    print()

    # Table header
    col_task = 24
    col_num = 12
    col_delta = 10
    hdr = (
        f"  {'Task':<{col_task}}"
        f"{'Old tok/s':>{col_num}}"
        f"{'New tok/s':>{col_num}}"
        f"{'Delta':>{col_delta}}"
    )
    print(hdr)
    print(f"  {'-' * (col_task + col_num * 2 + col_delta)}")

    for task_name in old_by_task:
        old_r = old_by_task[task_name]
        new_r = new_by_task.get(task_name)
        if new_r is None:
            continue

        old_tps = old_r["tokens_per_sec"]
        new_tps = new_r["tokens_per_sec"]
        delta = fmt_delta(old_tps, new_tps)

        print(
            f"  {task_name:<{col_task}}"
            f"{old_tps:>{col_num}.1f}"
            f"{new_tps:>{col_num}.1f}"
            f"{delta:>{col_delta}}"
        )

    print()


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results against a git ref")
    parser.add_argument("ref", nargs="?", default="HEAD", help="Git ref to compare against (default: HEAD)")
    parser.add_argument("--device", default=None, help="Device filter (e.g. mps, cpu, cuda_Tesla-T4)")
    parser.add_argument("--model", default=None, help="Model filter (e.g. xlm-roberta-base)")
    args = parser.parse_args()

    # Find all current benchmark files
    bench_files = sorted(BENCH_DIR.glob("benchmark_results_*.json"))
    if not bench_files:
        print("No benchmark result files found.", file=sys.stderr)
        sys.exit(1)

    repo_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )

    found_any = False
    for bench_file in bench_files:
        new_data = load_json(bench_file)
        model = new_data.get("embedding", "unknown")
        device = new_data.get("device", "unknown")

        # Apply filters
        if args.device and args.device not in bench_file.name:
            continue
        if args.model and args.model not in bench_file.name:
            continue

        rel_path = bench_file.relative_to(repo_root)
        old_data = load_git_json(str(rel_path), args.ref)
        if old_data is None:
            print(f"  (no {args.ref} data for {bench_file.name}, skipping)")
            continue

        found_any = True
        print_report(model, device, old_data, new_data)

    if not found_any:
        print(f"No comparable benchmark data found at ref '{args.ref}'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
