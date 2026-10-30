"""experiments/run_all.py — Benchmark execution pipeline entry point.

Usage:
    python -m experiments.run_all
    python -m experiments.run_all --quick
    python -m experiments.run_all --only "Min-Conflicts" --dataset small
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import pandas as pd

from core.loader import load_problem_from_json
from core.models import Problem
from core.problem_builder import build_sessions
from experiments.aggregate import (
    build_comparison_dataframe,
    export_summary_files,
    generate_summary_markdown,
)
from experiments.config import (
    ALGORITHMS,
    ALGO_BT_BASELINE,
    ALGO_BT_IMPROVED,
    ALGO_MIN_CONFLICTS,
    DATASETS,
    DATA_DIR,
    DEFAULT_SEEDS,
    DEFAULT_TIME_LIMIT,
    PLOT_COST_MIN_CONFLICTS,
    PLOT_COST_SA,
    PLOT_HYBRID_BEFORE_AFTER,
    PLOT_SCALING,
    PLOT_TIME_COMPARISON,
    QUICK_SEEDS,
    QUICK_TIME_LIMIT,
    RAW_RUNS_CSV,
    RESULTS_DIR,
    SCALING_ALGORITHMS,
    SCALING_CSV,
    SCALING_SEEDS,
    SCALING_SIZES,
    SCALING_TIME_LIMIT,
    SUMMARY_CSV,
    SUMMARY_MD,
)
from experiments.plots import (
    plot_cost_vs_iterations_min_conflicts,
    plot_cost_vs_iterations_sa,
    plot_hybrid_before_after,
    plot_scaling_experiment,
    plot_time_comparison,
)
from experiments.runner import RunRecord, run_benchmark
from scripts.generate_datasets import build_scaled


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run timetable scheduler experimental benchmarks.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast verification mode (3 seeds, 10s limit, skips scaling).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Run only the specified algorithm name.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["small", "large"],
        default=None,
        help="Run only on 'small' or 'large' dataset.",
    )
    return parser.parse_args()


def run_scaling_experiment(time_limit: float, seeds: List[int]) -> pd.DataFrame:
    """Execute the problem scaling experiment over 4 problem sizes."""
    print("\n" + "=" * 65)
    print("  RUNNING SCALING EXPERIMENT (3, 5, 8, 12 Classes)")
    print("=" * 65)

    scaling_rows = []

    for size in SCALING_SIZES:
        prob = build_scaled(size, seed=42)
        num_sessions = len(build_sessions(prob))
        print(f"\n--- Problem Size: {size} classes ({num_sessions} sessions) ---")

        for algo in SCALING_ALGORITHMS:
            # Baseline is deterministic, run once
            active_seeds = [seeds[0]] if algo == ALGO_BT_BASELINE and seeds else seeds
            sys.stdout.write(f"  {algo:46} ... ")
            sys.stdout.flush()

            records = run_benchmark(
                algo_name=algo,
                problem=prob,
                dataset_name=f"scaled_{size}",
                seeds=active_seeds,
                time_limit=time_limit,
            )

            succ_runs = [r for r in records if r.hard_violations == 0 and r.status == "solved"]
            active_for_avg = succ_runs if succ_runs else records
            mean_t = sum(r.time for r in active_for_avg) / len(active_for_avg)
            mean_nodes = sum(r.nodes_expanded for r in records) / len(records)
            succ_rate = len(succ_runs) / len(records)
            status = "solved" if succ_runs else records[0].status

            scaling_rows.append({
                "classes": size,
                "sessions": num_sessions,
                "algorithm": algo,
                "mean_time": round(mean_t, 5),
                "mean_nodes": round(mean_nodes, 1),
                "success_rate": round(succ_rate, 2),
                "status": status,
            })

            sys.stdout.write(f"done: {mean_t:.4f}s | {len(succ_runs)}/{len(records)} solved\n")
            sys.stdout.flush()

    df_scaling = pd.DataFrame(scaling_rows)
    df_scaling.to_csv(SCALING_CSV, index=False)
    plot_scaling_experiment(df_scaling, PLOT_SCALING)
    print(f"  Scaling CSV saved -> {SCALING_CSV}")
    print(f"  Scaling plot saved -> {PLOT_SCALING}")
    return df_scaling


def main() -> None:
    start_wall_time = time.time()
    args = parse_arguments()

    # Ensure output results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Configure active settings
    time_limit = QUICK_TIME_LIMIT if args.quick else DEFAULT_TIME_LIMIT
    seeds = QUICK_SEEDS if args.quick else DEFAULT_SEEDS

    target_algos = [args.only] if args.only else ALGORITHMS
    target_datasets = [args.dataset] if args.dataset else DATASETS

    print("=" * 65)
    print("  AI TIMETABLE SCHEDULER: EXPERIMENTAL BENCHMARK")
    print("=" * 65)
    print(f"Mode:          {'QUICK' if args.quick else 'STANDARD'}")
    print(f"Time limit:    {time_limit}s per run")
    print(f"Seeds:         {seeds} ({len(seeds)} runs; baseline run once)")
    print(f"Datasets:      {target_datasets}")
    print(f"Algorithms:    {len(target_algos)} configured")
    print("=" * 65 + "\n")

    all_records: List[RunRecord] = []

    for d_name in target_datasets:
        data_path = DATA_DIR / f"{d_name}.json"
        if not data_path.exists():
            print(f"Error: {data_path} not found.")
            sys.exit(1)

        prob = load_problem_from_json(data_path)
        print(f"\n[Dataset: {d_name.upper()} ({len(prob.classes)} classes, {len(prob.rooms)} rooms)]")

        for algo in target_algos:
            # Check runs count label
            n_runs = 1 if algo == ALGO_BT_BASELINE else len(seeds)
            sys.stdout.write(f"  Running {algo:<46} ({n_runs} run{'s' if n_runs > 1 else ''}) ... ")
            sys.stdout.flush()

            t0 = time.time()
            records = run_benchmark(
                algo_name=algo,
                problem=prob,
                dataset_name=d_name,
                seeds=seeds,
                time_limit=time_limit,
            )
            elapsed = time.time() - t0
            all_records.extend(records)

            succ_count = sum(1 for r in records if r.hard_violations == 0 and r.status == "solved")
            sys.stdout.write(f"done in {elapsed:.2f}s ({succ_count}/{n_runs} feasible)\n")
            sys.stdout.flush()

    # 1. Export raw runs CSV
    raw_df = pd.DataFrame([r.to_csv_dict() for r in all_records])
    raw_df.to_csv(RAW_RUNS_CSV, index=False)
    print(f"\nSaved raw runs -> {RAW_RUNS_CSV}")

    # 2. Export summary comparison CSV and Markdown
    df_small, df_large = export_summary_files(
        records=all_records,
        csv_path=str(SUMMARY_CSV),
        md_path=str(SUMMARY_MD),
        datasets=target_datasets,
    )
    print(f"Saved summary CSV -> {SUMMARY_CSV}")
    print(f"Saved summary Markdown -> {SUMMARY_MD}")

    # 3. Generate all visualizations
    plot_cost_vs_iterations_min_conflicts(all_records, PLOT_COST_MIN_CONFLICTS)
    plot_cost_vs_iterations_sa(all_records, PLOT_COST_SA)
    plot_hybrid_before_after(all_records, PLOT_HYBRID_BEFORE_AFTER)
    plot_time_comparison(all_records, PLOT_TIME_COMPARISON)
    print(f"Generated plots -> {RESULTS_DIR}")

    # 4. Run Scaling experiment if not --quick and running all algorithms
    if not args.quick and args.only is None and args.dataset is None:
        run_scaling_experiment(time_limit=SCALING_TIME_LIMIT, seeds=SCALING_SEEDS)

    total_elapsed = time.time() - start_wall_time
    print(f"\nTotal benchmark execution time: {total_elapsed:.2f} seconds.")

    # 5. Print summary tables to console
    print("\n" + "=" * 65)
    print("  SUMMARY COMPARISON TABLE")
    print("=" * 65)
    if df_small is not None:
        print("\n--- SMALL DATASET ---")
        print(df_small.to_string())
    if df_large is not None:
        print("\n--- LARGE DATASET ---")
        print(df_large.to_string())
    print("\n" + "=" * 65 + "\n")


if __name__ == "__main__":
    main()

