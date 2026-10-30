"""scripts/run_min_conflicts.py — CLI for running the Min-Conflicts CSP solver.

Usage
-----
    python scripts/run_min_conflicts.py --data data/small.json --seed 42
    python scripts/run_min_conflicts.py --data data/large.json --seed 0 --runs 10
    python scripts/run_min_conflicts.py --data data/large.json --strategy greedy --runs 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.min_conflicts import MinConflictsConfig, MinConflictsSolver
from core.loader import load_problem_from_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Min-Conflicts local search on a timetable problem."
    )
    p.add_argument("--data", required=True, help="Path to JSON problem file.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    p.add_argument("--runs", type=int, default=1, help="Number of runs with varying seeds (default 1).")
    p.add_argument(
        "--strategy",
        default="random",
        choices=["random", "greedy"],
        help="Initial assignment strategy (default: random).",
    )
    p.add_argument("--p-walk", type=float, default=0.02, help="Random-walk probability (default 0.02).")
    p.add_argument("--restart-threshold", type=int, default=2000, help="Iterations before restart (default 2000).")
    p.add_argument("--max-iterations", type=int, default=50000, help="Max iterations (default 50000).")
    p.add_argument("--time-limit", type=float, default=60.0, help="Time limit in seconds (default 60).")
    p.add_argument("--k-cost-history", type=int, default=1, help="Record cost every k iterations (default 1).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    problem = load_problem_from_json(args.data)

    config = MinConflictsConfig(
        initial_strategy=args.strategy,
        p_walk=args.p_walk,
        restart_threshold=args.restart_threshold,
        max_iterations=args.max_iterations,
        time_limit=args.time_limit,
        k_cost_history=args.k_cost_history,
    )
    solver = MinConflictsSolver(config)

    print(f"\nDataset  : {args.data}")
    print(f"Strategy : {args.strategy}")
    print(f"p_walk   : {args.p_walk}")
    print(f"Restarts : threshold={args.restart_threshold}")
    print(f"Limits   : time={args.time_limit}s, max_iterations={args.max_iterations}")
    print("-" * 75)

    if args.runs == 1:
        res = solver.solve(problem, seed=args.seed)
        print(f"Run (Seed {args.seed}):")
        print(f"  Status          : {res.status}")
        print(f"  Success         : {res.success}")
        print(f"  Time (s)        : {res.time_seconds:.4f}")
        print(f"  Iterations      : {res.nodes_expanded}")
        print(f"  Restarts        : {res.restarts}")
        print(f"  Hard violations : {res.hard_violations}")
        print(f"  Soft penalty    : {res.soft_penalty}")
        print(f"  Cost            : {res.cost}")
        print(f"  Cost history len: {len(res.cost_history)}")
        if res.cost_history:
            mid = len(res.cost_history) // 2
            print(f"  Cost hist (1st) : {res.cost_history[0]}")
            print(f"  Cost hist (mid) : {res.cost_history[mid]}")
            print(f"  Cost hist (end) : {res.cost_history[-1]}")
    else:
        results = []
        header = f"{'Run':>4} | {'Seed':>5} | {'Status':>15} | {'Time (s)':>8} | {'Iters':>7} | {'Restarts':>8} | {'Hard':>4} | {'Soft':>5} | {'Cost':>6}"
        print(header)
        print("-" * len(header))

        for i in range(args.runs):
            s = args.seed + i
            res = solver.solve(problem, seed=s)
            results.append(res)
            print(
                f"{i+1:4d} | {s:5d} | {res.status:>15} | {res.time_seconds:8.4f} | "
                f"{res.nodes_expanded:7d} | {res.restarts:8d} | {res.hard_violations:4d} | "
                f"{res.soft_penalty:5d} | {res.cost:6d}"
            )

        print("-" * len(header))
        n = len(results)
        successes = sum(1 for r in results if r.success)
        avg_time = sum(r.time_seconds for r in results) / n
        avg_iters = sum(r.nodes_expanded for r in results) / n
        avg_restarts = sum(r.restarts for r in results) / n
        avg_hard = sum(r.hard_violations for r in results) / n
        avg_soft = sum(r.soft_penalty for r in results) / n
        avg_cost = sum(r.cost for r in results) / n

        print(f"Summary over {n} runs:")
        print(f"  Success rate     : {successes}/{n} ({successes / n * 100:.1f}%)")
        print(f"  Average time (s) : {avg_time:.4f}")
        print(f"  Average iters    : {avg_iters:.1f}")
        print(f"  Average restarts : {avg_restarts:.2f}")
        print(f"  Average hard     : {avg_hard:.2f}")
        print(f"  Average soft     : {avg_soft:.2f}")
        print(f"  Average cost     : {avg_cost:.2f}")


if __name__ == "__main__":
    main()
