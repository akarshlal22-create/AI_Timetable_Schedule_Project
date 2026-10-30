"""scripts/run_sa.py — CLI for running Simulated Annealing and the Hybrid Solver.

Usage
-----
    python scripts/run_sa.py --data data/small.json --algo sa --seed 42
    python scripts/run_sa.py --data data/small.json --algo hybrid --seed 42
    python scripts/run_sa.py --data data/large.json --algo hybrid --seed 0 --runs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.hybrid import HybridConfig, HybridSolver
from algorithms.simulated_annealing import SimulatedAnnealingConfig, SimulatedAnnealingSolver
from core.loader import load_problem_from_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Simulated Annealing or Hybrid Solver on a timetable problem."
    )
    p.add_argument("--data", required=True, help="Path to JSON problem file.")
    p.add_argument(
        "--algo",
        required=True,
        choices=["sa", "hybrid"],
        help="Algorithm to run: 'sa' (standalone SA) or 'hybrid' (Backtracking -> SA).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    p.add_argument("--runs", type=int, default=1, help="Number of runs with varying seeds (default 1).")
    p.add_argument("--time-limit", type=float, default=60.0, help="Total time limit in seconds (default 60).")
    p.add_argument("--max-iterations", type=int, default=50000, help="Max iterations for SA (default 50000).")
    p.add_argument("--alpha", type=float, default=0.995, help="Geometric cooling alpha (default 0.995).")
    p.add_argument("--t0", type=float, default=None, help="Initial temperature (default: auto-calibrated).")
    p.add_argument("--p-swap", type=float, default=0.5, help="Probability of swap vs move (default 0.5).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    problem = load_problem_from_json(args.data)

    if args.algo == "sa":
        cfg = SimulatedAnnealingConfig(
            time_limit=args.time_limit,
            max_iterations=args.max_iterations,
            alpha=args.alpha,
            t0=args.t0,
            p_swap=args.p_swap,
        )
        solver = SimulatedAnnealingSolver(cfg)
    else:
        cfg = HybridConfig(
            bt_time_limit=args.time_limit / 2,
            sa_time_limit=args.time_limit / 2,
            max_sa_iterations=args.max_iterations,
            alpha=args.alpha,
            t0=args.t0,
            p_swap=args.p_swap,
        )
        solver = HybridSolver(cfg)

    print(f"\nDataset   : {args.data}")
    print(f"Algorithm : {args.algo.upper()}")
    print(f"Alpha     : {args.alpha}")
    print(f"Time Limit: {args.time_limit}s, Max Iters: {args.max_iterations}")
    print("-" * 80)

    if args.runs == 1:
        res = solver.solve(problem, seed=args.seed)
        print(f"Run (Seed {args.seed}):")
        print(f"  Status          : {res.status}")
        print(f"  Success         : {res.success}")
        print(f"  Time (s)        : {res.time_seconds:.4f}")
        print(f"  Nodes / Iters   : {res.nodes_expanded}")
        print(f"  Hard violations : {res.hard_violations}")
        print(f"  Soft penalty    : {res.soft_penalty}")
        print(f"  Total Cost      : {res.cost}")
        if args.algo == "hybrid" and "soft_before" in res.details:
            print(f"  Phase 1 Soft    : {res.details['soft_before']}")
            print(f"  Phase 2 Soft    : {res.details['soft_after']}")
            print(f"  Phase 1 Time (s): {res.details['phase1_time']:.4f}")
            print(f"  Phase 2 Time (s): {res.details['phase2_time']:.4f}")
        if "final_temperature" in res.details:
            print(f"  Final Temp      : {res.details['final_temperature']:.4f}")
        print(f"  Cost history len: {len(res.cost_history)}")
        if res.cost_history:
            mid = len(res.cost_history) // 2
            print(f"  Cost hist (1st) : {res.cost_history[0]}")
            print(f"  Cost hist (mid) : {res.cost_history[mid]}")
            print(f"  Cost hist (end) : {res.cost_history[-1]}")
    else:
        results = []
        if args.algo == "hybrid":
            header = f"{'Run':>4} | {'Seed':>5} | {'Status':>16} | {'Time (s)':>8} | {'Iters':>7} | {'Hard':>4} | {'Soft_0':>6} | {'Soft_F':>6} | {'Cost':>6}"
        else:
            header = f"{'Run':>4} | {'Seed':>5} | {'Status':>16} | {'Time (s)':>8} | {'Iters':>7} | {'Hard':>4} | {'Soft':>6} | {'Cost':>6}"
        print(header)
        print("-" * len(header))

        for i in range(args.runs):
            s = args.seed + i
            res = solver.solve(problem, seed=s)
            results.append(res)
            if args.algo == "hybrid":
                s0 = res.details.get("soft_before", "-")
                sf = res.details.get("soft_after", res.soft_penalty)
                print(
                    f"{i+1:4d} | {s:5d} | {res.status:>16} | {res.time_seconds:8.4f} | "
                    f"{res.nodes_expanded:7d} | {res.hard_violations:4d} | {s0:6} | {sf:6} | {res.cost:6d}"
                )
            else:
                print(
                    f"{i+1:4d} | {s:5d} | {res.status:>16} | {res.time_seconds:8.4f} | "
                    f"{res.nodes_expanded:7d} | {res.hard_violations:4d} | {res.soft_penalty:6d} | {res.cost:6d}"
                )

        print("-" * len(header))
        n = len(results)
        successes = sum(1 for r in results if r.success)
        avg_time = sum(r.time_seconds for r in results) / n
        avg_iters = sum(r.nodes_expanded for r in results) / n
        avg_hard = sum(r.hard_violations for r in results) / n
        avg_soft = sum(r.soft_penalty for r in results) / n
        avg_cost = sum(r.cost for r in results) / n

        print(f"Summary over {n} runs:")
        print(f"  Success rate     : {successes}/{n} ({successes / n * 100:.1f}%)")
        print(f"  Average time (s) : {avg_time:.4f}")
        print(f"  Average iters    : {avg_iters:.1f}")
        print(f"  Average hard     : {avg_hard:.2f}")
        print(f"  Average soft     : {avg_soft:.2f}")
        print(f"  Average cost     : {avg_cost:.2f}")
        if args.algo == "hybrid":
            avg_s0 = sum(r.details.get("soft_before", 0) for r in results) / n
            avg_sf = sum(r.details.get("soft_after", 0) for r in results) / n
            print(f"  Avg Soft Before  : {avg_s0:.2f}")
            print(f"  Avg Soft After   : {avg_sf:.2f}")
            print(f"  Avg Improvement  : {avg_s0 - avg_sf:.2f} penalty points ({((avg_s0 - avg_sf)/avg_s0)*100:.1f}%)")


if __name__ == "__main__":
    main()
