"""scripts/run_backtracking.py — CLI for running the backtracking solver.

Usage
-----
    python scripts/run_backtracking.py --data data/small.json --mode improved
    python scripts/run_backtracking.py --data data/large.json --mode baseline
    python scripts/run_backtracking.py --data data/small.json --mode custom \\
        --mrv --degree --no-fc --time-limit 30

Modes
-----
  baseline   All heuristics off (may be very slow on large datasets).
  improved   MRV + Degree + Forward Checking (recommended).
  custom     Build config from individual flags (--mrv, --degree, --fc, --lcv).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.backtracking import BacktrackingConfig, BacktrackingSolver
from core.loader import load_problem_from_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the Backtracking CSP solver on a timetable problem."
    )
    p.add_argument("--data",       required=True,  help="Path to JSON problem file.")
    p.add_argument("--mode",       default="improved",
                   choices=["baseline", "improved", "custom"],
                   help="Solver preset (default: improved).")
    p.add_argument("--seed",       type=int, default=42, help="Random seed.")
    p.add_argument("--time-limit", type=float, default=60.0,
                   help="Wall-clock time limit in seconds (default 60).")
    p.add_argument("--node-limit", type=int, default=None,
                   help="Max nodes expanded (default: unlimited).")
    # Custom flags (only used when --mode custom)
    p.add_argument("--mrv",    action="store_true")
    p.add_argument("--degree", action="store_true")
    p.add_argument("--fc",     action="store_true", help="Forward checking.")
    p.add_argument("--lcv",    action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    problem = load_problem_from_json(args.data)

    if args.mode == "baseline":
        solver = BacktrackingSolver(BacktrackingConfig(
            time_limit=args.time_limit,
            node_limit=args.node_limit,
        ))
    elif args.mode == "improved":
        solver = BacktrackingSolver(BacktrackingConfig(
            use_mrv=True, use_degree=True, use_forward_checking=True,
            time_limit=args.time_limit,
            node_limit=args.node_limit,
        ))
    else:  # custom
        solver = BacktrackingSolver(BacktrackingConfig(
            use_mrv=args.mrv,
            use_degree=args.degree,
            use_forward_checking=args.fc,
            use_lcv=args.lcv,
            time_limit=args.time_limit,
            node_limit=args.node_limit,
        ))

    print(f"\nDataset : {args.data}")
    print(f"Mode    : {args.mode}")
    print(f"Seed    : {args.seed}")
    print(f"Limits  : time={args.time_limit}s, nodes={args.node_limit or 'unlimited'}")
    print("-" * 60)

    result = solver.solve(problem, seed=args.seed)

    print(f"Status          : {result.status}")
    print(f"Time (s)        : {result.time_seconds:.4f}")
    print(f"Nodes expanded  : {result.nodes_expanded}")
    print(f"Hard violations : {result.hard_violations}")
    print(f"Soft penalty    : {result.soft_penalty}")
    print(f"Cost            : {result.cost}")

    if result.success:
        sessions_scheduled = len(result.timetable.assignment) if result.timetable else 0
        print(f"Sessions placed : {sessions_scheduled}")
    else:
        print("(no complete solution found)")


if __name__ == "__main__":
    main()
