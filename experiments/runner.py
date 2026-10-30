"""experiments/runner.py — Executes benchmark runs with independent re-scoring and mismatch detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from algorithms.base import SolverResult
from core.cost import evaluate
from core.models import Problem, Timetable
from services.solver_registry import create_solver
from experiments.config import ALGO_BT_BASELINE


@dataclass
class RunRecord:
    """Individual benchmark run result including verified cost scores.

    Attributes:
        algorithm:       Name of the solver algorithm.
        dataset:         Dataset identifier ('small', 'large', etc.).
        seed:            Random seed used.
        status:          Solver termination status ('solved', 'timeout', etc.).
        time:            Wall-clock execution time in seconds.
        nodes_expanded:  Search nodes (backtracking) or iterations (local search).
        hard_violations: Re-scored hard-constraint violations from core/cost.py.
        soft_penalty:    Re-scored soft penalty from core/cost.py.
        cost:            Re-scored total cost (1000 * hard + soft).
        solver_hard:     Hard violations reported by the solver directly.
        solver_soft:     Soft penalty reported by the solver directly.
        mismatch:        True if solver reported scores diverge from core/cost.py.
        phase1_time:     Hybrid Phase 1 runtime (seconds), if applicable.
        phase2_time:     Hybrid Phase 2 runtime (seconds), if applicable.
        soft_before:     Hybrid Phase 1 soft penalty before SA, if applicable.
        soft_after:      Hybrid Phase 2 soft penalty after SA, if applicable.
        cost_history:    Iteration cost snapshots for convergence plots.
        best_cost_history: Iteration best-cost snapshots for SA.
        timetable:       Final Timetable (if produced).
    """
    algorithm: str
    dataset: str
    seed: int
    status: str
    time: float
    nodes_expanded: int
    hard_violations: int
    soft_penalty: int
    cost: int
    solver_hard: int
    solver_soft: int
    mismatch: bool = False
    phase1_time: Optional[float] = None
    phase2_time: Optional[float] = None
    soft_before: Optional[int] = None
    soft_after: Optional[int] = None
    cost_history: List[int] = field(default_factory=list)
    best_cost_history: List[int] = field(default_factory=list)
    timetable: Optional[Timetable] = None

    def to_csv_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation suitable for CSV export."""
        return {
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "seed": self.seed,
            "status": self.status,
            "time": round(self.time, 5),
            "nodes_expanded": self.nodes_expanded,
            "hard_violations": self.hard_violations,
            "soft_penalty": self.soft_penalty,
            "cost": self.cost,
            "phase1_time": round(self.phase1_time, 5) if self.phase1_time is not None else "",
            "phase2_time": round(self.phase2_time, 5) if self.phase2_time is not None else "",
            "soft_before": self.soft_before if self.soft_before is not None else "",
            "soft_after": self.soft_after if self.soft_after is not None else "",
        }


def run_single(
    algo_name: str,
    problem: Problem,
    dataset_name: str,
    seed: int,
    time_limit: float,
) -> RunRecord:
    """Execute one run of *algo_name* on *problem* and verify results with core/cost.py.

    Args:
        algo_name:    Algorithm identifier.
        problem:      Target Problem instance.
        dataset_name: Problem label ('small', 'large', etc.).
        seed:         Random seed.
        time_limit:   Max execution time in seconds.

    Returns:
        A fully validated :class:`RunRecord`.
    """
    solver = create_solver(algo_name, time_limit=time_limit)
    res: SolverResult = solver.solve(problem, seed=seed)

    # 1. Re-score using core/cost.py rather than trusting the solver
    if res.timetable is not None:
        breakdown = evaluate(res.timetable, complete=True)
        rescored_hard = breakdown.total_hard_violations
        rescored_soft = breakdown.total_soft_penalty
        rescored_cost = breakdown.total_cost
    else:
        rescored_hard = res.hard_violations
        rescored_soft = res.soft_penalty
        rescored_cost = res.cost

    # 2. Check for mismatches between solver reported and re-scored numbers
    mismatch = (rescored_hard != res.hard_violations) or (rescored_soft != res.soft_penalty)
    if mismatch:
        print(
            f"  [MISMATCH DETECTED] {algo_name} ({dataset_name}, seed={seed}): "
            f"solver reported (hard={res.hard_violations}, soft={res.soft_penalty}) != "
            f"core/cost.py (hard={rescored_hard}, soft={rescored_soft})"
        )

    # 3. Extract phase metrics for Hybrid
    phase1_t = res.details.get("phase1_time")
    phase2_t = res.details.get("phase2_time")
    soft_b = res.details.get("soft_before")
    soft_a = res.details.get("soft_after")
    best_history = res.details.get("best_cost_history", [])

    return RunRecord(
        algorithm=algo_name,
        dataset=dataset_name,
        seed=seed,
        status=res.status,
        time=res.time_seconds,
        nodes_expanded=res.nodes_expanded,
        hard_violations=rescored_hard,
        soft_penalty=rescored_soft,
        cost=rescored_cost,
        solver_hard=res.hard_violations,
        solver_soft=res.soft_penalty,
        mismatch=mismatch,
        phase1_time=phase1_t,
        phase2_time=phase2_t,
        soft_before=soft_b,
        soft_after=soft_a,
        cost_history=res.cost_history,
        best_cost_history=best_history,
        timetable=res.timetable,
    )


def run_benchmark(
    algo_name: str,
    problem: Problem,
    dataset_name: str,
    seeds: List[int],
    time_limit: float,
) -> List[RunRecord]:
    """Run an algorithm over multiple seeds on a dataset.

    Baseline Backtracking is deterministic and therefore executed only once (seed 0).
    Other algorithms are run for every seed in *seeds*.

    Args:
        algo_name:    Algorithm identifier.
        problem:      Target Problem instance.
        dataset_name: Dataset name ('small', 'large').
        seeds:        List of random seeds to execute.
        time_limit:   Per-run wall-clock time limit in seconds.

    Returns:
        List of :class:`RunRecord` objects.
    """
    # Deterministic baseline runs once
    active_seeds = [seeds[0]] if algo_name == ALGO_BT_BASELINE and seeds else seeds

    records: List[RunRecord] = []
    for s in active_seeds:
        rec = run_single(algo_name, problem, dataset_name, seed=s, time_limit=time_limit)
        records.append(rec)

    return records

