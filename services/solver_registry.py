"""services/solver_registry.py — Maps algorithm display names to solver instances."""

from __future__ import annotations

from typing import Callable, Dict, List

from algorithms.backtracking import BacktrackingConfig, BacktrackingSolver
from algorithms.base import Solver
from algorithms.hybrid import HybridConfig, HybridSolver
from algorithms.min_conflicts import MinConflictsConfig, MinConflictsSolver
from algorithms.simulated_annealing import SimulatedAnnealingConfig, SimulatedAnnealingSolver

ALGORITHM_OPTIONS: List[str] = [
    "Backtracking (baseline)",
    "Backtracking (MRV + Degree + Forward Checking)",
    "Min-Conflicts",
    "Simulated Annealing",
    "Hybrid (Backtracking + SA)",
]


def create_solver(
    algo_name: str,
    time_limit: float = 60.0,
) -> Solver:
    """Build a configured Solver instance corresponding to *algo_name*.

    Args:
        algo_name:  One of the keys in :data:`ALGORITHM_OPTIONS`.
        time_limit: Maximum wall-clock execution time in seconds.

    Returns:
        An instance conforming to :class:`~algorithms.base.Solver`.

    Raises:
        ValueError: If *algo_name* is not recognized.
    """
    if algo_name == "Backtracking (baseline)":
        return BacktrackingSolver(BacktrackingConfig(time_limit=time_limit))

    if algo_name == "Backtracking (MRV + Degree + Forward Checking)":
        return BacktrackingSolver(
            BacktrackingConfig(
                use_mrv=True,
                use_degree=True,
                use_forward_checking=True,
                time_limit=time_limit,
            )
        )

    if algo_name == "Min-Conflicts":
        return MinConflictsSolver(MinConflictsConfig(time_limit=time_limit))

    if algo_name == "Simulated Annealing":
        return SimulatedAnnealingSolver(SimulatedAnnealingConfig(time_limit=time_limit))

    if algo_name == "Hybrid (Backtracking + SA)":
        half_limit = max(1.0, time_limit / 2.0)
        return HybridSolver(
            HybridConfig(
                bt_time_limit=half_limit,
                sa_time_limit=half_limit,
            )
        )

    raise ValueError(f"Unknown algorithm name: '{algo_name}'")
