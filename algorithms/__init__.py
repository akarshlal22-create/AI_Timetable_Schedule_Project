"""algorithms/__init__.py — Public re-exports for the algorithms package."""

from algorithms.base import SolverResult, Solver, fill_metrics
from algorithms.backtracking import BacktrackingSolver, BacktrackingConfig
from algorithms.min_conflicts import MinConflictsSolver, MinConflictsConfig
from algorithms.simulated_annealing import SimulatedAnnealingSolver, SimulatedAnnealingConfig
from algorithms.hybrid import HybridSolver, HybridConfig

__all__ = [
    "SolverResult", "Solver", "fill_metrics",
    "BacktrackingSolver", "BacktrackingConfig",
    "MinConflictsSolver", "MinConflictsConfig",
    "SimulatedAnnealingSolver", "SimulatedAnnealingConfig",
    "HybridSolver", "HybridConfig",
]
