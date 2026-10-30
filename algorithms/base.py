"""algorithms/base.py — Shared result type and solver interface.

Every solver (Backtracking, Min-Conflicts, SA, Hybrid) must:
  1. Implement the ``Solver`` protocol (a single ``solve`` method).
  2. Return a ``SolverResult`` populated via ``fill_metrics``.

This keeps the UI and experiment harness algorithm-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from core.cost import evaluate
from core.models import Problem, Timetable


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

StatusLiteral = Literal[
    "solved",
    "no_solution",
    "timeout",
    "node_limit",
    "iteration_limit",
    "stopped_by_limit",
    "frozen",
]


@dataclass
class SolverResult:
    """Unified result returned by every solver.

    Attributes:
        timetable:      Final (possibly partial) timetable, or None.
        success:        True only when ``status == "solved"``.
        status:         One of "solved", "no_solution", "timeout", "node_limit", "iteration_limit", "stopped_by_limit", "frozen".
        time_seconds:   Wall-clock seconds the solver ran.
        nodes_expanded: Assignment attempts (backtracking) or iterations (local).
        hard_violations: Number of hard-constraint violations in the timetable.
        soft_penalty:   Total soft-constraint penalty.
        cost:           ``1000 * hard_violations + soft_penalty``.
        cost_history:   Per-iteration cost snapshots (empty for backtracking).
        restarts:       Number of random restarts performed (local search).
        details:        Optional dictionary for algorithm-specific metrics (e.g. Hybrid phases, SA temperature).
    """
    timetable: Optional[Timetable]
    success: bool
    status: StatusLiteral
    time_seconds: float
    nodes_expanded: int
    hard_violations: int = 0
    soft_penalty: int = 0
    cost: int = 0
    cost_history: List[int] = field(default_factory=list)
    restarts: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line human-readable summary."""
        base_str = (
            f"status={self.status}  time={self.time_seconds:.3f}s  "
            f"nodes={self.nodes_expanded}  hard={self.hard_violations}  "
            f"soft={self.soft_penalty}  cost={self.cost}"
        )
        if self.restarts > 0:
            base_str += f"  restarts={self.restarts}"
        return base_str


# ---------------------------------------------------------------------------
# Metric helper (shared across all solvers)
# ---------------------------------------------------------------------------

def fill_metrics(result: SolverResult) -> SolverResult:
    """Populate ``hard_violations``, ``soft_penalty``, and ``cost`` from the
    timetable stored in *result* by calling :func:`core.cost.evaluate`.

    Pass ``complete=result.success`` so H7/H8 are only checked on full solutions.
    Mutates *result* in place and returns it for convenience.

    Args:
        result: A :class:`SolverResult` whose ``timetable`` field is set.

    Returns:
        The same *result* object with metrics filled in.
    """
    if result.timetable is None:
        result.hard_violations = 0
        result.soft_penalty = 0
        result.cost = 0
        return result

    breakdown = evaluate(result.timetable, complete=result.success)
    result.hard_violations = breakdown.total_hard_violations
    result.soft_penalty = breakdown.total_soft_penalty
    result.cost = breakdown.total_cost
    return result


# ---------------------------------------------------------------------------
# Solver protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Solver(Protocol):
    """Interface every solver must satisfy.

    Usage::

        solver: Solver = BacktrackingSolver.improved()
        result: SolverResult = solver.solve(problem, seed=42)
    """

    def solve(self, problem: Problem, seed: int = 42) -> SolverResult:
        """Run the solver on *problem* and return a :class:`SolverResult`.

        Args:
            problem: A validated :class:`~core.models.Problem`.
            seed:    Random seed for tie-breaking (reproducibility).

        Returns:
            A fully-populated :class:`SolverResult`.
        """
        ...
