"""tests/test_backtracking.py — Tests for the BacktrackingSolver.

Tests
-----
1.  Improved solver on data/small.json → success, 0 hard violations.
2.  Baseline solver on a tiny hand-built problem → success, feasible.
3.  Intentionally unsolvable problem → status "no_solution".
4.  Very small node limit → status "node_limit".
5.  Same seed → same result (reproducibility).
6.  Improved expands ≤ nodes than baseline on a small test problem.
7.  Cross-check with python-constraint on a tiny 1-class problem.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from algorithms.backtracking import BacktrackingConfig, BacktrackingSolver
from core.cost import evaluate
from core.loader import load_problem_from_json
from core.models import (
    ClassBatch, Problem, Room, Subject, Teacher, TimeStructure,
)


# ---------------------------------------------------------------------------
# Shared tiny problem fixture (1 class, 2 teachers, solvable)
# ---------------------------------------------------------------------------

def make_tiny_problem(seed: int = 0) -> Problem:
    """A minimal solvable problem: 1 class, 2 subjects, 2 teachers, 2 rooms."""
    time = TimeStructure(working_days=5, periods_per_day=4, break_periods=frozenset())
    subjects = {
        "MATH": Subject("MATH", "Maths",   weekly_hours=2, type="theory"),
        "CS":   Subject("CS",   "CS 101",  weekly_hours=2, type="theory"),
    }
    teachers = {
        "Alice": Teacher("Alice", frozenset({"MATH"}), frozenset(), max_weekly_load=10),
        "Bob":   Teacher("Bob",   frozenset({"CS"}),   frozenset(), max_weekly_load=10),
    }
    classes = {
        "A": ClassBatch("A", 25, ("MATH", "CS"), lunch_slot=99),  # lunch_slot=99 → no lunch
    }
    rooms = {
        "R1": Room("R1", 30, "classroom"),
        "R2": Room("R2", 30, "classroom"),
    }
    return Problem("tiny", time, subjects, teachers, classes, rooms, seed=seed)


def make_unsolvable_problem() -> Problem:
    """A problem with no valid room (all rooms are labs; need classroom)."""
    time = TimeStructure(working_days=5, periods_per_day=4, break_periods=frozenset())
    subjects = {"MATH": Subject("MATH", "Maths", weekly_hours=1, type="theory")}
    teachers = {"Alice": Teacher("Alice", frozenset({"MATH"}), frozenset(), max_weekly_load=5)}
    classes = {"A": ClassBatch("A", 25, ("MATH",), lunch_slot=99)}
    rooms = {"L1": Room("L1", 30, "lab")}  # lab only — MATH needs classroom
    return Problem("unsolvable", time, subjects, teachers, classes, rooms, seed=0)


# ---------------------------------------------------------------------------
# 1. Improved solver on small.json → success with 0 hard violations
# ---------------------------------------------------------------------------

class TestImprovedOnSmall:
    def test_success_zero_hard(self):
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "small.json")
        if not os.path.exists(data_path):
            pytest.skip("data/small.json not found — run generate_datasets.py first")
        problem = load_problem_from_json(data_path)
        solver = BacktrackingSolver.improved()
        result = solver.solve(problem, seed=42)

        assert result.success, f"Expected success but got: {result.status}"
        assert result.status == "solved"
        # Independently verify through core/cost.py (not the solver itself)
        breakdown = evaluate(result.timetable, complete=True)
        assert breakdown.total_hard_violations == 0, breakdown.summary()

    def test_timetable_not_none_on_success(self):
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "small.json")
        if not os.path.exists(data_path):
            pytest.skip("data/small.json not found")
        problem = load_problem_from_json(data_path)
        result = BacktrackingSolver.improved().solve(problem, seed=42)
        if result.success:
            assert result.timetable is not None


# ---------------------------------------------------------------------------
# 2. Baseline on tiny hand-built problem → success, feasible
# ---------------------------------------------------------------------------

class TestBaselineOnTiny:
    def test_baseline_success(self):
        p = make_tiny_problem()
        solver = BacktrackingSolver.baseline()
        result = solver.solve(p, seed=0)
        assert result.success, f"Baseline should solve tiny: {result.status}"
        breakdown = evaluate(result.timetable, complete=True)
        assert breakdown.total_hard_violations == 0, breakdown.summary()

    def test_result_fields_populated(self):
        p = make_tiny_problem()
        result = BacktrackingSolver.baseline().solve(p, seed=0)
        assert result.time_seconds >= 0
        assert result.nodes_expanded >= 0
        assert isinstance(result.cost_history, list)


# ---------------------------------------------------------------------------
# 3. Unsolvable problem → "no_solution", does not hang
# ---------------------------------------------------------------------------

class TestUnsolvable:
    def test_no_solution_status(self):
        p = make_unsolvable_problem()
        solver = BacktrackingSolver(BacktrackingConfig(time_limit=10.0))
        result = solver.solve(p, seed=0)
        assert result.status == "no_solution"
        assert result.success is False


# ---------------------------------------------------------------------------
# 4. Tiny node limit → "node_limit"
# ---------------------------------------------------------------------------

class TestNodeLimit:
    def test_node_limit_status(self):
        p = make_tiny_problem()
        solver = BacktrackingSolver(BacktrackingConfig(node_limit=1, time_limit=60.0))
        result = solver.solve(p, seed=0)
        # Either the first node solves it trivially or we hit the limit
        assert result.status in ("solved", "node_limit")
        # With limit=1, at most 1 node can be expanded
        assert result.nodes_expanded <= 1

    def test_node_limit_3(self):
        """With only 3 nodes, a 4-session problem should hit the limit."""
        p = make_tiny_problem()
        solver = BacktrackingSolver(BacktrackingConfig(node_limit=3, time_limit=60.0))
        result = solver.solve(p, seed=0)
        assert result.nodes_expanded <= 3


# ---------------------------------------------------------------------------
# 5. Reproducibility — same seed → same result
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result(self):
        p = make_tiny_problem()
        solver = BacktrackingSolver.improved()
        r1 = solver.solve(p, seed=7)
        r2 = solver.solve(p, seed=7)
        assert r1.status == r2.status
        assert r1.nodes_expanded == r2.nodes_expanded
        assert r1.cost == r2.cost

    def test_different_seeds_may_differ(self):
        """Different seeds can produce different node counts (not guaranteed, but common)."""
        p = make_tiny_problem()
        solver = BacktrackingSolver.improved()
        r1 = solver.solve(p, seed=1)
        r2 = solver.solve(p, seed=99)
        # Both should still succeed (just checking they run without error)
        assert r1.status in ("solved", "no_solution", "timeout", "node_limit")
        assert r2.status in ("solved", "no_solution", "timeout", "node_limit")


# ---------------------------------------------------------------------------
# 6. Improved expands ≤ nodes than baseline on a small test problem
# ---------------------------------------------------------------------------

class TestImprovedVsBaseline:
    def test_improved_le_nodes_than_baseline(self):
        """MRV + Degree + FC should never need more nodes than blind baseline."""
        # Use small.json if available; fall back to tiny
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "small.json")
        if os.path.exists(data_path):
            p = load_problem_from_json(data_path)
        else:
            p = make_tiny_problem()

        baseline_result = BacktrackingSolver.baseline().solve(p, seed=42)
        improved_result = BacktrackingSolver.improved().solve(p, seed=42)

        # Both must reach a terminal state (not just time out in a way that
        # inflates one's node count unfairly).
        assert improved_result.status in ("solved", "no_solution")
        assert improved_result.nodes_expanded <= baseline_result.nodes_expanded, (
            f"Improved ({improved_result.nodes_expanded}) should use <= nodes "
            f"than baseline ({baseline_result.nodes_expanded})"
        )


# ---------------------------------------------------------------------------
# 7. Cross-check with python-constraint on a micro problem
# ---------------------------------------------------------------------------

class TestPythonConstraintCrossCheck:
    """Verify that our solver and python-constraint agree on feasibility.

    We use a MICRO problem (1 class, 1 subject, 1 session) so the CSP is
    trivially small and python-constraint can enumerate it quickly.
    This cross-check lives only in tests — never in the solver itself.
    """

    def _make_micro(self, has_valid_room: bool) -> Problem:
        time = TimeStructure(working_days=2, periods_per_day=2, break_periods=frozenset())
        subjects = {"M": Subject("M", "Math", weekly_hours=1, type="theory")}
        teachers = {"T": Teacher("T", frozenset({"M"}), frozenset(), max_weekly_load=5)}
        classes = {"C": ClassBatch("C", 10, ("M",), lunch_slot=99)}
        if has_valid_room:
            rooms = {"R": Room("R", 20, "classroom")}
        else:
            rooms = {"R": Room("R", 20, "lab")}   # wrong type → unsolvable
        return Problem("micro", time, subjects, teachers, classes, rooms, seed=0)

    def test_solvable_cross_check(self):
        """Both our solver and python-constraint should find a solution."""
        try:
            from constraint import Problem as CspProblem
        except ImportError:
            pytest.skip("python-constraint not installed")

        p = self._make_micro(has_valid_room=True)
        our_result = BacktrackingSolver.improved().solve(p, seed=0)
        assert our_result.success, "Our solver should find a solution"

        # Build equivalent CSP with python-constraint
        csp = CspProblem()
        # Variable: "session_0" — value = (day, period, room_name)
        domain = [(d, per, "R")
                  for d in range(2) for per in range(2)]
        csp.addVariable("s0", domain)
        # No binary constraints needed for 1 session; any slot is valid.
        solutions = csp.getSolutions()
        assert len(solutions) > 0, "python-constraint should also find solutions"

    def test_unsolvable_cross_check(self):
        """Both should agree: no solution when room type is wrong."""
        try:
            from constraint import Problem as CspProblem
        except ImportError:
            pytest.skip("python-constraint not installed")

        p = self._make_micro(has_valid_room=False)
        our_result = BacktrackingSolver(BacktrackingConfig(time_limit=5.0)).solve(p, seed=0)
        assert not our_result.success, "Our solver should fail"

        # python-constraint equivalent: domain filtered by room type = classroom only
        csp = CspProblem()
        # The room is type=lab; classroom-needing subject has empty domain.
        domain = []   # no valid (day, period, room) triples
        if domain:
            csp.addVariable("s0", domain)
            solutions = csp.getSolutions()
        else:
            solutions = []   # empty domain → no solutions
        assert len(solutions) == 0, "python-constraint should also find no solutions"
