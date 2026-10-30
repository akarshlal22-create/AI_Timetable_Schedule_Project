"""tests/test_hybrid.py — Unit tests for HybridSolver (Backtracking -> SA).

Verifies:
1. Hybrid on data/small.json returns 0 hard violations AND soft penalty <= backtracking result.
2. In the hybrid, hard violations stay 0 for every accepted move in Phase 2.
3. Same seed gives identical results.
4. A tiny unsolvable problem returns the backtracking failure cleanly without hanging.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from algorithms.backtracking import BacktrackingSolver
from algorithms.hybrid import HybridConfig, HybridSolver
from core.cost import evaluate
from core.loader import load_problem_from_json
from core.models import ClassBatch, Problem, Room, Subject, Teacher, TimeStructure


@pytest.fixture
def small_problem() -> Problem:
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "small.json")
    if not os.path.exists(data_path):
        pytest.skip("data/small.json not found")
    return load_problem_from_json(data_path)


def make_unsolvable_problem() -> Problem:
    """Classroom theory session needed, but only lab rooms exist."""
    time = TimeStructure(working_days=2, periods_per_day=2, break_periods=frozenset())
    subjects = {"MATH": Subject("MATH", "Maths", weekly_hours=1, type="theory")}
    teachers = {"Alice": Teacher("Alice", frozenset({"MATH"}), frozenset(), max_weekly_load=5)}
    classes = {"A": ClassBatch("A", 25, ("MATH",), lunch_slot=99)}
    rooms = {"L1": Room("L1", 30, "lab")}
    return Problem("unsolvable", time, subjects, teachers, classes, rooms, seed=0)


class TestHybrid:
    def test_hybrid_solves_small_and_improves_soft(self, small_problem):
        """1. Hybrid returns 0 hard violations AND soft penalty <= Phase 1 backtracking soft penalty."""
        # Baseline backtracking soft penalty
        bt_res = BacktrackingSolver.improved().solve(small_problem, seed=42)
        assert bt_res.success is True
        bt_soft = bt_res.soft_penalty

        # Hybrid solver
        hybrid_solver = HybridSolver.default()
        res = hybrid_solver.solve(small_problem, seed=42)

        assert res.success is True
        assert res.hard_violations == 0

        # Verify independently with core/cost.py
        bd = evaluate(res.timetable, complete=True)
        assert bd.total_hard_violations == 0

        # Soft penalty after Phase 2 SA must be <= Phase 1 soft penalty
        soft_before = res.details["soft_before"]
        soft_after = res.details["soft_after"]
        assert soft_before == bt_soft
        assert soft_after <= soft_before
        assert res.soft_penalty <= bt_soft

    def test_hard_violations_strictly_zero_throughout_phase2(self, small_problem):
        """2. In the hybrid, hard violations stay 0 for every accepted move in Phase 2."""
        # Enable debug_check to assert 0 hard violations at every accepted move
        cfg = HybridConfig(debug_check=True, max_sa_iterations=1000)
        solver = HybridSolver(cfg)
        res = solver.solve(small_problem, seed=42)

        assert res.success is True
        assert res.hard_violations == 0

    def test_reproducibility_same_seed(self, small_problem):
        """3. Same seed gives identical results and cost history."""
        solver = HybridSolver.default()
        r1 = solver.solve(small_problem, seed=42)
        r2 = solver.solve(small_problem, seed=42)

        assert r1.status == r2.status
        assert r1.cost == r2.cost
        assert r1.soft_penalty == r2.soft_penalty
        assert r1.cost_history == r2.cost_history
        assert r1.timetable.assignment == r2.timetable.assignment

    def test_unsolvable_returns_backtracking_failure_cleanly(self):
        """4. A tiny unsolvable problem returns the backtracking failure cleanly without hanging."""
        p = make_unsolvable_problem()
        solver = HybridSolver(HybridConfig(bt_time_limit=5.0))
        res = solver.solve(p, seed=0)

        assert res.success is False
        assert res.status in ("no_solution", "timeout")
        assert res.details["phase2_iterations"] == 0
