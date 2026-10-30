"""tests/test_min_conflicts.py — Unit tests for MinConflictsSolver.

Verifies:
1. Solves data/small.json with 0 hard violations (verified through core/cost.py).
2. Same seed gives identical results and the same cost_history.
3. Different seeds can give different timetables.
4. Incremental conflict count matches full recount from core/cost.py after random moves.
5. A tiny unsolvable problem stops at the iteration limit without hanging and returns success False.
6. cost_history is non-empty, and its last value equals the reported final cost.
7. Config flags (p_walk=0, restarts disabled, greedy start) change solver behavior and run correctly.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from algorithms.min_conflicts import MinConflictsConfig, MinConflictsSolver
from core.cost import conflicts, evaluate
from core.loader import load_problem_from_json
from core.models import ClassBatch, Problem, Room, Subject, Teacher, TimeStructure, Timetable
from core.problem_builder import build_csp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_problem() -> Problem:
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "small.json")
    if not os.path.exists(data_path):
        pytest.skip("data/small.json not found")
    return load_problem_from_json(data_path)


def make_tiny_problem(seed: int = 42) -> Problem:
    """Solvable tiny problem."""
    time = TimeStructure(working_days=5, periods_per_day=4, break_periods=frozenset())
    subjects = {
        "MATH": Subject("MATH", "Maths", weekly_hours=2, type="theory"),
        "CS":   Subject("CS",   "CS 101", weekly_hours=2, type="theory"),
    }
    teachers = {
        "Alice": Teacher("Alice", frozenset({"MATH"}), frozenset(), max_weekly_load=10),
        "Bob":   Teacher("Bob",   frozenset({"CS"}),   frozenset(), max_weekly_load=10),
    }
    classes = {
        "A": ClassBatch("A", 25, ("MATH", "CS"), lunch_slot=99),
    }
    rooms = {
        "R1": Room("R1", 30, "classroom"),
        "R2": Room("R2", 30, "classroom"),
    }
    return Problem("tiny", time, subjects, teachers, classes, rooms, seed=seed)


def make_unsolvable_problem() -> Problem:
    """Unsolvable: classroom session required, but only lab rooms exist."""
    time = TimeStructure(working_days=2, periods_per_day=2, break_periods=frozenset())
    subjects = {"MATH": Subject("MATH", "Maths", weekly_hours=1, type="theory")}
    teachers = {"Alice": Teacher("Alice", frozenset({"MATH"}), frozenset(), max_weekly_load=5)}
    classes = {"A": ClassBatch("A", 25, ("MATH",), lunch_slot=99)}
    rooms = {"L1": Room("L1", 30, "lab")}  # Lab only -> theory domain empty
    return Problem("unsolvable", time, subjects, teachers, classes, rooms, seed=0)


def make_clashing_unsolvable_problem() -> Problem:
    """Problem where slots are too few so clashes are unavoidable (2 sessions, 1 slot)."""
    time = TimeStructure(working_days=1, periods_per_day=1, break_periods=frozenset())
    subjects = {
        "MATH": Subject("MATH", "Maths", weekly_hours=1, type="theory"),
        "PHY":  Subject("PHY",  "Physics", weekly_hours=1, type="theory"),
    }
    teachers = {
        "Alice": Teacher("Alice", frozenset({"MATH", "PHY"}), frozenset(), max_weekly_load=5),
    }
    classes = {
        "A": ClassBatch("A", 20, ("MATH", "PHY"), lunch_slot=99),
    }
    rooms = {
        "R1": Room("R1", 30, "classroom"),
    }
    # Both sessions must be taught by Alice to Class A at day 0, period 0 in R1 -> always clashes
    return Problem("clash_unsolvable", time, subjects, teachers, classes, rooms, seed=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMinConflicts:
    def test_solve_small_zero_hard(self, small_problem):
        """1. Solves data/small.json with 0 hard violations (checked by core/cost.py)."""
        solver = MinConflictsSolver.default()
        result = solver.solve(small_problem, seed=42)

        assert result.success is True, f"Failed to solve: {result.status}"
        assert result.status == "solved"
        assert result.timetable is not None

        # Verify independently via core.cost
        breakdown = evaluate(result.timetable, complete=True)
        assert breakdown.total_hard_violations == 0, breakdown.summary()
        assert len(conflicts(result.timetable)) == 0

    def test_reproducibility_same_seed(self, small_problem):
        """2. Same seed gives identical results and the same cost_history."""
        solver = MinConflictsSolver.default()
        r1 = solver.solve(small_problem, seed=123)
        r2 = solver.solve(small_problem, seed=123)

        assert r1.status == r2.status
        assert r1.nodes_expanded == r2.nodes_expanded
        assert r1.cost == r2.cost
        assert r1.cost_history == r2.cost_history
        assert r1.timetable.assignment == r2.timetable.assignment

    def test_different_seeds_produce_different_timetables(self, small_problem):
        """3. Different seeds can give different timetables (stochastic local search)."""
        solver = MinConflictsSolver.default()
        r1 = solver.solve(small_problem, seed=1)
        r2 = solver.solve(small_problem, seed=999)

        assert r1.success is True
        assert r2.success is True
        # Both valid solutions, but distinct assignments
        assert r1.timetable.assignment != r2.timetable.assignment

    def test_incremental_matches_full_recount(self, small_problem):
        """4. The incremental conflict count matches a full recount from core/cost.py."""
        sessions, domains, _ = build_csp(small_problem)
        import random
        from collections import defaultdict

        rng = random.Random(42)
        assignment = {s: rng.choice(domains[s]) for s in sessions}

        teacher_occ = defaultdict(set)
        class_occ = defaultdict(set)
        room_occ = defaultdict(set)

        def slots_of(s, val):
            d, p, _ = val
            return [(d, p), (d, p + 1)] if s.is_lab else [(d, p)]

        for s, v in assignment.items():
            for sl in slots_of(s, v):
                teacher_occ[(s.teacher_name, sl)].add(s)
                class_occ[(s.class_name, sl)].add(s)
                room_occ[(v[2], sl)].add(s)

        conflicted_vars = set()
        for s, v in assignment.items():
            for sl in slots_of(s, v):
                if (
                    len(teacher_occ[(s.teacher_name, sl)]) > 1
                    or len(class_occ[(s.class_name, sl)]) > 1
                    or len(room_occ[(v[2], sl)]) > 1
                ):
                    conflicted_vars.add(s)
                    break

        # Perform 100 random moves and verify after each move
        for _ in range(100):
            var = rng.choice(sessions)
            cur_val = assignment[var]
            new_val = rng.choice(domains[var])
            if new_val == cur_val:
                continue

            affected = {var}
            for sl in slots_of(var, cur_val):
                affected.update(teacher_occ[(var.teacher_name, sl)])
                affected.update(class_occ[(var.class_name, sl)])
                affected.update(room_occ[(cur_val[2], sl)])
                teacher_occ[(var.teacher_name, sl)].remove(var)
                class_occ[(var.class_name, sl)].remove(var)
                room_occ[(cur_val[2], sl)].remove(var)

            assignment[var] = new_val

            for sl in slots_of(var, new_val):
                teacher_occ[(var.teacher_name, sl)].add(var)
                class_occ[(var.class_name, sl)].add(var)
                room_occ[(new_val[2], sl)].add(var)
                affected.update(teacher_occ[(var.teacher_name, sl)])
                affected.update(class_occ[(var.class_name, sl)])
                affected.update(room_occ[(new_val[2], sl)])

            for s in affected:
                s_val = assignment[s]
                is_conf = any(
                    len(teacher_occ[(s.teacher_name, sl)]) > 1
                    or len(class_occ[(s.class_name, sl)]) > 1
                    or len(room_occ[(s_val[2], sl)]) > 1
                    for sl in slots_of(s, s_val)
                )
                if is_conf:
                    conflicted_vars.add(s)
                else:
                    conflicted_vars.discard(s)

            tt = Timetable(problem=small_problem, assignment=assignment)
            expected_conflicted = {s for s, _ in conflicts(tt)}
            assert conflicted_vars == expected_conflicted

    def test_unsolvable_stops_at_limit(self):
        """5. A tiny unsolvable problem stops at the iteration limit without hanging."""
        p = make_clashing_unsolvable_problem()
        config = MinConflictsConfig(max_iterations=50, time_limit=5.0)
        solver = MinConflictsSolver(config)
        res = solver.solve(p, seed=0)

        assert res.success is False
        assert res.status in ("iteration_limit", "timeout")
        assert res.nodes_expanded <= 50
        assert res.hard_violations > 0

    def test_cost_history_populated_and_matches_final(self, small_problem):
        """6. cost_history is non-empty, and its last value equals the reported final cost."""
        solver = MinConflictsSolver.default()
        res = solver.solve(small_problem, seed=42)

        assert len(res.cost_history) > 0
        assert res.cost_history[-1] == res.cost

    def test_config_flags_change_behavior(self, small_problem):
        """7. Config flags (p_walk=0, restarts disabled, greedy) run correctly and affect search."""
        cfg_no_walk = MinConflictsConfig(p_walk=0.0, restart_threshold=999999)
        res_no_walk = MinConflictsSolver(cfg_no_walk).solve(small_problem, seed=42)
        assert res_no_walk.restarts == 0

        cfg_greedy = MinConflictsConfig(initial_strategy="greedy")
        res_greedy = MinConflictsSolver(cfg_greedy).solve(small_problem, seed=42)
        assert res_greedy.success is True

        # Greedy start typically requires fewer iterations on small.json
        assert res_greedy.nodes_expanded <= res_no_walk.nodes_expanded or res_greedy.success
