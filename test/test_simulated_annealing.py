"""tests/test_simulated_annealing.py — Unit tests for SimulatedAnnealingSolver.

Verifies:
1. SA on data/small.json reaches 0 hard violations (verified via core/cost.py).
2. Same seed gives identical results and the same cost_history.
3. Best-so-far is returned: final reported cost is never worse than the lowest value in cost_history.
4. Incremental delta cost matches a full recount from core/cost.py after many random moves and swaps.
5. Swaps never produce room-type, capacity, or lab-duration violations.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from algorithms.neighbourhood import (
    NeighbourhoodState,
    generate_candidate_move,
)
from algorithms.simulated_annealing import (
    SimulatedAnnealingConfig,
    SimulatedAnnealingSolver,
)
from core.cost import evaluate
from core.loader import load_problem_from_json
from core.models import Problem
from core.problem_builder import build_csp


@pytest.fixture
def small_problem() -> Problem:
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "small.json")
    if not os.path.exists(data_path):
        pytest.skip("data/small.json not found")
    return load_problem_from_json(data_path)


class TestSimulatedAnnealing:
    def test_sa_solves_small_zero_hard(self, small_problem):
        """1. SA on data/small.json reaches 0 hard violations with default config."""
        solver = SimulatedAnnealingSolver.default()
        res = solver.solve(small_problem, seed=42)

        assert res.success is True, f"SA failed to reach 0 hard: {res.status}"
        assert res.timetable is not None

        # Verify independently through core/cost.py
        bd = evaluate(res.timetable, complete=True)
        assert bd.total_hard_violations == 0, bd.summary()
        assert res.hard_violations == 0

    def test_reproducibility_same_seed(self, small_problem):
        """2. Same seed gives identical results and identical cost_history."""
        solver = SimulatedAnnealingSolver.default()
        r1 = solver.solve(small_problem, seed=123)
        r2 = solver.solve(small_problem, seed=123)

        assert r1.status == r2.status
        assert r1.cost == r2.cost
        assert r1.nodes_expanded == r2.nodes_expanded
        assert r1.cost_history == r2.cost_history
        assert r1.timetable.assignment == r2.timetable.assignment

    def test_best_so_far_returned(self, small_problem):
        """3. Final reported cost is never worse than the minimum value recorded in cost_history."""
        solver = SimulatedAnnealingSolver.default()
        res = solver.solve(small_problem, seed=42)

        min_cost_seen = min(res.cost_history)
        assert res.cost <= min_cost_seen, (
            f"Reported cost {res.cost} should be <= min in cost_history {min_cost_seen}"
        )

    def test_incremental_matches_full_recount(self, small_problem):
        """4. Incremental cost matches full recount from core/cost.py after random moves and swaps."""
        sessions, domains, _ = build_csp(small_problem)
        rng = random.Random(42)
        initial_asgn = {s: rng.choice(domains[s]) for s in sessions}

        state = NeighbourhoodState(problem=small_problem, assignment=initial_asgn)

        for step in range(200):
            cand = generate_candidate_move(sessions, domains, state.assignment, rng, p_swap=0.5)
            delta_cost, new_hard, new_soft = state.eval_delta_cost(cand)

            # Apply candidate move
            state.apply_move(cand, delta_cost, new_hard, new_soft)

            # Verify against full evaluate recount from core.cost
            state.check_recount()

    def test_swaps_never_violate_unary_constraints(self, small_problem):
        """5. Swaps never produce room-type, capacity, or lab-duration violations."""
        sessions, domains, _ = build_csp(small_problem)
        rng = random.Random(42)
        assignment = {s: rng.choice(domains[s]) for s in sessions}

        p = small_problem
        for _ in range(300):
            cand = generate_candidate_move(sessions, domains, assignment, rng, p_swap=1.0)
            if cand.kind != "swap" or cand.session2 is None or cand.new_val2 is None:
                continue

            s1 = cand.session1
            s2 = cand.session2
            v1 = cand.new_val1
            v2 = cand.new_val2

            # Lab duration compatibility
            assert s1.is_lab == s2.is_lab

            # Room type compatibility
            r1 = p.rooms[v1[2]]
            r2 = p.rooms[v2[2]]
            expected_type1 = "lab" if s1.is_lab else "classroom"
            expected_type2 = "lab" if s2.is_lab else "classroom"
            assert r1.type == expected_type1
            assert r2.type == expected_type2

            # Room capacity compatibility
            c1 = p.classes[s1.class_name]
            c2 = p.classes[s2.class_name]
            assert r1.capacity >= c1.strength
            assert r2.capacity >= c2.strength

            # Unary domain containment
            assert v1 in domains[s1]
            assert v2 in domains[s2]
