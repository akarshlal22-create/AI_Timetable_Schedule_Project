"""tests/test_solver_registry.py — Test that all registered solvers build and run on data/small.json."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from algorithms.base import SolverResult
from core.loader import load_problem_from_json
from core.models import Problem
from services.solver_registry import ALGORITHM_OPTIONS, create_solver


@pytest.fixture
def small_problem() -> Problem:
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "small.json")
    if not os.path.exists(data_path):
        pytest.skip("data/small.json not found")
    return load_problem_from_json(data_path)


class TestSolverRegistry:
    @pytest.mark.parametrize("algo_name", ALGORITHM_OPTIONS)
    def test_all_options_build_and_run(self, small_problem, algo_name):
        solver = create_solver(algo_name, time_limit=10.0)
        res = solver.solve(small_problem, seed=42)

        assert isinstance(res, SolverResult)
        assert res.status in ("solved", "no_solution", "timeout", "node_limit", "iteration_limit", "stopped_by_limit", "frozen")
        assert res.time_seconds >= 0.0
        assert res.nodes_expanded >= 0

    def test_invalid_algorithm_name_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm name"):
            create_solver("NonExistentAlgorithm")
