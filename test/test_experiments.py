"""tests/test_experiments.py — Tests for experiment runner, aggregation, re-scoring, and reproducibility."""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from algorithms.base import SolverResult
from core.loader import load_problem_from_json
from experiments.aggregate import (
    build_comparison_dataframe,
    compute_metrics_for_group,
    export_summary_files,
)
from experiments.config import ALGO_BT_IMPROVED, ALGO_MIN_CONFLICTS
from experiments.plots import (
    plot_cost_vs_iterations_min_conflicts,
    plot_cost_vs_iterations_sa,
    plot_hybrid_before_after,
    plot_time_comparison,
)
from experiments.runner import RunRecord, run_benchmark, run_single


def test_aggregate_computes_correct_stats():
    """Verify aggregation logic on a known synthetic set of runs."""
    fake_records = [
        RunRecord(
            algorithm="TestAlgo",
            dataset="small",
            seed=0,
            status="solved",
            time=1.0,
            nodes_expanded=100,
            hard_violations=0,
            soft_penalty=10,
            cost=10,
            solver_hard=0,
            solver_soft=10,
        ),
        RunRecord(
            algorithm="TestAlgo",
            dataset="small",
            seed=1,
            status="solved",
            time=3.0,
            nodes_expanded=200,
            hard_violations=0,
            soft_penalty=20,
            cost=20,
            solver_hard=0,
            solver_soft=20,
        ),
        RunRecord(
            algorithm="TestAlgo",
            dataset="small",
            seed=2,
            status="timeout",
            time=60.0,
            nodes_expanded=500,
            hard_violations=2,
            soft_penalty=50,
            cost=2050,
            solver_hard=2,
            solver_soft=50,
        ),
        RunRecord(
            algorithm="TestAlgo",
            dataset="small",
            seed=3,
            status="iteration_limit",
            time=5.0,
            nodes_expanded=250,
            hard_violations=1,
            soft_penalty=30,
            cost=1030,
            solver_hard=1,
            solver_soft=30,
        ),
    ]

    metrics = compute_metrics_for_group(fake_records)

    assert metrics["total_runs"] == 4
    # 2 out of 4 succeeded
    assert metrics["success_rate"] == "2/4 (50%)"
    assert metrics["timeouts"] == 1
    # Time mean and soft mean over successful runs: (1.0 + 3.0) / 2 = 2.0; (10 + 20) / 2 = 15.0
    assert metrics["time_mean"] == pytest.approx(2.0)
    assert metrics["soft_mean"] == pytest.approx(15.0)
    # Nodes mean over all runs: (100 + 200 + 500 + 250) / 4 = 262.5
    assert metrics["nodes_mean"] == pytest.approx(262.5)
    # Hard violations mean over all runs: (0 + 0 + 2 + 1) / 4 = 0.75
    assert metrics["hard_mean"] == pytest.approx(0.75)


def test_rescore_catches_deliberate_corruption():
    """Verify that re-scoring against core/cost.py catches fraudulent solver metrics."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "small.json")
    problem = load_problem_from_json(data_path)

    # Mock solver to return an invalid hard_violations count of 0 despite having conflicts,
    # or return a feasible timetable but claim 99 violations.
    with patch("experiments.runner.create_solver") as mock_create:
        mock_solver = MagicMock()
        # Solve real problem to get a real timetable
        from services.solver_registry import create_solver
        real_solver = create_solver(ALGO_BT_IMPROVED, time_limit=10.0)
        real_res = real_solver.solve(problem, seed=42)

        # Corrupt the reported numbers
        corrupted_res = SolverResult(
            timetable=real_res.timetable,
            success=True,
            status="solved",
            time_seconds=real_res.time_seconds,
            nodes_expanded=real_res.nodes_expanded,
            hard_violations=999,  # FAKE
            soft_penalty=888,     # FAKE
            cost=999888,
        )
        mock_solver.solve.return_value = corrupted_res
        mock_create.return_value = mock_solver

        rec = run_single(ALGO_BT_IMPROVED, problem, "small", seed=42, time_limit=10.0)

        assert rec.mismatch is True
        assert rec.solver_hard == 999
        assert rec.hard_violations == 0  # Re-scored to 0 by core/cost.py
        assert rec.soft_penalty != 888    # Re-scored by core/cost.py


def test_seed_reproducibility():
    """Verify that runs with identical seeds produce identical non-time metric outputs."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "small.json")
    problem = load_problem_from_json(data_path)

    rec1 = run_single(ALGO_MIN_CONFLICTS, problem, "small", seed=42, time_limit=10.0)
    rec2 = run_single(ALGO_MIN_CONFLICTS, problem, "small", seed=42, time_limit=10.0)

    assert rec1.status == rec2.status
    assert rec1.nodes_expanded == rec2.nodes_expanded
    assert rec1.hard_violations == rec2.hard_violations
    assert rec1.soft_penalty == rec2.soft_penalty
    assert rec1.cost == rec2.cost
    assert rec1.cost_history == rec2.cost_history


def test_quick_run_creates_expected_files():
    """Test generating files with a quick run produces all CSVs, Markdown, and plots."""
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "small.json")
    problem = load_problem_from_json(data_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        raw_csv = tmp_path / "raw_runs.csv"
        sum_csv = tmp_path / "summary.csv"
        sum_md = tmp_path / "summary.md"
        plot_mc = tmp_path / "cost_mc.png"
        plot_sa = tmp_path / "cost_sa.png"
        plot_hb = tmp_path / "hybrid_ba.png"
        plot_time = tmp_path / "time_comp.png"

        # Run 2 seeds for improved backtracking and min-conflicts on small
        records = []
        for algo in [ALGO_BT_IMPROVED, ALGO_MIN_CONFLICTS]:
            records.extend(run_benchmark(algo, problem, "small", seeds=[0, 1], time_limit=5.0))

        # Save raw runs
        raw_df = pd.DataFrame([r.to_csv_dict() for r in records])
        raw_df.to_csv(raw_csv, index=False)
        assert raw_csv.exists()

        # Check raw CSV columns
        expected_raw_cols = [
            "algorithm", "dataset", "seed", "status", "time", "nodes_expanded",
            "hard_violations", "soft_penalty", "cost", "phase1_time", "phase2_time",
            "soft_before", "soft_after"
        ]
        assert list(raw_df.columns) == expected_raw_cols

        # Save summary CSV & Markdown
        df_small, _ = export_summary_files(records, str(sum_csv), str(sum_md), datasets=["small"])
        assert sum_csv.exists()
        assert sum_md.exists()

        # Verify summary Markdown content
        md_text = sum_md.read_text(encoding="utf-8")
        assert "## Small Dataset" in md_text
        assert "Success Rate" in md_text

        # Generate plots
        plot_cost_vs_iterations_min_conflicts(records, plot_mc)
        plot_cost_vs_iterations_sa(records, plot_sa)
        plot_hybrid_before_after(records, plot_hb)
        plot_time_comparison(records, plot_time)

        assert plot_mc.exists()
        assert plot_time.exists()

