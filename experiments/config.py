"""experiments/config.py — Benchmark configurations, paths, seeds, and constants."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Algorithm options
ALGO_BT_BASELINE = "Backtracking (baseline)"
ALGO_BT_IMPROVED = "Backtracking (MRV + Degree + Forward Checking)"
ALGO_MIN_CONFLICTS = "Min-Conflicts"
ALGO_SA = "Simulated Annealing"
ALGO_HYBRID = "Hybrid (Backtracking + SA)"

ALGORITHMS: List[str] = [
    ALGO_BT_BASELINE,
    ALGO_BT_IMPROVED,
    ALGO_MIN_CONFLICTS,
    ALGO_SA,
    ALGO_HYBRID,
]

# Datasets
DATASETS: List[str] = ["small", "large"]

# Execution parameters
DEFAULT_SEEDS: List[int] = list(range(10))  # Seeds 0 to 9 (10 runs)
QUICK_SEEDS: List[int] = list(range(3))     # Seeds 0 to 2 (3 runs)

DEFAULT_TIME_LIMIT: float = 60.0            # 60 seconds per run
QUICK_TIME_LIMIT: float = 10.0              # 10 seconds per run for quick mode
SCALING_TIME_LIMIT: float = 30.0            # 30 seconds per run for scaling

# Scaling experiment configuration
SCALING_SIZES: List[int] = [3, 5, 8, 12]     # 4 problem sizes
SCALING_SEEDS: List[int] = list(range(5))   # 5 seeds for randomized methods
SCALING_ALGORITHMS: List[str] = [
    ALGO_BT_BASELINE,
    ALGO_BT_IMPROVED,
    ALGO_MIN_CONFLICTS,
]

# Output file paths
RAW_RUNS_CSV: Path = RESULTS_DIR / "raw_runs.csv"
SUMMARY_CSV: Path = RESULTS_DIR / "summary.csv"
SUMMARY_MD: Path = RESULTS_DIR / "summary.md"
SCALING_CSV: Path = RESULTS_DIR / "scaling.csv"

PLOT_COST_MIN_CONFLICTS: Path = RESULTS_DIR / "cost_vs_iterations_min_conflicts.png"
PLOT_COST_SA: Path = RESULTS_DIR / "cost_vs_iterations_sa.png"
PLOT_HYBRID_BEFORE_AFTER: Path = RESULTS_DIR / "hybrid_before_after.png"
PLOT_TIME_COMPARISON: Path = RESULTS_DIR / "time_comparison.png"
PLOT_SCALING: Path = RESULTS_DIR / "scaling.png"

# Consistent visualization palette
ALGO_COLORS: Dict[str, str] = {
    ALGO_BT_BASELINE: "#64748B",      # Slate grey
    ALGO_BT_IMPROVED: "#2563EB",      # Blue
    ALGO_MIN_CONFLICTS: "#059669",    # Green
    ALGO_SA: "#D97706",               # Amber / Orange
    ALGO_HYBRID: "#7C3AED",           # Purple
}

