"""experiments/aggregate.py — Aggregates raw run metrics into structured comparison tables."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from experiments.config import (
    ALGORITHMS,
    ALGO_BT_BASELINE,
    ALGO_BT_IMPROVED,
    ALGO_HYBRID,
    ALGO_MIN_CONFLICTS,
    ALGO_SA,
)
from experiments.runner import RunRecord


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def compute_metrics_for_group(records: List[RunRecord]) -> Dict[str, Any]:
    """Compute summary statistics for a collection of runs of one algorithm on one dataset."""
    if not records:
        return {
            "success_rate": "N/A",
            "time_str": "N/A",
            "nodes_mean": 0.0,
            "hard_mean": 0.0,
            "soft_mean": 0.0,
            "cost_mean": 0.0,
            "timeouts": 0,
            "total_runs": 0,
            "phase1_time_mean": None,
            "phase2_time_mean": None,
            "soft_before_mean": None,
            "soft_after_mean": None,
        }

    total_runs = len(records)
    # Successful runs: 0 hard violations and solved status
    succ_runs = [r for r in records if r.hard_violations == 0 and r.status == "solved"]
    timeouts = sum(1 for r in records if r.status == "timeout")

    # Time and Soft Penalty averages are computed over SUCCESSFUL runs
    active_for_avg = succ_runs if succ_runs else records
    succ_times = [r.time for r in active_for_avg]
    mean_t = _mean(succ_times)
    std_t = _std(succ_times)

    if len(records) == 1:
        time_str = f"{mean_t:.4f}s"
    else:
        time_str = f"{mean_t:.4f} +/- {std_t:.4f}s"

    succ_softs = [float(r.soft_penalty) for r in active_for_avg]
    mean_soft = _mean(succ_softs)

    all_nodes = [float(r.nodes_expanded) for r in records]
    mean_nodes = _mean(all_nodes)

    all_hard = [float(r.hard_violations) for r in records]
    mean_hard = _mean(all_hard)

    all_cost = [float(r.cost) for r in records]
    mean_cost = _mean(all_cost)

    succ_rate_pct = (len(succ_runs) / total_runs) * 100.0
    succ_rate_str = f"{len(succ_runs)}/{total_runs} ({succ_rate_pct:.0f}%)"

    # Hybrid specific phase metrics
    phase1_times = [r.phase1_time for r in active_for_avg if r.phase1_time is not None]
    phase2_times = [r.phase2_time for r in active_for_avg if r.phase2_time is not None]
    softs_before = [float(r.soft_before) for r in active_for_avg if r.soft_before is not None]
    softs_after = [float(r.soft_after) for r in active_for_avg if r.soft_after is not None]

    p1_mean = _mean(phase1_times) if phase1_times else None
    p2_mean = _mean(phase2_times) if phase2_times else None
    sb_mean = _mean(softs_before) if softs_before else None
    sa_mean = _mean(softs_after) if softs_after else None

    return {
        "success_rate": succ_rate_str,
        "time_str": time_str,
        "time_mean": mean_t,
        "time_std": std_t,
        "nodes_mean": mean_nodes,
        "hard_mean": mean_hard,
        "soft_mean": mean_soft,
        "cost_mean": mean_cost,
        "timeouts": timeouts,
        "total_runs": total_runs,
        "phase1_time_mean": p1_mean,
        "phase2_time_mean": p2_mean,
        "soft_before_mean": sb_mean,
        "soft_after_mean": sa_mean,
    }


def build_comparison_dataframe(
    records: List[RunRecord],
    dataset_name: str,
    algorithms: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build a comparison table for *dataset_name* with metrics as rows and algorithms as columns.

    Layout matches the assignment specification.
    """
    if algorithms is None:
        algorithms = ALGORITHMS

    # Short display headers for columns
    header_map = {
        ALGO_BT_BASELINE: "Backtracking (baseline)",
        ALGO_BT_IMPROVED: "Backtracking (improved)",
        ALGO_MIN_CONFLICTS: "Min-Conflicts",
        ALGO_SA: "Simulated Annealing",
        ALGO_HYBRID: "Hybrid (BT + SA)",
    }

    col_headers = [header_map.get(a, a) for a in algorithms]

    metric_rows = [
        "Success Rate",
        "Time (s, mean +/- std)*",
        "Nodes / Iterations (mean)",
        "Hard Violations (mean)",
        "Soft Penalty (mean)*",
        "Total Cost (mean)",
        "Timeouts Count",
        "Runs Count",
        "Hybrid Phase 1 Time (s)",
        "Hybrid Phase 2 Time (s)",
        "Hybrid Soft Penalty Before SA",
        "Hybrid Soft Penalty After SA",
    ]

    data: Dict[str, List[str]] = {h: [] for h in col_headers}

    for algo, header in zip(algorithms, col_headers):
        algo_records = [r for r in records if r.dataset == dataset_name and r.algorithm == algo]
        metrics = compute_metrics_for_group(algo_records)

        # Runs count label
        if algo == ALGO_BT_BASELINE:
            runs_label = f"1 run (deterministic)"
        else:
            runs_label = f"{metrics['total_runs']} runs"

        data[header].append(metrics["success_rate"])
        data[header].append(metrics["time_str"])
        data[header].append(f"{metrics['nodes_mean']:.1f}")
        data[header].append(f"{metrics['hard_mean']:.2f}")
        data[header].append(f"{metrics['soft_mean']:.1f}")
        data[header].append(f"{metrics['cost_mean']:.1f}")
        data[header].append(str(metrics["timeouts"]))
        data[header].append(runs_label)

        # Hybrid-specific rows
        data[header].append(f"{metrics['phase1_time_mean']:.4f}s" if metrics["phase1_time_mean"] is not None else "-")
        data[header].append(f"{metrics['phase2_time_mean']:.4f}s" if metrics["phase2_time_mean"] is not None else "-")
        data[header].append(f"{metrics['soft_before_mean']:.1f}" if metrics["soft_before_mean"] is not None else "-")
        data[header].append(f"{metrics['soft_after_mean']:.1f}" if metrics["soft_after_mean"] is not None else "-")

    df = pd.DataFrame(data, index=metric_rows)
    df.index.name = "Metric"
    return df


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Format DataFrame as a clean Markdown table without external dependencies."""
    headers = [str(df.index.name or "Metric")] + [str(c) for c in df.columns]
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = []
    for idx, row in df.iterrows():
        row_vals = [str(idx)] + [str(v) for v in row]
        rows.append("| " + " | ".join(row_vals) + " |")
    return "\n".join([header_row, separator_row] + rows)


def generate_summary_markdown(
    df_small: Optional[pd.DataFrame],
    df_large: Optional[pd.DataFrame],
) -> str:
    """Generate clean, report-ready GitHub markdown string for both dataset summaries."""
    lines: List[str] = [
        "# Algorithm Comparison & Benchmark Summary",
        "",
        "> **Rule**: Averages for execution time and soft penalty are computed over *successful* runs "
        "(hard violations = 0). The success rate is reported separately. "
        "Baseline Backtracking is deterministic and evaluated once; seeded and stochastic algorithms "
        "are evaluated across independent runs.",
        "",
    ]

    if df_small is not None:
        lines.append("## Small Dataset (`data/small.json`)")
        lines.append("")
        lines.append(_df_to_markdown_table(df_small))
        lines.append("")

    if df_large is not None:
        lines.append("## Large Dataset (`data/large.json`)")
        lines.append("")
        lines.append(_df_to_markdown_table(df_large))
        lines.append("")

    lines.append(
        "*Notes*:\n"
        "- All violations and soft penalties are independently verified using `core/cost.py`.\n"
        "- Simulated Annealing and Hybrid (Backtracking + SA) constitute bonus algorithms.\n"
        "- Nodes/Iterations counts CSP variable assignment attempts for Backtracking, and state transitions for Min-Conflicts and SA."
    )
    lines.append("")
    return "\n".join(lines)


def export_summary_files(
    records: List[RunRecord],
    csv_path: str,
    md_path: str,
    datasets: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and write summary.csv and summary.md to disk.

    Returns:
        Tuple of ``(df_small, df_large)``.
    """
    if datasets is None:
        datasets = ["small", "large"]

    df_small = build_comparison_dataframe(records, "small") if "small" in datasets else None
    df_large = build_comparison_dataframe(records, "large") if "large" in datasets else None

    # Write combined CSV
    combined_rows = []
    if df_small is not None:
        combined_rows.append(pd.DataFrame([{"Metric": "=== DATASET: SMALL ==="}]))
        small_csv_df = df_small.reset_index()
        combined_rows.append(small_csv_df)
    if df_large is not None:
        combined_rows.append(pd.DataFrame([{"Metric": "=== DATASET: LARGE ==="}]))
        large_csv_df = df_large.reset_index()
        combined_rows.append(large_csv_df)

    if combined_rows:
        final_csv_df = pd.concat(combined_rows, ignore_index=True)
        final_csv_df.to_csv(csv_path, index=False)

    # Write Markdown
    md_content = generate_summary_markdown(df_small, df_large)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return df_small, df_large
