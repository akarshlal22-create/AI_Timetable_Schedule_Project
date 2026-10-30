"""experiments/plots.py — Publication-quality, minimalist matplotlib visualisations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.config import (
    ALGORITHMS,
    ALGO_BT_BASELINE,
    ALGO_BT_IMPROVED,
    ALGO_COLORS,
    ALGO_HYBRID,
    ALGO_MIN_CONFLICTS,
    ALGO_SA,
)
from experiments.runner import RunRecord


def _setup_plot_style() -> None:
    """Configure minimalist, publication-ready plot aesthetics."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#F1F5F9",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#E2E8F0",
    })


def plot_cost_vs_iterations_min_conflicts(
    records: List[RunRecord],
    output_path: Path,
) -> None:
    """Plot cost vs iterations for Min-Conflicts runs on the large dataset."""
    _setup_plot_style()
    mc_large = [r for r in records if r.algorithm == ALGO_MIN_CONFLICTS and r.dataset == "large"]
    if not mc_large:
        # Fallback to small if large was not run
        mc_large = [r for r in records if r.algorithm == ALGO_MIN_CONFLICTS]

    if not mc_large:
        return

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    # Plot up to 4 runs in faint lines, seed 0 in bold
    color = ALGO_COLORS[ALGO_MIN_CONFLICTS]
    seed_0_run = next((r for r in mc_large if r.seed == 0), mc_large[0])

    for r in mc_large[:5]:
        if r.seed != seed_0_run.seed and r.cost_history:
            ax.plot(
                range(len(r.cost_history)),
                r.cost_history,
                color=color,
                alpha=0.25,
                linewidth=1.0,
                label=f"Seed {r.seed}" if r.seed == 1 else None,
            )

    if seed_0_run.cost_history:
        ax.plot(
            range(len(seed_0_run.cost_history)),
            seed_0_run.cost_history,
            color=color,
            alpha=0.95,
            linewidth=2.0,
            label=f"Seed {seed_0_run.seed} (primary)",
        )

    ax.set_title("Min-Conflicts: Cost vs. Iterations (Large Dataset)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Cost (1000 × Hard Violations + Soft Penalty)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_cost_vs_iterations_sa(
    records: List[RunRecord],
    output_path: Path,
) -> None:
    """Plot current cost and best cost vs iterations for one Simulated Annealing run."""
    _setup_plot_style()
    sa_runs = [r for r in records if r.algorithm == ALGO_SA and r.cost_history]
    if not sa_runs:
        return

    # Choose seed 0 on large if available, otherwise first available
    run = next((r for r in sa_runs if r.dataset == "large" and r.seed == 0), sa_runs[0])

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)

    color_curr = "#F59E0B"  # Light amber
    color_best = "#B45309"  # Dark amber

    steps = list(range(len(run.cost_history)))
    ax.plot(steps, run.cost_history, color=color_curr, alpha=0.45, linewidth=0.8, label="Current Candidate Cost")

    if run.best_cost_history:
        best_steps = list(range(len(run.best_cost_history)))
        ax.plot(best_steps, run.best_cost_history, color=color_best, alpha=1.0, linewidth=2.0, label="Best Cost Encountered")

    dataset_label = run.dataset.capitalize()
    ax.set_title(f"Simulated Annealing: Cost Convergence ({dataset_label} Dataset, Seed {run.seed})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Cost (1000 × Hard Violations + Soft Penalty)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_hybrid_before_after(
    records: List[RunRecord],
    output_path: Path,
) -> None:
    """Plot soft penalty before and after Simulated Annealing for Hybrid solver."""
    _setup_plot_style()
    hybrid_runs = [r for r in records if r.algorithm == ALGO_HYBRID and r.soft_before is not None]
    if not hybrid_runs:
        return

    datasets_found = []
    for d in ["small", "large"]:
        if any(r.dataset == d for r in hybrid_runs):
            datasets_found.append(d)

    if not datasets_found:
        return

    before_means = []
    after_means = []
    labels = []

    for d in datasets_found:
        d_runs = [r for r in hybrid_runs if r.dataset == d and r.hard_violations == 0]
        if not d_runs:
            d_runs = [r for r in hybrid_runs if r.dataset == d]
        b_mean = np.mean([r.soft_before for r in d_runs])
        a_mean = np.mean([r.soft_after for r in d_runs])
        before_means.append(b_mean)
        after_means.append(a_mean)
        labels.append(f"{d.capitalize()} Dataset")

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)

    color_before = "#A78BFA"  # Light purple
    color_after = "#6D28D9"   # Dark purple

    rects1 = ax.bar(x - width / 2, before_means, width, label="Phase 1 (Backtracking)", color=color_before)
    rects2 = ax.bar(x + width / 2, after_means, width, label="Phase 2 (After SA Refinement)", color=color_after)

    # Value labels on top of bars
    for rects in (rects1, rects2):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("Hybrid Solver: Soft Penalty Optimization")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Soft Penalty Score (lower is better)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_time_comparison(
    records: List[RunRecord],
    output_path: Path,
) -> None:
    """Plot mean runtime per algorithm per dataset on log scale."""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    datasets = [d for d in ["small", "large"] if any(r.dataset == d for r in records)]
    if not datasets:
        return

    n_algos = len(ALGORITHMS)
    x = np.arange(len(datasets))
    bar_width = 0.15

    for idx, algo in enumerate(ALGORITHMS):
        means = []
        for d in datasets:
            d_runs = [r for r in records if r.algorithm == algo and r.dataset == d and r.hard_violations == 0]
            if not d_runs:
                d_runs = [r for r in records if r.algorithm == algo and r.dataset == d]
            if d_runs:
                means.append(np.mean([r.time for r in d_runs]))
            else:
                means.append(0.0)

        offset = (idx - (n_algos - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means,
            bar_width,
            label=algo,
            color=ALGO_COLORS.get(algo, "#64748B"),
        )

    ax.set_yscale("log")
    ax.set_title("Algorithm Execution Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.set_ylabel("Mean Execution Time in Seconds (log scale)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_scaling_experiment(
    scaling_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot execution time vs problem size (sessions) for the scaling experiment."""
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    algos = scaling_df["algorithm"].unique()
    for algo in algos:
        sub_df = scaling_df[scaling_df["algorithm"] == algo].sort_values("sessions")
        color = ALGO_COLORS.get(algo, "#3B82F6")

        # Separate successful and timed-out runs
        succ = sub_df[sub_df["status"] == "solved"]
        failed = sub_df[sub_df["status"] != "solved"]

        if not succ.empty:
            ax.plot(
                succ["sessions"],
                succ["mean_time"],
                marker="o",
                linewidth=2.0,
                color=color,
                label=algo,
            )

        if not failed.empty:
            ax.scatter(
                failed["sessions"],
                failed["mean_time"],
                marker="x",
                s=90,
                color="#EF4444",
                linewidths=2.5,
                zorder=5,
                label="Timeout / Infeasible" if algo == algos[0] else None,
            )

    ax.set_title("Scaling Analysis: Runtime vs. Problem Size")
    ax.set_xlabel("Problem Size (Number of Scheduled Sessions)")
    ax.set_ylabel("Execution Time (Seconds)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

