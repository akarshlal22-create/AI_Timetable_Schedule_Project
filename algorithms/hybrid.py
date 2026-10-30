"""algorithms/hybrid.py — Hybrid Solver (Backtracking Phase 1 -> Simulated Annealing Phase 2).

Architecture:
-------------
- Phase 1 (Backtracking Search):
    Runs BacktrackingSolver.improved() (MRV + Degree + Forward Checking) to find
    a strictly feasible timetable (0 hard violations). If no feasible timetable is found,
    the solver terminates immediately and reports the failure honestly without faking.

- Phase 2 (Feasibility-Preserving Simulated Annealing):
    Starts directly from the feasible timetable obtained in Phase 1.
    A refinement temperature is calibrated based purely on soft-penalty variations.
    Crucially, feasibility is strictly preserved: any proposed move or swap that would
    introduce even a single hard violation (new_hard > 0) is REJECTED IMMEDIATELY.
    Hard violations remain exactly 0 throughout Phase 2, allowing SA to purely optimize
    soft constraints (S1 gaps, S2 spread, S4 teacher preferences, S5 consecutive loads).

- Metrics Reporting:
    Records soft penalty before and after Phase 2, individual phase execution times,
    Phase 1 nodes expanded, Phase 2 SA iterations, and populated SolverResult.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from algorithms.backtracking import BacktrackingConfig, BacktrackingSolver
from algorithms.base import SolverResult, fill_metrics
from algorithms.neighbourhood import (
    NeighbourhoodState,
    generate_candidate_move,
)
from core.models import AssignmentTuple, Problem, Session, Timetable
from core.problem_builder import build_csp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class HybridConfig:
    """Configuration parameters for HybridSolver.

    Attributes:
        bt_time_limit:              Time limit in seconds for Phase 1 Backtracking (default 30.0).
        sa_time_limit:              Time limit in seconds for Phase 2 SA (default 30.0).
        t0:                         Refinement initial temperature. If None, auto-calibrated.
        target_initial_acceptance:  Target acceptance rate for soft worsening moves (default 0.40).
        alpha:                      Cooling factor for Phase 2 (default 0.995).
        t_min:                      Minimum temperature to terminate (default 1e-3).
        moves_per_temperature:      Moves per cooling step (default 2 * num_sessions).
        max_sa_iterations:          Maximum SA iterations in Phase 2 (default 30000).
        plateau_temp_levels:        Stop if no soft improvement for N temp levels (default 30).
        p_swap:                     Probability of proposing a swap move (default 0.5).
        k_cost_history:             Record cost history every k iterations (default 1).
        debug_check:                Verify that hard violations remain 0 at every step.
    """
    bt_time_limit: float = 30.0
    sa_time_limit: float = 30.0
    t0: Optional[float] = None
    target_initial_acceptance: float = 0.40
    alpha: float = 0.995
    t_min: float = 1e-3
    moves_per_temperature: Optional[int] = None
    max_sa_iterations: int = 30000
    plateau_temp_levels: int = 30
    p_swap: float = 0.5
    k_cost_history: int = 1
    debug_check: bool = False


# ---------------------------------------------------------------------------
# Solver implementation
# ---------------------------------------------------------------------------

class HybridSolver:
    """Two-phase Hybrid Solver: Backtracking -> Feasibility-Preserving SA."""

    def __init__(self, config: Optional[HybridConfig] = None) -> None:
        self.config = config if config is not None else HybridConfig()

    @classmethod
    def default(cls) -> "HybridSolver":
        """Default preset."""
        return cls(HybridConfig())

    def solve(self, problem: Problem, seed: int = 42) -> SolverResult:
        """Run Phase 1 Backtracking followed by Phase 2 Simulated Annealing.

        Args:
            problem: Validated Problem instance.
            seed:    Random seed for reproducibility.

        Returns:
            A populated SolverResult instance with detailed Phase 1 & 2 metrics.
        """
        rng = random.Random(seed)
        sessions, domains, _ = build_csp(problem, rng=rng)

        if not sessions:
            tt = Timetable(problem=problem, assignment={})
            res = SolverResult(
                timetable=tt,
                success=True,
                status="solved",
                time_seconds=0.0,
                nodes_expanded=0,
            )
            return fill_metrics(res)

        cfg = self.config
        total_start = time.perf_counter()

        # ==================================================================
        # PHASE 1: Backtracking Search (MRV + Degree + Forward Checking)
        # ==================================================================
        bt_solver = BacktrackingSolver(BacktrackingConfig(
            use_mrv=True,
            use_degree=True,
            use_forward_checking=True,
            time_limit=cfg.bt_time_limit,
        ))
        bt_res = bt_solver.solve(problem, seed=seed)

        # If Backtracking failed to find a feasible solution, terminate honestly
        if not bt_res.success or bt_res.timetable is None:
            bt_res.time_seconds = time.perf_counter() - total_start
            bt_res.details = {
                "phase1_time": bt_res.time_seconds,
                "phase1_nodes": bt_res.nodes_expanded,
                "phase2_time": 0.0,
                "phase2_iterations": 0,
                "soft_before": bt_res.soft_penalty,
                "soft_after": bt_res.soft_penalty,
            }
            return bt_res

        soft_before = bt_res.soft_penalty
        phase1_time = bt_res.time_seconds
        phase1_nodes = bt_res.nodes_expanded

        # ==================================================================
        # PHASE 2: Feasibility-Preserving Simulated Annealing
        # ==================================================================
        phase2_start = time.perf_counter()
        state = NeighbourhoodState(problem=problem, assignment=bt_res.timetable.assignment)
        assert state.current_hard_violations == 0, "Phase 1 solution must have 0 hard violations."

        # Calibrate initial temperature for refinement (based purely on soft penalty deltas)
        if cfg.t0 is not None:
            T = float(cfg.t0)
        else:
            sample_soft_deltas: List[int] = []
            sample_count = min(80, max(20, len(sessions)))
            for _ in range(sample_count):
                cand = generate_candidate_move(sessions, domains, state.assignment, rng, p_swap=cfg.p_swap)
                delta_cost, new_hard, _ = state.eval_delta_cost(cand)
                # Only consider moves that maintain feasibility (new_hard == 0)
                if new_hard == 0 and delta_cost > 0:
                    sample_soft_deltas.append(delta_cost)

            if sample_soft_deltas:
                avg_delta = sum(sample_soft_deltas) / len(sample_soft_deltas)
                target = max(0.05, min(0.95, cfg.target_initial_acceptance))
                T = max(0.5, -avg_delta / math.log(target))
            else:
                T = 5.0  # Mild refinement temperature

        moves_per_temp = cfg.moves_per_temperature
        if moves_per_temp is None:
            moves_per_temp = max(10, 2 * len(sessions))

        best_assignment: Dict[Session, AssignmentTuple] = dict(state.assignment)
        best_cost = state.current_cost
        best_soft = state.current_soft_penalty

        cost_history: List[int] = [state.current_cost]
        best_cost_history: List[int] = [best_cost]

        sa_iterations = 0
        stagnant_temp_levels = 0
        status: Literal["solved", "stopped_by_limit", "frozen"] = "solved"

        while sa_iterations < cfg.max_sa_iterations:
            if (time.perf_counter() - phase2_start) > cfg.sa_time_limit:
                break

            if T < cfg.t_min:
                break

            improved_this_temp = False

            for _ in range(moves_per_temp):
                sa_iterations += 1

                cand = generate_candidate_move(
                    sessions, domains, state.assignment, rng, p_swap=cfg.p_swap
                )

                delta_cost, new_hard, new_soft = state.eval_delta_cost(cand)

                # STRICT FEASIBILITY PRESERVATION:
                # Reject any move that introduces any hard violation!
                if new_hard > 0:
                    continue

                # Metropolis acceptance decision for soft penalty
                accept = False
                if delta_cost <= 0:
                    accept = True
                else:
                    prob = math.exp(-delta_cost / T)
                    if rng.random() < prob:
                        accept = True

                if accept:
                    state.apply_move(cand, delta_cost, new_hard, new_soft)

                    if cfg.debug_check:
                        assert state.current_hard_violations == 0, (
                            "Hard violation introduced in Phase 2!"
                        )

                    if state.current_cost < best_cost:
                        best_cost = state.current_cost
                        best_soft = state.current_soft_penalty
                        best_assignment = dict(state.assignment)
                        improved_this_temp = True

                if sa_iterations % cfg.k_cost_history == 0:
                    cost_history.append(state.current_cost)
                    best_cost_history.append(best_cost)

                if sa_iterations >= cfg.max_sa_iterations:
                    break

            if improved_this_temp:
                stagnant_temp_levels = 0
            else:
                stagnant_temp_levels += 1

            if stagnant_temp_levels >= cfg.plateau_temp_levels:
                break

            T = cfg.alpha * T

        phase2_time = time.perf_counter() - phase2_start
        total_time = time.perf_counter() - total_start

        final_timetable = Timetable(problem=problem, assignment=best_assignment)

        res = SolverResult(
            timetable=final_timetable,
            success=True,
            status="solved",
            time_seconds=total_time,
            nodes_expanded=phase1_nodes + sa_iterations,
            restarts=0,
            details={
                "phase1_time": phase1_time,
                "phase1_nodes": phase1_nodes,
                "phase2_time": phase2_time,
                "phase2_iterations": sa_iterations,
                "soft_before": soft_before,
                "soft_after": best_soft,
                "final_temperature": T,
                "best_cost_history": best_cost_history,
            },
        )
        fill_metrics(res)

        if not cost_history or cost_history[-1] != res.cost:
            cost_history.append(res.cost)
        res.cost_history = cost_history

        return res
