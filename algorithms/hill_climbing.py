"""algorithms/hill_climbing.py — Hill Climbing local search solver.

Design choices and justifications (for report and demo):
--------------------------------------------------------
1. Initial Solution:
   - A complete assignment where every variable is drawn from its unary-filtered domain.
   - Default: random complete assignment (explores from an arbitrary state, same default as SA).
   - Option: ``initial_strategy="min_conflicts"`` runs Min-Conflicts first to achieve
     initial feasibility before climbing for soft penalty reduction.

2. Cost Function:
   - Cost = 1000 * hard_violations + soft_penalties (directly mirroring core.cost).
   - Any hard constraint violation carries a 1000 penalty weight, heavily penalising clashes.

3. Step Operator (Sampled Best-Improvement):
   - In each step, samples up to `neighbours_per_step` candidate neighbours (move or swap)
     from algorithms.neighbourhood.
   - Evaluates each candidate's delta-cost using the fast O(1) incremental evaluation
     in NeighbourhoodState.
   - Moves to the candidate with the lowest cost ONLY IF it is STRICTLY BETTER (delta_cost < 0)
     than the current solution.
   - Crucial distinction from SA: Never accepts neutral (delta_cost == 0) or worsening
     (delta_cost > 0) moves. This makes Hill Climbing purely greedy and vulnerable to
     getting trapped in local optima.

4. Stopping Rule:
   - Terminates when no strictly improving move is found among sampled neighbours for
     `patience` consecutive steps (indicating a local optimum / plateau).
   - Terminates if `max_iterations` or `time_limit` (default 60s) is exceeded.
   - Terminates early if cost reaches 0 (global optimum with 0 hard violations and 0 soft penalties).
   - Stops at a local optimum with hard violations > 0 are reported as UNSUCCESSFUL
     (success = False, status = "local_optimum").

5. Optional Random Restarts:
   - Controlled by `max_restarts` (default 0 for pure hill climbing).
   - When restarts > 0, upon hitting a local optimum, the solver draws a fresh initial
     state up to `max_restarts` times, tracking the global best-so-far solution across all restarts.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

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
class HillClimbingConfig:
    """Configuration parameters for HillClimbingSolver.

    Attributes:
        time_limit:           Maximum wall-clock execution time in seconds (default 60.0).
        max_iterations:       Maximum iterations per climb (default 20000).
        neighbours_per_step:  Number of random neighbours sampled per step (default 50).
        patience:             Consecutive non-improving steps before declaring local optimum
                              (default None: auto-scales to 2 * num_sessions).
        max_restarts:         Number of random restarts allowed (default 0: pure hill climbing).
        p_swap:               Probability of generating a swap vs move (default 0.5).
        initial_strategy:     "random" or "min_conflicts" (default "random").
        k_cost_history:       Record cost history every k steps (default 1).
        debug_check:          Periodically recount costs from scratch to test incremental state.
    """
    time_limit: float = 60.0
    max_iterations: int = 20000
    neighbours_per_step: int = 50
    patience: Optional[int] = None
    max_restarts: int = 0
    p_swap: float = 0.5
    initial_strategy: Literal["random", "min_conflicts"] = "random"
    k_cost_history: int = 1
    debug_check: bool = False


# ---------------------------------------------------------------------------
# Solver Implementation
# ---------------------------------------------------------------------------

class HillClimbingSolver:
    """Hill Climbing local search solver with optional random restarts."""

    def __init__(self, config: Optional[HillClimbingConfig] = None) -> None:
        self.config = config if config is not None else HillClimbingConfig()

    @classmethod
    def default(cls) -> "HillClimbingSolver":
        """Default preset."""
        return cls(HillClimbingConfig())

    def solve(self, problem: Problem, seed: Optional[int] = None) -> SolverResult:
        """Execute Hill Climbing on *problem* with optional random restarts.

        Args:
            problem: A validated :class:`~core.models.Problem` instance.
            seed:    Random seed for reproducibility.

        Returns:
            A :class:`~algorithms.base.SolverResult` tracking success, time, iterations,
            violations, and cost history.
        """
        cfg = self.config
        actual_seed = seed if seed is not None else problem.seed
        rng = random.Random(actual_seed)
        start_time = time.perf_counter()

        sessions, domains, _ = build_csp(problem, rng=rng)
        num_sessions = len(sessions)
        patience_limit = cfg.patience if cfg.patience is not None else max(10, 2 * num_sessions)

        global_best_assignment: Optional[Dict[Session, AssignmentTuple]] = None
        global_best_cost = float("inf")
        global_best_hard = float("inf")
        global_best_soft = float("inf")
        global_best_status: str = "local_optimum"

        total_iterations = 0
        restarts_used = 0
        overall_cost_history: List[int] = []

        max_restarts = max(0, cfg.max_restarts)

        for restart_idx in range(max_restarts + 1):
            if restart_idx > 0:
                restarts_used += 1

            if (time.perf_counter() - start_time) >= cfg.time_limit:
                if global_best_assignment is None:
                    global_best_status = "timeout"
                break

            # --------------------------------------------------------------
            # Initial Solution construction
            # --------------------------------------------------------------
            if cfg.initial_strategy == "min_conflicts":
                from algorithms.min_conflicts import MinConflictsConfig, MinConflictsSolver
                mc_limit = min(5.0, cfg.time_limit * 0.25)
                mc = MinConflictsSolver(MinConflictsConfig(time_limit=mc_limit))
                mc_res = mc.solve(problem, seed=rng.randint(0, 1_000_000))
                if mc_res.timetable is not None and len(mc_res.timetable.assignment) == num_sessions:
                    init_assignment = dict(mc_res.timetable.assignment)
                else:
                    init_assignment = {s: rng.choice(domains[s]) for s in sessions}
            else:
                init_assignment = {s: rng.choice(domains[s]) for s in sessions}

            state = NeighbourhoodState(problem=problem, assignment=init_assignment)

            if not overall_cost_history:
                overall_cost_history.append(state.current_cost)

            # Check if this initial state is already better than global best
            if state.current_cost < global_best_cost:
                global_best_cost = state.current_cost
                global_best_hard = state.current_hard_violations
                global_best_soft = state.current_soft_penalty
                global_best_assignment = dict(state.assignment)

            consecutive_no_improve = 0
            climb_status = "local_optimum"

            # --------------------------------------------------------------
            # Hill Climbing Loop
            # --------------------------------------------------------------
            while total_iterations < cfg.max_iterations:
                if (time.perf_counter() - start_time) >= cfg.time_limit:
                    climb_status = "timeout"
                    break

                if state.current_cost == 0:
                    climb_status = "solved"
                    break

                total_iterations += 1

                # Sample up to `neighbours_per_step` candidate neighbours
                best_cand = None
                best_delta = 0
                best_new_hard = state.current_hard_violations
                best_new_soft = state.current_soft_penalty

                conflicted = state.get_conflicted_sessions() if state.current_hard_violations > 0 else None

                for _ in range(cfg.neighbours_per_step):
                    cand = generate_candidate_move(
                        sessions, domains, state.assignment, rng,
                        p_swap=cfg.p_swap,
                        conflicted_sessions=conflicted,
                    )
                    delta_cost, new_hard, new_soft = state.eval_delta_cost(cand)

                    # STRICTLY BETTER: strictly lower cost
                    if delta_cost < best_delta:
                        best_delta = delta_cost
                        best_cand = cand
                        best_new_hard = new_hard
                        best_new_soft = new_soft

                # Only move if a strictly improving neighbour was found
                if best_cand is not None and best_delta < 0:
                    state.apply_move(best_cand, best_delta, best_new_hard, best_new_soft)
                    consecutive_no_improve = 0

                    if cfg.debug_check:
                        state.check_recount()

                    if state.current_cost < global_best_cost:
                        global_best_cost = state.current_cost
                        global_best_hard = state.current_hard_violations
                        global_best_soft = state.current_soft_penalty
                        global_best_assignment = dict(state.assignment)
                else:
                    consecutive_no_improve += 1

                # Record cost history (guaranteed non-increasing during a single climb)
                if total_iterations % cfg.k_cost_history == 0:
                    overall_cost_history.append(state.current_cost)

                # Check stopping patience (stuck in local optimum)
                if consecutive_no_improve >= patience_limit:
                    climb_status = "local_optimum"
                    break

            if total_iterations >= cfg.max_iterations and climb_status == "local_optimum":
                climb_status = "iteration_limit"

            # Update overall status
            if global_best_hard == 0:
                global_best_status = "solved"
            else:
                global_best_status = climb_status

            # If global optimal achieved with 0 violations and 0 soft penalties, terminate early
            if global_best_cost == 0:
                break

        elapsed_time = time.perf_counter() - start_time
        assert global_best_assignment is not None

        # Success requires exactly 0 hard violations
        is_success = (global_best_hard == 0)
        final_status = "solved" if is_success else global_best_status

        final_timetable = Timetable(problem=problem, assignment=global_best_assignment)

        res = SolverResult(
            timetable=final_timetable,
            success=is_success,
            status=final_status,
            time_seconds=elapsed_time,
            nodes_expanded=total_iterations,
            hard_violations=int(global_best_hard),
            soft_penalty=int(global_best_soft),
            cost=int(global_best_cost),
            cost_history=overall_cost_history,
            details={
                "restarts_used": restarts_used,
                "max_restarts": cfg.max_restarts,
                "neighbours_per_step": cfg.neighbours_per_step,
                "patience": patience_limit,
            },
        )
        return fill_metrics(res)
