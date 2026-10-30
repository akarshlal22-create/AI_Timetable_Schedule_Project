"""algorithms/simulated_annealing.py — Simulated Annealing (SA) local search solver.

Design choices and justifications (for report and demo):
--------------------------------------------------------
1. Initial Solution:
   - A complete assignment where every variable is drawn from its unary-filtered domain.
   - Default: random complete assignment (explores from arbitrary state).
   - Option: ``initial_strategy="min_conflicts"`` runs Min-Conflicts first to achieve
     feasibility before annealing on soft penalties.

2. Energy / Cost Function:
   - Energy E = 1000 * hard_violations + soft_penalties (directly mirroring core.cost).
   - This provides a smooth objective landscape where escaping local minima is possible
     while heavily penalizing any clash, driving search towards 0 hard violations.

3. Neighbourhood Operators:
   - Combines single moves and pairwise swaps via algorithms.neighbourhood.
   - Swaps exchange slots between two compatible sessions (same duration: lab with lab,
     theory with theory; room type and capacity valid for both).
   - Fast O(1) delta-cost evaluation computes cost change strictly on affected entities.

4. Acceptance Rule:
   - Metropolis acceptance criterion:
       Δ = E(new) - E(current)
       If Δ <= 0: accept unconditionally (improving or neutral move).
       If Δ > 0: accept with probability P = exp(-Δ / T).

5. Initial Temperature Calibration (Kirkpatrick Method):
   - If T0 is not specified manually, it is auto-calibrated by sampling random moves
     from the initial state to measure average worsening delta (Δ_pos).
   - Given target initial acceptance p0 = 0.80:
       exp(-Δ_pos / T0) = p0  ==>  T0 = -Δ_pos / ln(p0).
   - This ensures roughly 80% of worsening moves are accepted at the start, preventing
     premature greedy freezing while allowing broad exploration.

6. Cooling Schedule:
   - Geometric cooling: T_{k+1} = alpha * T_k (default alpha = 0.995).
   - Moves per temperature level: defaults to 2 * len(sessions), giving each variable
     sufficient opportunity to explore at each temperature plateau.

7. Stopping Conditions:
   - Stop when T falls below T_min (frozen).
   - Stop when the best cost does not improve for ``plateau_temp_levels`` consecutive
     temperature levels (stagnation).
   - Stop when max iterations or time limit is reached.
   - Always return the BEST solution encountered during the entire search, not the last state.
"""

from __future__ import annotations

import math
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
class SimulatedAnnealingConfig:
    """Configuration parameters for SimulatedAnnealingSolver.

    Attributes:
        initial_strategy:           "random" (default) or "min_conflicts".
        t0:                         Initial temperature. If None, auto-calibrated.
        target_initial_acceptance:  Target acceptance rate for auto-calibration (default 0.80).
        alpha:                      Geometric cooling factor T = alpha * T (default 0.995).
        t_min:                      Minimum temperature to terminate (default 1e-3).
        moves_per_temperature:      Moves between cooling steps. If None, 2 * num_sessions.
        max_iterations:             Maximum total moves evaluated (default 50000).
        time_limit:                 Wall-clock seconds limit (default 60.0).
        plateau_temp_levels:        Stop if no best improvement for N temp levels (default 50).
        p_swap:                     Probability of proposing a swap vs single move (default 0.5).
        k_cost_history:             Record cost history every k iterations (default 1).
        debug_check:                If True, verifies delta cost against core.cost.evaluate.
    """
    initial_strategy: Literal["random", "min_conflicts"] = "random"
    t0: Optional[float] = None
    target_initial_acceptance: float = 0.80
    alpha: float = 0.995
    t_min: float = 1e-3
    moves_per_temperature: Optional[int] = None
    max_iterations: int = 50000
    time_limit: float = 60.0
    plateau_temp_levels: int = 50
    p_swap: float = 0.5
    k_cost_history: int = 1
    debug_check: bool = False


# ---------------------------------------------------------------------------
# Solver implementation
# ---------------------------------------------------------------------------

class SimulatedAnnealingSolver:
    """Simulated Annealing solver with Kirkpatrick temperature calibration."""

    def __init__(self, config: Optional[SimulatedAnnealingConfig] = None) -> None:
        self.config = config if config is not None else SimulatedAnnealingConfig()

    @classmethod
    def default(cls) -> "SimulatedAnnealingSolver":
        """Default preset."""
        return cls(SimulatedAnnealingConfig())

    def solve(self, problem: Problem, seed: int = 42) -> SolverResult:
        """Run Simulated Annealing on *problem*.

        Args:
            problem: Validated Problem instance.
            seed:    Random seed for reproducibility.

        Returns:
            A populated SolverResult instance.
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

        start_time = time.perf_counter()
        cfg = self.config

        # ------------------------------------------------------------------
        # 1. Initial Assignment
        # ------------------------------------------------------------------
        if cfg.initial_strategy == "min_conflicts":
            from algorithms.min_conflicts import MinConflictsConfig, MinConflictsSolver
            mc_solver = MinConflictsSolver(MinConflictsConfig(time_limit=cfg.time_limit / 2))
            mc_res = mc_solver.solve(problem, seed=seed)
            if mc_res.timetable is not None:
                assignment = dict(mc_res.timetable.assignment)
            else:
                assignment = {s: rng.choice(domains[s]) for s in sessions}
        else:
            assignment = {s: rng.choice(domains[s]) for s in sessions}

        # Setup neighbourhood state with lookup tables and delta cost
        state = NeighbourhoodState(problem=problem, assignment=assignment)

        # ------------------------------------------------------------------
        # 2. Initial Temperature Calibration (Kirkpatrick Method)
        # ------------------------------------------------------------------
        if cfg.t0 is not None:
            T = float(cfg.t0)
        else:
            # Sample random moves to measure average worsening delta
            sample_deltas: List[int] = []
            sample_moves = min(100, max(20, len(sessions)))
            conflicted_init = state.get_conflicted_sessions() if state.current_hard_violations > 0 else None
            for _ in range(sample_moves):
                cand = generate_candidate_move(
                    sessions, domains, state.assignment, rng, p_swap=cfg.p_swap,
                    conflicted_sessions=conflicted_init,
                )
                delta, _, _ = state.eval_delta_cost(cand)
                if delta > 0:
                    sample_deltas.append(delta)

            if sample_deltas:
                avg_delta = sum(sample_deltas) / len(sample_deltas)
                # P(accept) = exp(-avg_delta / T0) = target_acceptance
                # ln(target) = -avg_delta / T0  ==>  T0 = -avg_delta / ln(target)
                target = max(0.01, min(0.99, cfg.target_initial_acceptance))
                T = max(10.0, -avg_delta / math.log(target))
            else:
                T = 100.0

            # If starting from an infeasible solution, ensure T0 is warm enough to explore hard constraint transitions
            if state.current_hard_violations > 0:
                T = max(T, 100.0)

        # Moves per temperature
        moves_per_temp = cfg.moves_per_temperature
        if moves_per_temp is None:
            moves_per_temp = max(10, 2 * len(sessions))

        # ------------------------------------------------------------------
        # 3. Annealing State Tracking
        # ------------------------------------------------------------------
        best_assignment: Dict[Session, AssignmentTuple] = dict(state.assignment)
        best_cost = state.current_cost
        best_hard = state.current_hard_violations
        best_soft = state.current_soft_penalty

        cost_history: List[int] = [state.current_cost]
        best_cost_history: List[int] = [best_cost]

        total_iterations = 0
        stagnant_temp_levels = 0
        status: Literal["solved", "stopped_by_limit", "frozen"] = "stopped_by_limit"

        # ------------------------------------------------------------------
        # 4. Annealing Loop
        # ------------------------------------------------------------------
        while total_iterations < cfg.max_iterations:
            if (time.perf_counter() - start_time) > cfg.time_limit:
                status = "stopped_by_limit"
                break

            if T < cfg.t_min:
                status = "frozen"
                break

            improved_this_temp = False

            for _ in range(moves_per_temp):
                total_iterations += 1

                # Generate candidate move with conflict bias if hard clashes exist
                conflicted = state.get_conflicted_sessions() if state.current_hard_violations > 0 else None
                cand = generate_candidate_move(
                    sessions, domains, state.assignment, rng, p_swap=cfg.p_swap,
                    conflicted_sessions=conflicted,
                )

                # Evaluate delta cost in O(1)
                delta_cost, new_hard, new_soft = state.eval_delta_cost(cand)

                # Metropolis acceptance decision
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
                        state.check_recount()

                    # Track best-so-far
                    if state.current_cost < best_cost:
                        best_cost = state.current_cost
                        best_hard = state.current_hard_violations
                        best_soft = state.current_soft_penalty
                        best_assignment = dict(state.assignment)
                        improved_this_temp = True

                # Record cost history
                if total_iterations % cfg.k_cost_history == 0:
                    cost_history.append(state.current_cost)
                    best_cost_history.append(best_cost)

                if total_iterations >= cfg.max_iterations:
                    break

            # Check stagnation and apply geometric cooling
            if improved_this_temp:
                stagnant_temp_levels = 0
            else:
                stagnant_temp_levels += 1

            if stagnant_temp_levels >= cfg.plateau_temp_levels:
                status = "frozen"
                break

            T = cfg.alpha * T

        elapsed_time = time.perf_counter() - start_time

        # If zero hard violations was achieved at best, mark solved
        if best_hard == 0:
            status = "solved"
            success = True
        else:
            success = False

        # Build final timetable using the BEST assignment seen
        final_timetable = Timetable(problem=problem, assignment=best_assignment)

        res = SolverResult(
            timetable=final_timetable,
            success=success,
            status=status,
            time_seconds=elapsed_time,
            nodes_expanded=total_iterations,
            restarts=0,
            details={
                "final_temperature": T,
                "best_cost_history": best_cost_history,
            },
        )
        fill_metrics(res)

        # Guarantee cost_history ends with the reported final cost
        if not cost_history or cost_history[-1] != res.cost:
            cost_history.append(res.cost)
        res.cost_history = cost_history

        return res
