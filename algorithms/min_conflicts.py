"""algorithms/min_conflicts.py — Min-Conflicts local search CSP solver.

Design choices and justifications (for report and demo):
--------------------------------------------------------
1. Initial Assignment:
   - Complete assignment where every variable is assigned a value from its
     unary-filtered domain (satisfying H4 room capacity, H5 room type,
     H6 teacher availability, H9 lunch slot, and breaks by construction).
   - Default: random selection per variable (exploration from diverse states).
   - Config flag ``initial_strategy="greedy"``: assigns each variable in turn
     to the value minimizing conflicts with previously assigned variables.

2. Conflicted Variable Selection:
   - A variable is conflicted if it participates in at least one hard conflict
     (H1 teacher clash, H2 class clash, or H3 room clash).
   - Conflicted variables are maintained incrementally in a set (O(1) updates)
     without rescanning the entire timetable on each step.
   - On each iteration, a conflicted variable is chosen uniformly at random (seeded).

3. Value Selection & Tie-Breaking:
   - We move the chosen variable to the value in its domain that minimizes
     the number of hard conflicts (excluding its current value).
   - Tie-breaking: when multiple domain values achieve the minimal hard conflicts,
     we prefer the value with the lower soft-constraint penalty (S1–S5 evaluated
     via core/constraints). Remaining ties are broken uniformly at random (seeded).

4. Plateau Escape & Restarts:
   - Random Walk: with probability ``p_walk`` (default 0.02), we take a random-walk
     step by picking a random alternate value in the variable's domain to escape local minima.
   - Random Restarts: if the best hard violation count does not improve for
     ``restart_threshold`` iterations (default 2000), the search restarts from a
     fresh initial assignment while preserving the best assignment found so far.

5. Incremental (Delta) Evaluation & Verification:
   - Occupancy lookup tables (teacher_occ, class_occ, room_occ) map each resource
     and slot to the set of assigned sessions.
   - Lab sessions occupy 2 consecutive periods (p and p+1), handled across all
     lookup tables and conflict counts.
   - Debug check flag verifies that the incremental conflicted set matches
     a full recount from ``core.cost.conflicts()``.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

from algorithms.base import SolverResult, fill_metrics
from core.constraints import (
    penalty_s1_class_gaps,
    penalty_s2_subject_spread,
    penalty_s3_lab_consecutive,
    penalty_s4_teacher_preferences,
    penalty_s5_teacher_consecutive,
)
from core.cost import conflicts, evaluate
from core.models import AssignmentTuple, Problem, Session, SlotTuple, Timetable
from core.problem_builder import build_csp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MinConflictsConfig:
    """Configuration flags for MinConflictsSolver.

    Attributes:
        initial_strategy:  "random" (default) or "greedy".
        p_walk:            Probability of taking a random-walk step (default 0.02).
        restart_threshold: Max iterations without improvement before restart (default 2000).
        max_iterations:    Maximum total iterations (default 50000).
        time_limit:        Wall-clock time limit in seconds (default 60.0).
        k_cost_history:    Record cost_history every k iterations (default 1).
        debug_check:       If True, asserts incremental conflicts match core.cost.conflicts.
    """
    initial_strategy: Literal["random", "greedy"] = "random"
    p_walk: float = 0.02
    restart_threshold: int = 2000
    max_iterations: int = 50000
    time_limit: float = 60.0
    k_cost_history: int = 1
    debug_check: bool = False


# ---------------------------------------------------------------------------
# Solver implementation
# ---------------------------------------------------------------------------

class MinConflictsSolver:
    """Min-Conflicts local search solver implementing the Solver interface."""

    def __init__(self, config: Optional[MinConflictsConfig] = None) -> None:
        self.config = config if config is not None else MinConflictsConfig()

    @classmethod
    def default(cls) -> "MinConflictsSolver":
        """Default configuration preset."""
        return cls(MinConflictsConfig())

    @classmethod
    def greedy(cls) -> "MinConflictsSolver":
        """Greedy initialization preset."""
        return cls(MinConflictsConfig(initial_strategy="greedy"))

    def solve(self, problem: Problem, seed: int = 42) -> SolverResult:
        """Solve *problem* using Min-Conflicts local search.

        Args:
            problem: Validated Problem instance.
            seed:    Random seed for reproducibility.

        Returns:
            A populated SolverResult instance.
        """
        rng = random.Random(seed)
        sessions, domains, _ = build_csp(problem, rng=rng)

        # Edge case: no sessions
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
        # Helper: slot expansion for a session (labs occupy 2 consecutive periods)
        # ------------------------------------------------------------------
        def slots_of(s: Session, val: AssignmentTuple) -> List[SlotTuple]:
            d, p, _ = val
            if s.is_lab:
                return [(d, p), (d, p + 1)]
            return [(d, p)]

        # ------------------------------------------------------------------
        # Lookup tables for resource occupancy
        # ------------------------------------------------------------------
        teacher_occ: Dict[Tuple[str, SlotTuple], Set[Session]] = defaultdict(set)
        class_occ: Dict[Tuple[str, SlotTuple], Set[Session]] = defaultdict(set)
        room_occ: Dict[Tuple[str, SlotTuple], Set[Session]] = defaultdict(set)

        assignment: Dict[Session, AssignmentTuple] = {}
        conflicted_vars: Set[Session] = set()

        def clear_occupancy() -> None:
            teacher_occ.clear()
            class_occ.clear()
            room_occ.clear()
            assignment.clear()
            conflicted_vars.clear()

        # ------------------------------------------------------------------
        # Initial assignment generation
        # ------------------------------------------------------------------
        def init_assignment() -> None:
            clear_occupancy()
            if cfg.initial_strategy == "greedy":
                for s in sessions:
                    dom = domains[s]
                    if not dom:
                        continue
                    # Count clashes for each value against already-placed sessions
                    def count_initial_clashes(val: AssignmentTuple) -> int:
                        cnt = 0
                        for sl in slots_of(s, val):
                            cnt += len(teacher_occ[(s.teacher_name, sl)])
                            cnt += len(class_occ[(s.class_name, sl)])
                            cnt += len(room_occ[(val[2], sl)])
                        return cnt

                    min_c = min(count_initial_clashes(v) for v in dom)
                    best_vals = [v for v in dom if count_initial_clashes(v) == min_c]
                    chosen_val = rng.choice(best_vals)
                    assignment[s] = chosen_val
                    for sl in slots_of(s, chosen_val):
                        teacher_occ[(s.teacher_name, sl)].add(s)
                        class_occ[(s.class_name, sl)].add(s)
                        room_occ[(chosen_val[2], sl)].add(s)
            else:
                for s in sessions:
                    val = rng.choice(domains[s])
                    assignment[s] = val
                    for sl in slots_of(s, val):
                        teacher_occ[(s.teacher_name, sl)].add(s)
                        class_occ[(s.class_name, sl)].add(s)
                        room_occ[(val[2], sl)].add(s)

            # Populate initial conflicted set
            for s in sessions:
                val = assignment[s]
                for sl in slots_of(s, val):
                    if (
                        len(teacher_occ[(s.teacher_name, sl)]) > 1
                        or len(class_occ[(s.class_name, sl)]) > 1
                        or len(room_occ[(val[2], sl)]) > 1
                    ):
                        conflicted_vars.add(s)
                        break

        init_assignment()

        # Track best assignment found
        best_assignment: Dict[Session, AssignmentTuple] = dict(assignment)
        best_conflicted_count: int = len(conflicted_vars)
        restarts: int = 0
        iterations_without_improvement: int = 0

        # Cost history
        cost_history: List[int] = []
        tt_wrapper = Timetable(problem=problem, assignment=assignment)
        cost_history.append(evaluate(tt_wrapper, complete=True).total_cost)

        # ------------------------------------------------------------------
        # Helper: Soft penalty evaluation for tie-breaking
        # ------------------------------------------------------------------
        def calc_soft_penalty() -> int:
            return (
                penalty_s1_class_gaps(tt_wrapper)
                + penalty_s2_subject_spread(tt_wrapper)
                + penalty_s3_lab_consecutive(tt_wrapper)
                + penalty_s4_teacher_preferences(tt_wrapper)
                + penalty_s5_teacher_consecutive(tt_wrapper)
            )

        status: Literal["solved", "iteration_limit", "timeout"] = "iteration_limit"
        total_iterations = 0

        # ------------------------------------------------------------------
        # Main Min-Conflicts loop
        # ------------------------------------------------------------------
        while total_iterations < cfg.max_iterations:
            if not conflicted_vars:
                status = "solved"
                break

            if (time.perf_counter() - start_time) > cfg.time_limit:
                status = "timeout"
                break

            total_iterations += 1

            # Pick a conflicted variable uniformly at random (seeded)
            var = rng.choice(tuple(conflicted_vars))
            cur_val = assignment[var]
            var_domain = domains[var]
            cands = [v for v in var_domain if v != cur_val]

            if not cands:
                # If only one value exists in domain, cannot change
                continue

            # Plateau escape: random walk step
            if rng.random() < cfg.p_walk:
                new_val = rng.choice(cands)
            else:
                # Value selection: evaluate candidate values
                def val_conflicts(v: AssignmentTuple) -> int:
                    cnt = 0
                    for sl in slots_of(var, v):
                        cnt += len(teacher_occ[(var.teacher_name, sl)] - {var})
                        cnt += len(class_occ[(var.class_name, sl)] - {var})
                        cnt += len(room_occ[(v[2], sl)] - {var})
                    return cnt

                min_c = min(val_conflicts(v) for v in cands)
                tied_vals = [v for v in cands if val_conflicts(v) == min_c]

                if len(tied_vals) == 1:
                    new_val = tied_vals[0]
                else:
                    # Tie-breaking by soft-constraint penalty
                    soft_penalties: List[int] = []
                    for v in tied_vals:
                        assignment[var] = v
                        soft_penalties.append(calc_soft_penalty())
                    assignment[var] = cur_val  # restore

                    min_soft = min(soft_penalties)
                    best_soft_vals = [
                        v for v, sp in zip(tied_vals, soft_penalties) if sp == min_soft
                    ]
                    new_val = rng.choice(best_soft_vals)

            # Move var to new_val and incrementally update occupancies
            affected: Set[Session] = {var}

            # Remove from old slots
            for sl in slots_of(var, cur_val):
                affected.update(teacher_occ[(var.teacher_name, sl)])
                affected.update(class_occ[(var.class_name, sl)])
                affected.update(room_occ[(cur_val[2], sl)])
                teacher_occ[(var.teacher_name, sl)].remove(var)
                class_occ[(var.class_name, sl)].remove(var)
                room_occ[(cur_val[2], sl)].remove(var)

            # Apply new assignment
            assignment[var] = new_val

            # Add to new slots
            for sl in slots_of(var, new_val):
                teacher_occ[(var.teacher_name, sl)].add(var)
                class_occ[(var.class_name, sl)].add(var)
                room_occ[(new_val[2], sl)].add(var)
                affected.update(teacher_occ[(var.teacher_name, sl)])
                affected.update(class_occ[(var.class_name, sl)])
                affected.update(room_occ[(new_val[2], sl)])

            # Recompute conflict status only for affected variables
            for s in affected:
                s_val = assignment[s]
                is_conf = False
                for sl in slots_of(s, s_val):
                    if (
                        len(teacher_occ[(s.teacher_name, sl)]) > 1
                        or len(class_occ[(s.class_name, sl)]) > 1
                        or len(room_occ[(s_val[2], sl)]) > 1
                    ):
                        is_conf = True
                        break
                if is_conf:
                    conflicted_vars.add(s)
                else:
                    conflicted_vars.discard(s)

            # Debug correctness check
            if cfg.debug_check:
                expected_conflicted = {s for s, _ in conflicts(tt_wrapper)}
                assert conflicted_vars == expected_conflicted, (
                    f"Mismatch at iteration {total_iterations}: "
                    f"{conflicted_vars} vs {expected_conflicted}"
                )

            # Record cost_history
            if total_iterations % cfg.k_cost_history == 0:
                cost_history.append(evaluate(tt_wrapper, complete=True).total_cost)

            # Check improvement and manage random restarts
            cur_conflicted_count = len(conflicted_vars)
            if cur_conflicted_count < best_conflicted_count:
                best_conflicted_count = cur_conflicted_count
                best_assignment = dict(assignment)
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            if iterations_without_improvement >= cfg.restart_threshold:
                restarts += 1
                init_assignment()
                iterations_without_improvement = 0
                if len(conflicted_vars) < best_conflicted_count:
                    best_conflicted_count = len(conflicted_vars)
                    best_assignment = dict(assignment)

        # Check final solved condition
        if not conflicted_vars:
            status = "solved"

        elapsed_time = time.perf_counter() - start_time

        if status == "solved":
            final_timetable = Timetable(problem=problem, assignment=dict(assignment))
            success = True
        else:
            final_timetable = Timetable(problem=problem, assignment=dict(best_assignment))
            success = False

        res = SolverResult(
            timetable=final_timetable,
            success=success,
            status=status,
            time_seconds=elapsed_time,
            nodes_expanded=total_iterations,
            restarts=restarts,
        )
        fill_metrics(res)

        # Ensure cost_history is non-empty and its last value matches final cost
        if not cost_history or cost_history[-1] != res.cost:
            cost_history.append(res.cost)
        res.cost_history = cost_history

        return res

    # ------------------------------------------------------------------
    # Helper for testing / verification
    # ------------------------------------------------------------------
    @staticmethod
    def count_conflicts(timetable: Timetable) -> int:
        """Count total hard clash interactions using core.cost.conflicts."""
        return len(conflicts(timetable))
