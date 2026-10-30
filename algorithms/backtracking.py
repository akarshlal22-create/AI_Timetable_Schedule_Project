"""algorithms/backtracking.py — Backtracking CSP solver with optional heuristics.

Design notes
------------
* We use an **iterative** search stack instead of Python recursion to avoid
  hitting the default recursion limit (1 000) on large problems that can have
  hundreds of variables.  The stack holds ``(session_index, assignment_so_far,
  domain_prune_trail)`` frames, which mirrors textbook recursive backtracking
  but without stack-frame overhead.

* Constraint checking is **incremental**: when we attempt to assign
  session *s* to slot *v*, we only check consistency against the already-
  assigned sessions (H1–H6, H9).  H7 (subject hours) and H8 (teacher load)
  hold by construction from :func:`core.problem_builder.build_csp` — each
  session is already counted exactly once, and teachers are pre-balanced.

* Forward checking uses a **trail/undo log** (a list of ``(session, value)``
  pairs representing pruned domain entries).  On backtrack we pop the trail
  back to the saved checkpoint, restoring pruned values without copying whole
  domain lists.

* Soft-constraint optimisation is deliberately absent.  The solver stops at
  the first feasible assignment.  Soft penalty is measured afterwards by
  :func:`algorithms.base.fill_metrics`.

Public API
----------
  BacktrackingConfig   — flags dataclass
  BacktrackingSolver   — the solver class; use .baseline() / .improved()
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from algorithms.base import SolverResult, fill_metrics
from core.models import (
    AssignmentTuple, Problem, Session, Timetable,
)
from core.problem_builder import build_csp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktrackingConfig:
    """Flags controlling solver behaviour.

    Attributes:
        use_mrv:              MRV variable ordering (fewest remaining values).
        use_degree:           Degree heuristic for tie-breaking (or primary order).
        use_forward_checking: Prune neighbours' domains after each assignment.
        use_lcv:              Least Constraining Value ordering (off by default).
        time_limit:           Wall-clock seconds before returning "timeout".
        node_limit:           Max assignment attempts before returning "node_limit".
                              ``None`` means unlimited.
    """
    use_mrv: bool = False
    use_degree: bool = False
    use_forward_checking: bool = False
    use_lcv: bool = False
    time_limit: float = 60.0
    node_limit: Optional[int] = None


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class BacktrackingSolver:
    """Backtracking CSP solver with configurable heuristics.

    Args:
        config: A :class:`BacktrackingConfig` instance.
    """

    def __init__(self, config: BacktrackingConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Named presets
    # ------------------------------------------------------------------

    @classmethod
    def baseline(cls) -> "BacktrackingSolver":
        """All heuristics off — fixed variable order, no pruning."""
        return cls(BacktrackingConfig())

    @classmethod
    def improved(cls) -> "BacktrackingSolver":
        """MRV + Degree + Forward Checking (recommended preset)."""
        return cls(BacktrackingConfig(
            use_mrv=True,
            use_degree=True,
            use_forward_checking=True,
        ))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self, problem: Problem, seed: int = 42) -> SolverResult:
        """Run backtracking search on *problem*.

        Args:
            problem: A validated :class:`~core.models.Problem`.
            seed:    Random seed for tie-breaking.

        Returns:
            A :class:`~algorithms.base.SolverResult`.
        """
        rng = random.Random(seed)
        sessions, domains, graph = build_csp(problem, rng=rng)

        # Working copies of domains (lists, mutable)
        live_domains: Dict[Session, List[AssignmentTuple]] = {
            s: list(d) for s, d in domains.items()
        }

        start = time.perf_counter()
        nodes = 0
        cfg = self.config

        # --- iterative backtracking stack ---
        # Each frame: (session_idx, value_iterator_index, assignment_snapshot_size,
        #              trail_checkpoint)
        # We store the variable order once; heuristics may reorder it each time we
        # pick the next unassigned variable (MRV/Degree) or keep it fixed (baseline).
        assignment: Dict[Session, AssignmentTuple] = {}
        # Trail: list of (session, value) that were pruned by forward-checking.
        trail: List[Tuple[Session, AssignmentTuple]] = []

        # For fixed-order baseline we precompute the order once.
        fixed_order: List[Session] = list(sessions)

        # Stack frame: (session being tried, index into its domain list, trail checkpoint)
        stack: List[Tuple[Session, int, int]] = []

        def elapsed() -> float:
            return time.perf_counter() - start

        def pick_unassigned() -> Optional[Session]:
            """Choose the next variable to assign."""
            unassigned = [s for s in sessions if s not in assignment]
            if not unassigned:
                return None
            if not cfg.use_mrv and not cfg.use_degree:
                # Fixed order: return first unassigned in fixed_order
                for s in fixed_order:
                    if s not in assignment:
                        return s
                return None
            if cfg.use_mrv:
                min_len = min(len(live_domains[s]) for s in unassigned)
                candidates = [s for s in unassigned if len(live_domains[s]) == min_len]
            else:
                candidates = unassigned
            if cfg.use_degree and len(candidates) > 1:
                # Degree = number of unassigned neighbours
                unassigned_set: Set[Session] = set(unassigned)
                def degree(s: Session) -> int:
                    return sum(1 for nb in graph[s] if nb in unassigned_set)
                candidates.sort(key=degree, reverse=True)
            # Shuffle among equal candidates for reproducibility
            if len(candidates) > 1:
                rng.shuffle(candidates)
            return candidates[0]

        def order_values(sess: Session) -> List[AssignmentTuple]:
            """Return domain values in search order."""
            vals = list(live_domains[sess])
            if not cfg.use_lcv or not vals:
                return vals
            # LCV: prefer values that prune the fewest neighbour domain entries
            def lcv_score(v: AssignmentTuple) -> int:
                return sum(
                    1 for nb in graph[sess]
                    if nb not in assignment
                    for nv in live_domains[nb]
                    if _conflicts(sess, v, nb, nv)
                )
            vals.sort(key=lcv_score)
            return vals

        def _conflicts(
            s1: Session, v1: AssignmentTuple,
            s2: Session, v2: AssignmentTuple,
        ) -> bool:
            """Return True when assigning v1 to s1 conflicts with v2 for s2."""
            d1, p1, r1 = v1
            d2, p2, r2 = v2
            # Slots occupied by each session
            slots1 = [(d1, p1), (d1, p1 + 1)] if s1.is_lab else [(d1, p1)]
            slots2 = [(d2, p2), (d2, p2 + 1)] if s2.is_lab else [(d2, p2)]
            slot_set1 = set(slots1)
            slot_set2 = set(slots2)
            overlap = slot_set1 & slot_set2
            if not overlap:
                return False
            # H1 teacher clash
            if s1.teacher_name == s2.teacher_name:
                return True
            # H2 class clash
            if s1.class_name == s2.class_name:
                return True
            # H3 room clash
            if r1 == r2:
                return True
            return False

        def is_consistent(sess: Session, val: AssignmentTuple) -> bool:
            """Check val for sess against all already-assigned sessions (H1–H6, H9)."""
            d, p, room = val
            p = p  # local alias
            slots_new = [(d, p), (d, p + 1)] if sess.is_lab else [(d, p)]
            problem_ref = problem

            # H4 room capacity (also filtered in domain, but check for safety)
            cls = problem_ref.classes[sess.class_name]
            rm = problem_ref.rooms[room]
            if rm.capacity < cls.strength:
                return False

            # H5 room type (also in domain; safety check)
            expected_room_type = "lab" if sess.is_lab else "classroom"
            if rm.type != expected_room_type:
                return False

            # H6 teacher availability
            teacher = problem_ref.teachers[sess.teacher_name]
            for slot in slots_new:
                if slot in teacher.unavailable_slots:
                    return False

            # H9 lunch slot
            for (_, period) in slots_new:
                if period == cls.lunch_slot:
                    return False

            # H1 / H2 / H3 against assigned sessions
            slot_set_new = set(slots_new)
            for other_sess, (od, op, oroom) in assignment.items():
                other_slots = [(od, op), (od, op + 1)] if other_sess.is_lab else [(od, op)]
                overlap = slot_set_new & set(other_slots)
                if not overlap:
                    continue
                if other_sess.teacher_name == sess.teacher_name:
                    return False  # H1
                if other_sess.class_name == sess.class_name:
                    return False  # H2
                if oroom == room:
                    return False  # H3
            return True

        def forward_check(sess: Session, val: AssignmentTuple) -> bool:
            """Prune domains of unassigned neighbours; return False if any go empty.

            Pruned values are recorded on ``trail`` so backtracking can undo them.
            Returns True if all neighbours still have non-empty domains.
            """
            for nb in graph[sess]:
                if nb in assignment:
                    continue
                to_remove = [
                    nv for nv in live_domains[nb]
                    if _conflicts(sess, val, nb, nv)
                ]
                for nv in to_remove:
                    live_domains[nb].remove(nv)
                    trail.append((nb, nv))
                if not live_domains[nb]:
                    return False  # domain wipe-out
            return True

        def undo_trail(checkpoint: int) -> None:
            """Restore pruned values back to checkpoint on the trail."""
            while len(trail) > checkpoint:
                nb, nv = trail.pop()
                live_domains[nb].append(nv)

        # ------------------------------------------------------------------
        # Main iterative search
        # ------------------------------------------------------------------
        # Instead of a classic recursive DFS, we use an explicit stack.
        # Each stack entry is (session, ordered_values_list, trail_checkpoint).
        # We push when we descend, pop+undo when we backtrack.

        # Bootstrap: pick first variable
        first = pick_unassigned()
        if first is None:
            # No sessions at all — trivially solved
            tt = Timetable(problem=problem, assignment={})
            result = SolverResult(
                timetable=tt, success=True, status="solved",
                time_seconds=elapsed(), nodes_expanded=0,
            )
            return fill_metrics(result)

        stack.append((first, order_values(first), len(trail)))

        solution: Optional[Dict[Session, AssignmentTuple]] = None
        final_status: str = "no_solution"

        while stack:
            # Time/node limit checks
            if elapsed() > cfg.time_limit:
                final_status = "timeout"
                break
            if cfg.node_limit is not None and nodes >= cfg.node_limit:
                final_status = "node_limit"
                break

            sess, values, trail_cp = stack[-1]

            if not values:
                # All values for this variable exhausted — backtrack
                stack.pop()
                if sess in assignment:
                    del assignment[sess]
                undo_trail(trail_cp)
                continue

            # Try the next value
            val = values.pop(0)
            nodes += 1

            if not is_consistent(sess, val):
                continue

            # Tentative assignment
            assignment[sess] = val
            trail_cp_after = len(trail)

            # Forward checking
            if cfg.use_forward_checking:
                if not forward_check(sess, val):
                    # Wipe-out — undo FC pruning and try next value
                    undo_trail(trail_cp_after)
                    del assignment[sess]
                    continue

            # Pick next variable
            nxt = pick_unassigned()
            if nxt is None:
                # All variables assigned — solution found
                solution = dict(assignment)
                final_status = "solved"
                break

            # Descend
            stack.append((nxt, order_values(nxt), len(trail)))

        elapsed_time = elapsed()

        if solution is not None:
            tt = Timetable(problem=problem, assignment=solution)
            result = SolverResult(
                timetable=tt,
                success=True,
                status="solved",
                time_seconds=elapsed_time,
                nodes_expanded=nodes,
            )
        else:
            # Partial timetable for reporting (best partial = longest assignment)
            partial = Timetable(problem=problem, assignment=dict(assignment)) if assignment else None
            result = SolverResult(
                timetable=partial,
                success=False,
                status=final_status,  # type: ignore[arg-type]
                time_seconds=elapsed_time,
                nodes_expanded=nodes,
            )

        return fill_metrics(result)
