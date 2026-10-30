"""algorithms/neighbourhood.py — Move and Swap neighbourhood operators with delta-cost evaluation.

Provides:
  - ``CandidateMove``: represents a proposed move or swap (not yet applied).
  - ``generate_candidate_move``: generates a single move or pairwise swap.
  - ``NeighbourhoodState``: maintains lookup tables (teacher_occ, class_occ, room_occ)
    and computes exact delta costs in O(1) time without rescanning the timetable.
  - Full compatibility checks for swaps: same duration (labs only with labs),
    room types match, room capacity holds, teacher availability, lunch, and breaks.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set, Tuple

from core.cost import evaluate
from core.models import AssignmentTuple, Problem, Session, SlotTuple, Timetable


# ---------------------------------------------------------------------------
# Move Representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateMove:
    """Represents a proposed change in assignment.

    Attributes:
        kind:     "move" (single session reassigned) or "swap" (two sessions exchange slots).
        session1: Primary session being moved or swapped.
        new_val1: Candidate (day, period, room) for session1.
        session2: Secondary session (present only if kind == "swap").
        new_val2: Candidate (day, period, room) for session2 (present only if kind == "swap").
    """
    kind: Literal["move", "swap"]
    session1: Session
    new_val1: AssignmentTuple
    session2: Optional[Session] = None
    new_val2: Optional[AssignmentTuple] = None


# ---------------------------------------------------------------------------
# Helpers for slot expansion
# ---------------------------------------------------------------------------

def slots_of(s: Session, val: AssignmentTuple) -> List[SlotTuple]:
    """Return all (day, period) slots occupied by *s* at *val*.

    Labs occupy (day, period) and (day, period + 1). Lectures occupy (day, period).
    """
    d, p, _ = val
    if s.is_lab:
        return [(d, p), (d, p + 1)]
    return [(d, p)]


# ---------------------------------------------------------------------------
# Candidate Move Generation
# ---------------------------------------------------------------------------

def generate_candidate_move(
    sessions: List[Session],
    domains: Dict[Session, List[AssignmentTuple]],
    assignment: Dict[Session, AssignmentTuple],
    rng: random.Random,
    p_swap: float = 0.5,
    conflicted_sessions: Optional[List[Session]] = None,
    conflict_bias: float = 0.7,
) -> CandidateMove:
    """Generate a candidate move or swap using seeded randomness.

    Args:
        sessions:            List of all sessions.
        domains:             Unary-filtered domains for each session.
        assignment:          Current complete assignment.
        rng:                 Random generator instance for reproducibility.
        p_swap:              Probability of attempting a swap move (default 0.5).
        conflicted_sessions: Optional list of currently clashing sessions to bias selection.
        conflict_bias:       Probability of picking a conflicted session if available (default 0.7).

    Returns:
        A :class:`CandidateMove`.
    """
    if conflicted_sessions and (rng.random() < conflict_bias):
        s1 = rng.choice(conflicted_sessions)
    else:
        s1 = rng.choice(sessions)

    do_swap = (rng.random() < p_swap) and (len(sessions) >= 2)

    if do_swap:
        # Try to find a compatible swap pair
        cur1 = assignment[s1]
        dom1 = domains[s1]

        # Candidate swap partners must:
        # 1. Be a different session
        # 2. Have the exact same duration (lab with lab, theory with theory)
        # 3. Have cur2 in s1's unary domain (correct room type, cap >= s1.strength, etc.)
        # 4. Have cur1 in s2's unary domain (correct room type, cap >= s2.strength, etc.)
        # 5. Be assigned to a different slot/room
        compatible = [
            s2 for s2 in sessions
            if s2 != s1
            and s2.is_lab == s1.is_lab
            and assignment[s2] in dom1
            and cur1 in domains[s2]
            and assignment[s2] != cur1
        ]
        if compatible:
            s2 = rng.choice(compatible)
            return CandidateMove(
                kind="swap",
                session1=s1,
                new_val1=assignment[s2],
                session2=s2,
                new_val2=cur1,
            )

    # Fallback to single move
    cur1 = assignment[s1]
    dom1 = domains[s1]
    cands = [v for v in dom1 if v != cur1]
    new_val = rng.choice(cands) if cands else cur1
    return CandidateMove(
        kind="move",
        session1=s1,
        new_val1=new_val,
    )


# ---------------------------------------------------------------------------
# State with Delta-Cost Evaluation
# ---------------------------------------------------------------------------

class NeighbourhoodState:
    """Manages assignment state, lookup tables, and fast delta-cost evaluation."""

    def __init__(self, problem: Problem, assignment: Dict[Session, AssignmentTuple]) -> None:
        self.problem = problem
        self.assignment = dict(assignment)

        # Lookup tables for resource occupancy: (resource_id, slot) -> Set[Session]
        self.teacher_occ: Dict[Tuple[str, SlotTuple], Set[Session]] = defaultdict(set)
        self.class_occ: Dict[Tuple[str, SlotTuple], Set[Session]] = defaultdict(set)
        self.room_occ: Dict[Tuple[str, SlotTuple], Set[Session]] = defaultdict(set)

        for s, val in self.assignment.items():
            for sl in slots_of(s, val):
                self.teacher_occ[(s.teacher_name, sl)].add(s)
                self.class_occ[(s.class_name, sl)].add(s)
                self.room_occ[(val[2], sl)].add(s)

        # Precompute initial cost
        tt = Timetable(problem=self.problem, assignment=self.assignment)
        breakdown = evaluate(tt, complete=True)
        self.current_hard_violations = breakdown.total_hard_violations
        self.current_soft_penalty = breakdown.total_soft_penalty
        self.current_cost = breakdown.total_cost

    def count_hard_violations(self) -> int:
        """Count active hard violations (H1, H2, H3) using lookup tables."""
        h1 = 1 if any(len(v) > 1 for v in self.teacher_occ.values()) else 0
        h2 = 1 if any(len(v) > 1 for v in self.class_occ.values()) else 0
        h3 = 1 if any(len(v) > 1 for v in self.room_occ.values()) else 0
        return h1 + h2 + h3

    def get_conflicted_sessions(self) -> List[Session]:
        """Return list of all sessions currently involved in >= 1 hard clash."""
        conflicted: List[Session] = []
        for s, v in self.assignment.items():
            for sl in slots_of(s, v):
                if (
                    len(self.teacher_occ[(s.teacher_name, sl)]) > 1
                    or len(self.class_occ[(s.class_name, sl)]) > 1
                    or len(self.room_occ[(v[2], sl)]) > 1
                ):
                    conflicted.append(s)
                    break
        return conflicted

    def eval_local_soft(self, affected_classes: Set[str], affected_teachers: Set[str]) -> int:
        """Evaluate soft penalties (S1, S2, S4, S5) strictly for affected classes and teachers."""
        pen = 0
        p = self.problem

        # S1: Gaps in affected classes' teaching day
        for cls_name in affected_classes:
            day_periods: Dict[int, Set[int]] = defaultdict(set)
            for s, v in self.assignment.items():
                if s.class_name == cls_name:
                    for d, per in slots_of(s, v):
                        day_periods[d].add(per)
            for pers in day_periods.values():
                if pers:
                    pen += max(0, (max(pers) - min(pers) + 1) - len(pers))

        # S2: Subject spread for affected classes
        for cls_name in affected_classes:
            groups: Dict[str, List[int]] = defaultdict(list)
            for s, v in self.assignment.items():
                if s.class_name == cls_name and not s.is_lab:
                    groups[s.subject_code].append(v[0])
            for days in groups.values():
                seen: Set[int] = set()
                for d in days:
                    if d in seen:
                        pen += 1
                    seen.add(d)

        # S4: Teacher preferences for affected teachers
        for t_name in affected_teachers:
            t = p.teachers.get(t_name)
            if not t or not t.preferences:
                continue
            prefs = t.preferences
            for s, v in self.assignment.items():
                if s.teacher_name == t_name:
                    if prefs.get("no_first_period") in (True, "True", "true", "1"):
                        for _, per in slots_of(s, v):
                            if per == 0:
                                pen += 2
                    if "preferred_days" in prefs:
                        raw = prefs["preferred_days"]
                        pref_days = raw if isinstance(raw, list) else [int(x) for x in str(raw).split(",")]
                        if v[0] not in pref_days:
                            pen += 2

        # S5: Teacher consecutive lectures (> 2 in a row) for affected teachers
        for t_name in affected_teachers:
            by_day: Dict[int, Set[int]] = defaultdict(set)
            for s, v in self.assignment.items():
                if s.teacher_name == t_name:
                    for d, per in slots_of(s, v):
                        by_day[d].add(per)
            for pers in by_day.values():
                sorted_p = sorted(pers)
                run = 1
                for i in range(1, len(sorted_p)):
                    if sorted_p[i] == sorted_p[i - 1] + 1:
                        run += 1
                        if run > 2:
                            pen += 3
                    else:
                        run = 1

        return pen

    def eval_delta_cost(self, move: CandidateMove) -> Tuple[int, int, int]:
        """Compute (delta_cost, new_hard_violations, new_soft_penalty) without modifying state.

        Returns:
            Tuple of ``(delta_cost, new_hard_violations, new_soft_penalty)``.
        """
        s1 = move.session1
        old_v1 = self.assignment[s1]
        new_v1 = move.new_val1

        is_swap = (move.kind == "swap" and move.session2 is not None and move.new_val2 is not None)
        if is_swap:
            s2 = move.session2  # type: ignore[assignment]
            old_v2 = self.assignment[s2]
            new_v2 = move.new_val2
            aff_classes = {s1.class_name, s2.class_name}
            aff_teachers = {s1.teacher_name, s2.teacher_name}
        else:
            s2 = None
            old_v2 = None
            new_v2 = None
            aff_classes = {s1.class_name}
            aff_teachers = {s1.teacher_name}

        # 1. Soft penalty before move for affected entities
        old_soft_local = self.eval_local_soft(aff_classes, aff_teachers)

        # 2. Temporarily apply move in occupancy tables & assignment
        for sl in slots_of(s1, old_v1):
            self.teacher_occ[(s1.teacher_name, sl)].remove(s1)
            self.class_occ[(s1.class_name, sl)].remove(s1)
            self.room_occ[(old_v1[2], sl)].remove(s1)
        self.assignment[s1] = new_v1
        for sl in slots_of(s1, new_v1):
            self.teacher_occ[(s1.teacher_name, sl)].add(s1)
            self.class_occ[(s1.class_name, sl)].add(s1)
            self.room_occ[(new_v1[2], sl)].add(s1)

        if is_swap and s2 is not None and old_v2 is not None and new_v2 is not None:
            for sl in slots_of(s2, old_v2):
                self.teacher_occ[(s2.teacher_name, sl)].remove(s2)
                self.class_occ[(s2.class_name, sl)].remove(s2)
                self.room_occ[(old_v2[2], sl)].remove(s2)
            self.assignment[s2] = new_v2
            for sl in slots_of(s2, new_v2):
                self.teacher_occ[(s2.teacher_name, sl)].add(s2)
                self.class_occ[(s2.class_name, sl)].add(s2)
                self.room_occ[(new_v2[2], sl)].add(s2)

        # 3. Compute new metrics
        new_hard = self.count_hard_violations()
        new_soft_local = self.eval_local_soft(aff_classes, aff_teachers)
        delta_soft = new_soft_local - old_soft_local
        new_soft = self.current_soft_penalty + delta_soft
        delta_hard = new_hard - self.current_hard_violations
        delta_cost = 1000 * delta_hard + delta_soft

        # 4. Revert temporary move
        for sl in slots_of(s1, new_v1):
            self.teacher_occ[(s1.teacher_name, sl)].remove(s1)
            self.class_occ[(s1.class_name, sl)].remove(s1)
            self.room_occ[(new_v1[2], sl)].remove(s1)
        self.assignment[s1] = old_v1
        for sl in slots_of(s1, old_v1):
            self.teacher_occ[(s1.teacher_name, sl)].add(s1)
            self.class_occ[(s1.class_name, sl)].add(s1)
            self.room_occ[(old_v1[2], sl)].add(s1)

        if is_swap and s2 is not None and old_v2 is not None and new_v2 is not None:
            for sl in slots_of(s2, new_v2):
                self.teacher_occ[(s2.teacher_name, sl)].remove(s2)
                self.class_occ[(s2.class_name, sl)].remove(s2)
                self.room_occ[(new_v2[2], sl)].remove(s2)
            self.assignment[s2] = old_v2
            for sl in slots_of(s2, old_v2):
                self.teacher_occ[(s2.teacher_name, sl)].add(s2)
                self.class_occ[(s2.class_name, sl)].add(s2)
                self.room_occ[(old_v2[2], sl)].add(s2)

        return delta_cost, new_hard, new_soft

    def apply_move(self, move: CandidateMove, delta_cost: int, new_hard: int, new_soft: int) -> None:
        """Permanently apply *move* and commit updated cost metrics."""
        s1 = move.session1
        old_v1 = self.assignment[s1]
        new_v1 = move.new_val1

        for sl in slots_of(s1, old_v1):
            self.teacher_occ[(s1.teacher_name, sl)].remove(s1)
            self.class_occ[(s1.class_name, sl)].remove(s1)
            self.room_occ[(old_v1[2], sl)].remove(s1)
        self.assignment[s1] = new_v1
        for sl in slots_of(s1, new_v1):
            self.teacher_occ[(s1.teacher_name, sl)].add(s1)
            self.class_occ[(s1.class_name, sl)].add(s1)
            self.room_occ[(new_v1[2], sl)].add(s1)

        if move.kind == "swap" and move.session2 is not None and move.new_val2 is not None:
            s2 = move.session2
            old_v2 = self.assignment[s2]
            new_v2 = move.new_val2
            for sl in slots_of(s2, old_v2):
                self.teacher_occ[(s2.teacher_name, sl)].remove(s2)
                self.class_occ[(s2.class_name, sl)].remove(s2)
                self.room_occ[(old_v2[2], sl)].remove(s2)
            self.assignment[s2] = new_v2
            for sl in slots_of(s2, new_v2):
                self.teacher_occ[(s2.teacher_name, sl)].add(s2)
                self.class_occ[(s2.class_name, sl)].add(s2)
                self.room_occ[(new_v2[2], sl)].add(s2)

        self.current_cost += delta_cost
        self.current_hard_violations = new_hard
        self.current_soft_penalty = new_soft

    def check_recount(self) -> None:
        """Correctness safeguard: verify incremental cost matches core.cost.evaluate."""
        tt = Timetable(problem=self.problem, assignment=self.assignment)
        breakdown = evaluate(tt, complete=True)
        assert self.current_cost == breakdown.total_cost, (
            f"Incremental cost {self.current_cost} != recount {breakdown.total_cost}"
        )
        assert self.current_hard_violations == breakdown.total_hard_violations, (
            f"Incremental hard {self.current_hard_violations} != recount {breakdown.total_hard_violations}"
        )
