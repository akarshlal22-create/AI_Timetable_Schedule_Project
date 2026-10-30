"""core/cost.py — Violation counter and cost function for a Timetable.

cost = 1000 × (total hard violations) + (total soft penalties)

Works on partial and complete timetables.  Hard constraints H7 and H8 are
only checked on the *assigned* portion (under-assignment is not penalised
until the timetable is complete; see ``evaluate`` for the completeness flag).

Public API
----------
evaluate(timetable, complete=True)
    → CostBreakdown

conflicts(timetable)
    → list of (session, reason) pairs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from core.constraints import (
    check_h1_teacher_clash,
    check_h2_class_clash,
    check_h3_room_clash,
    check_h4_room_capacity,
    check_h5_room_type,
    check_h6_teacher_availability,
    check_h7_subject_hours,
    check_h8_teacher_load,
    check_h9_lunch_slot,
    penalty_s1_class_gaps,
    penalty_s2_subject_spread,
    penalty_s3_lab_consecutive,
    penalty_s4_teacher_preferences,
    penalty_s5_teacher_consecutive,
)
from core.models import Session, Timetable


HARD_WEIGHT = 1000  # penalty per hard-constraint violation


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CostBreakdown:
    """Full breakdown of constraint violations and their cost.

    Attributes:
        hard: Dict mapping constraint id ('H1'…'H9') to violation count (0 or 1).
        soft: Dict mapping constraint id ('S1'…'S5') to penalty points.
        total_hard_violations: Sum of hard violation counts.
        total_soft_penalty:    Sum of soft penalty points.
        total_cost:            ``HARD_WEIGHT × hard + soft``.
    """
    hard: Dict[str, int] = field(default_factory=dict)
    soft: Dict[str, int] = field(default_factory=dict)
    total_hard_violations: int = 0
    total_soft_penalty: int = 0
    total_cost: int = 0

    def is_feasible(self) -> bool:
        """True when there are zero hard violations."""
        return self.total_hard_violations == 0

    def summary(self) -> str:
        """One-line human-readable summary."""
        hard_str = ", ".join(f"{k}={v}" for k, v in self.hard.items() if v)
        soft_str = ", ".join(f"{k}={v}" for k, v in self.soft.items() if v)
        return (
            f"Cost={self.total_cost} "
            f"(hard={self.total_hard_violations}, soft={self.total_soft_penalty}) "
            f"| Hard: [{hard_str or 'none'}] "
            f"| Soft: [{soft_str or 'none'}]"
        )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def evaluate(timetable: Timetable, complete: bool = True) -> CostBreakdown:
    """Count all constraint violations and compute the total cost.

    Args:
        timetable: The (partial or complete) :class:`~core.models.Timetable`.
        complete:  When True, H7 (subject hours) and H8 (teacher load) are
                   checked against the full problem requirements.  Set to
                   False during search to avoid false positives on partial
                   assignments.

    Returns:
        A :class:`CostBreakdown` with per-constraint counts and totals.
    """
    hard: Dict[str, int] = {}

    # Hard constraints — each returns True (ok) / False (violated)
    hard["H1"] = 0 if check_h1_teacher_clash(timetable) else 1
    hard["H2"] = 0 if check_h2_class_clash(timetable) else 1
    hard["H3"] = 0 if check_h3_room_clash(timetable) else 1
    hard["H4"] = 0 if check_h4_room_capacity(timetable) else 1
    hard["H5"] = 0 if check_h5_room_type(timetable) else 1
    hard["H6"] = 0 if check_h6_teacher_availability(timetable) else 1
    hard["H7"] = (0 if check_h7_subject_hours(timetable) else 1) if complete else 0
    hard["H8"] = (0 if check_h8_teacher_load(timetable) else 1) if complete else 0
    hard["H9"] = 0 if check_h9_lunch_slot(timetable) else 1

    soft: Dict[str, int] = {
        "S1": penalty_s1_class_gaps(timetable),
        "S2": penalty_s2_subject_spread(timetable),
        "S3": penalty_s3_lab_consecutive(timetable),
        "S4": penalty_s4_teacher_preferences(timetable),
        "S5": penalty_s5_teacher_consecutive(timetable),
    }

    total_hard = sum(hard.values())
    total_soft = sum(soft.values())
    total_cost = HARD_WEIGHT * total_hard + total_soft

    return CostBreakdown(
        hard=hard,
        soft=soft,
        total_hard_violations=total_hard,
        total_soft_penalty=total_soft,
        total_cost=total_cost,
    )


def conflicts(timetable: Timetable) -> List[Tuple[Session, str]]:
    """Return a list of ``(session, reason)`` for every conflicting session.

    Useful for the manual-edit warning in the UI and for Min-Conflicts.
    Only checks constraints that involve pairs of sessions (H1, H2, H3,
    H4, H5, H6, H9); completeness constraints H7/H8 are reported globally.

    Args:
        timetable: Any (partial or complete) :class:`~core.models.Timetable`.

    Returns:
        List of ``(Session, str)`` tuples.  A session may appear multiple
        times if it participates in multiple conflicts.
    """
    p = timetable.problem
    result: List[Tuple[Session, str]] = []

    # Build a slot → session list for clash detection
    from collections import defaultdict

    slot_to_sessions_by_teacher: Dict = defaultdict(list)
    slot_to_sessions_by_class: Dict = defaultdict(list)
    slot_to_sessions_by_room: Dict = defaultdict(list)

    for sess, (day, period, room) in timetable.assignment.items():
        for slot in timetable.slots_occupied_by(sess):
            slot_to_sessions_by_teacher[(sess.teacher_name, slot)].append(sess)
            slot_to_sessions_by_class[(sess.class_name, slot)].append(sess)
            slot_to_sessions_by_room[(room, slot)].append(sess)

    seen: set = set()

    # H1 teacher clash
    for (teacher, slot), sessions in slot_to_sessions_by_teacher.items():
        if len(sessions) > 1:
            for s in sessions:
                key = (s, "H1")
                if key not in seen:
                    result.append((s, f"H1: Teacher '{teacher}' clash at {slot}"))
                    seen.add(key)

    # H2 class clash
    for (cls, slot), sessions in slot_to_sessions_by_class.items():
        if len(sessions) > 1:
            for s in sessions:
                key = (s, "H2")
                if key not in seen:
                    result.append((s, f"H2: Class '{cls}' clash at {slot}"))
                    seen.add(key)

    # H3 room clash
    for (room, slot), sessions in slot_to_sessions_by_room.items():
        if len(sessions) > 1:
            for s in sessions:
                key = (s, "H3")
                if key not in seen:
                    result.append((s, f"H3: Room '{room}' clash at {slot}"))
                    seen.add(key)

    # H4 room capacity
    for sess, (day, period, room_name) in timetable.assignment.items():
        cls = p.classes[sess.class_name]
        room = p.rooms[room_name]
        if room.capacity < cls.strength:
            result.append((
                sess,
                f"H4: Room '{room_name}' capacity {room.capacity} < "
                f"class '{cls.name}' strength {cls.strength}",
            ))

    # H5 room type mismatch
    for sess, (day, period, room_name) in timetable.assignment.items():
        subj = p.subjects[sess.subject_code]
        room = p.rooms[room_name]
        expected = "lab" if subj.type == "lab" else "classroom"
        if room.type != expected:
            result.append((
                sess,
                f"H5: Subject '{sess.subject_code}' needs {expected} "
                f"but assigned room '{room_name}' is {room.type}",
            ))

    # H6 teacher unavailability
    for sess, (day, period, _) in timetable.assignment.items():
        teacher = p.teachers[sess.teacher_name]
        for slot in timetable.slots_occupied_by(sess):
            if slot in teacher.unavailable_slots:
                result.append((
                    sess,
                    f"H6: Teacher '{sess.teacher_name}' unavailable at {slot}",
                ))

    # H9 lunch slot
    for sess in timetable.assignment:
        cls = p.classes[sess.class_name]
        for (day, period) in timetable.slots_occupied_by(sess):
            if period == cls.lunch_slot:
                result.append((
                    sess,
                    f"H9: Class '{cls.name}' lunch at period {cls.lunch_slot} violated",
                ))

    return result
