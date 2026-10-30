"""services/editing.py — Pure functions for manual timetable editing, previewing conflicts, undo, and reset."""

from __future__ import annotations

from typing import List, Optional, Tuple

from core.cost import conflicts
from core.models import AssignmentTuple, Session, Timetable


def preview_move(
    timetable: Timetable,
    session: Session,
    new_day: int,
    new_period: int,
    new_room: str,
) -> List[str]:
    """Compute the list of hard-constraint conflicts that moving *session* to
    ``(new_day, new_period, new_room)`` would produce, without altering *timetable*.

    Args:
        timetable:  Current Timetable (unmutated).
        session:    The Session to be moved.
        new_day:    Target day index.
        new_period: Target period index.
        new_room:   Target room name.

    Returns:
        List of human-readable conflict explanation strings (e.g. "H1: ...").
        Returns an empty list if there are no hard conflicts.
    """
    p = timetable.problem
    reasons: List[str] = []

    # 1. Day / Period boundary checks
    if new_day < 0 or new_day >= p.time.working_days:
        reasons.append(f"Day index {new_day} is out of bounds [0, {p.time.working_days - 1}]")

    if new_period < 0 or new_period >= p.time.periods_per_day:
        reasons.append(f"Period index {new_period} is out of bounds [0, {p.time.periods_per_day - 1}]")

    # 2. Lab consecutive period overflow check
    if session.is_lab and (new_period + 1) >= p.time.periods_per_day:
        reasons.append(
            f"H3 / Lab Duration: Lab requires 2 consecutive periods and overflows past the end of day "
            f"(period {new_period + 1} exceeds max period {p.time.periods_per_day - 1})"
        )

    # 3. Global break periods check
    if new_period in p.time.break_periods:
        reasons.append(f"H9 / Break: Period {new_period} is a global scheduled break")
    if session.is_lab and (new_period + 1) in p.time.break_periods:
        reasons.append(f"H9 / Break: Lab second period ({new_period + 1}) overlaps a global scheduled break")

    # 4. Check core/cost.py conflicts on simulated timetable
    temp_assignment = dict(timetable.assignment)
    temp_assignment[session] = (new_day, new_period, new_room)
    temp_tt = Timetable(problem=p, assignment=temp_assignment)

    all_conflicts = conflicts(temp_tt)
    # Collect conflicts specifically associated with the moved session
    for s, reason in all_conflicts:
        if s == session:
            if reason not in reasons:
                reasons.append(reason)

    return reasons


def apply_move(
    timetable: Timetable,
    session: Session,
    new_day: int,
    new_period: int,
    new_room: str,
) -> Tuple[Timetable, List[str]]:
    """Return a new Timetable with *session* placed at ``(new_day, new_period, new_room)``,
    along with any conflicts the new timetable possesses. Does NOT mutate the input *timetable*.

    Args:
        timetable:  The original Timetable.
        session:    The Session to move.
        new_day:    Target day index.
        new_period: Target period index.
        new_room:   Target room name.

    Returns:
        Tuple of ``(new_timetable, conflict_messages_for_session)``.
    """
    conflicts_preview = preview_move(timetable, session, new_day, new_period, new_room)

    new_assignment = dict(timetable.assignment)
    new_assignment[session] = (new_day, new_period, new_room)
    new_tt = Timetable(problem=timetable.problem, assignment=new_assignment)

    return new_tt, conflicts_preview


class MoveHistory:
    """Manages an undo stack and reset state for manual timetable modifications."""

    def __init__(self, original_timetable: Timetable) -> None:
        self.original: Timetable = original_timetable.copy()
        self.stack: List[Timetable] = [self.original.copy()]

    @property
    def current(self) -> Timetable:
        """The currently active timetable."""
        return self.stack[-1]

    def push(self, timetable: Timetable) -> None:
        """Push a newly modified timetable onto the history stack."""
        self.stack.append(timetable.copy())

    def undo(self) -> Optional[Timetable]:
        """Revert the last move if possible and return the restored timetable."""
        if len(self.stack) > 1:
            self.stack.pop()
            return self.stack[-1]
        return None

    def reset(self) -> Timetable:
        """Reset back to the original generated timetable."""
        self.stack = [self.original.copy()]
        return self.stack[-1]

    def can_undo(self) -> bool:
        """True if there is at least one move that can be undone."""
        return len(self.stack) > 1
