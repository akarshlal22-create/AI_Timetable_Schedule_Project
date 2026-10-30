"""core/models.py — Dataclasses for every domain entity.

All entities use frozen=True where mutation is not needed so they are
hashable and safe to use as dict keys or set members.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Literal, Optional, Tuple


# ---------------------------------------------------------------------------
# Primitive type aliases
# ---------------------------------------------------------------------------
SlotTuple = Tuple[int, int]          # (day_index, period_index)
AssignmentTuple = Tuple[int, int, str]  # (day, period, room_name)


# ---------------------------------------------------------------------------
# Input entities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Subject:
    """A single course unit.

    Attributes:
        code:         Short unique identifier, e.g. 'CS201'.
        name:         Human-readable name, e.g. 'Data Structures'.
        weekly_hours: Number of lecture/lab sessions per week.
        type:         'theory' for regular lectures, 'lab' for practicals.
    """
    code: str
    name: str
    weekly_hours: int
    type: Literal["theory", "lab"]


@dataclass(frozen=True)
class Teacher:
    """A faculty member who can teach one or more subjects.

    Attributes:
        name:               Unique name used as identifier.
        teachable_subjects: Set of subject codes this teacher can handle.
        unavailable_slots:  Frozen set of (day, period) when teacher is busy.
        max_weekly_load:    Maximum sessions per week (hard cap).
        preferences:        Optional hint dict, e.g. {'no_first_period': True}.
    """
    name: str
    teachable_subjects: FrozenSet[str]
    unavailable_slots: FrozenSet[SlotTuple] = field(default_factory=frozenset)
    max_weekly_load: int = 20
    preferences: Dict[str, object] = field(default_factory=dict, compare=False, hash=False)

    # Make preferences hashable-friendly by excluding from hash (handled above)
    def __post_init__(self) -> None:
        # Ensure teachable_subjects is always a frozenset
        object.__setattr__(self, "teachable_subjects", frozenset(self.teachable_subjects))
        object.__setattr__(self, "unavailable_slots", frozenset(self.unavailable_slots))


@dataclass(frozen=True)
class ClassBatch:
    """A student group that follows a common timetable.

    Attributes:
        name:          Identifier, e.g. 'CS-A'.
        strength:      Number of students (used for room capacity checks).
        subject_codes: Ordered list of subject codes this batch must study.
        lunch_slot:    Period index reserved for lunch (nothing scheduled here).
        teacher_for:   Optional explicit mapping {subject_code: teacher_name}.
    """
    name: str
    strength: int
    subject_codes: Tuple[str, ...]
    lunch_slot: int
    teacher_for: Dict[str, str] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_codes", tuple(self.subject_codes))


@dataclass(frozen=True)
class Room:
    """A physical space where sessions are held.

    Attributes:
        name:     Unique identifier, e.g. 'LH-101'.
        capacity: Maximum student count.
        type:     'classroom' for theory, 'lab' for practicals.
    """
    name: str
    capacity: int
    type: Literal["classroom", "lab"]


@dataclass(frozen=True)
class TimeStructure:
    """Defines the weekly grid.

    Attributes:
        working_days:    Number of days per week (e.g. 5 for Mon–Fri).
        periods_per_day: Number of teaching periods each day.
        break_periods:   Frozenset of period indices that are always breaks
                         (applies globally, every day).
    """
    working_days: int
    periods_per_day: int
    break_periods: FrozenSet[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "break_periods", frozenset(self.break_periods))

    def all_slots(self) -> List[SlotTuple]:
        """Return every (day, period) pair that is NOT a break."""
        return [
            (d, p)
            for d in range(self.working_days)
            for p in range(self.periods_per_day)
            if p not in self.break_periods
        ]

    def teaching_periods(self) -> List[int]:
        """Return period indices that are not breaks."""
        return [p for p in range(self.periods_per_day) if p not in self.break_periods]


# ---------------------------------------------------------------------------
# Problem bundle
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """Top-level container for a scheduling problem instance.

    Attributes:
        name:          Human-readable label for this problem.
        time:          The weekly grid definition.
        subjects:      Dict mapping code -> Subject.
        teachers:      Dict mapping name -> Teacher.
        classes:       Dict mapping name -> ClassBatch.
        rooms:         Dict mapping name -> Room.
        seed:          Random seed for reproducibility (default 42).
    """
    name: str
    time: TimeStructure
    subjects: Dict[str, Subject]
    teachers: Dict[str, Teacher]
    classes: Dict[str, ClassBatch]
    rooms: Dict[str, Room]
    seed: int = 42


# ---------------------------------------------------------------------------
# CSP output entities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Session:
    """One CSP variable — a single scheduled meeting of a class+subject.

    A lab session occupies period p **and** p+1; this is recorded in
    ``is_lab`` so constraint checks can handle the double-period case.

    Attributes:
        class_name:     Name of the ClassBatch.
        subject_code:   Code of the Subject.
        session_index:  0-based index when a subject has multiple sessions.
        teacher_name:   Pre-assigned teacher (fixed before search).
        is_lab:         True when this session requires two consecutive periods.
    """
    class_name: str
    subject_code: str
    session_index: int
    teacher_name: str
    is_lab: bool = False

    def __repr__(self) -> str:
        kind = "Lab" if self.is_lab else "Lec"
        return (
            f"Session({self.class_name}|{self.subject_code}"
            f"|{kind}#{self.session_index}|T:{self.teacher_name})"
        )


@dataclass
class Timetable:
    """A (partial or complete) assignment of sessions to slots.

    The assignment maps each Session to ``(day, period, room_name)``.
    For lab sessions, ``period`` is the *first* of the two consecutive periods.

    Attributes:
        problem:    The Problem this timetable belongs to.
        assignment: Mapping from Session to (day, period, room_name).
    """
    problem: Problem
    assignment: Dict[Session, AssignmentTuple] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # View helpers
    # ------------------------------------------------------------------

    def sessions_for_class(self, class_name: str) -> Dict[Session, AssignmentTuple]:
        """Return all assigned sessions belonging to *class_name*."""
        return {s: v for s, v in self.assignment.items() if s.class_name == class_name}

    def sessions_for_teacher(self, teacher_name: str) -> Dict[Session, AssignmentTuple]:
        """Return all assigned sessions taught by *teacher_name*."""
        return {s: v for s, v in self.assignment.items() if s.teacher_name == teacher_name}

    def sessions_for_room(self, room_name: str) -> Dict[Session, AssignmentTuple]:
        """Return all assigned sessions in *room_name*."""
        return {s: v for s, v in self.assignment.items() if v[2] == room_name}

    def slots_occupied_by(self, session: Session) -> List[SlotTuple]:
        """Return the (day, period) pairs occupied by *session*.

        Labs occupy two consecutive periods; lectures occupy one.
        """
        if session not in self.assignment:
            return []
        day, period, _ = self.assignment[session]
        if session.is_lab:
            return [(day, period), (day, period + 1)]
        return [(day, period)]

    def is_complete(self) -> bool:
        """True when every session in the problem has been assigned."""
        from core.problem_builder import build_sessions  # lazy import to avoid cycles
        sessions = build_sessions(self.problem)
        return all(s in self.assignment for s in sessions)

    def copy(self) -> "Timetable":
        """Return a shallow copy with an independent assignment dict."""
        return Timetable(problem=self.problem, assignment=dict(self.assignment))
