"""tests/test_constraints.py — Unit tests for each hard and soft constraint.

Uses a minimal hand-built problem (1 class, 2 teachers) to isolate each
constraint function and verify it fires exactly when it should.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.models import (
    ClassBatch, Problem, Room, Session, Subject, Teacher, TimeStructure, Timetable,
)
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_problem() -> Problem:
    """1 class, 2 teachers, 2 theory + 1 lab subject, 2 rooms, 5×6 grid."""
    time = TimeStructure(working_days=5, periods_per_day=6, break_periods=frozenset({3}))
    subjects = {
        "MATH": Subject(code="MATH", name="Mathematics", weekly_hours=3, type="theory"),
        "PHY":  Subject(code="PHY",  name="Physics",     weekly_hours=2, type="theory"),
        "LAB":  Subject(code="LAB",  name="Lab Work",    weekly_hours=1, type="lab"),
    }
    teachers = {
        "Alice": Teacher(
            name="Alice",
            teachable_subjects=frozenset({"MATH", "PHY"}),
            unavailable_slots=frozenset({(0, 0)}),  # Monday period 0
            max_weekly_load=10,
            preferences={"no_first_period": True},
        ),
        "Bob": Teacher(
            name="Bob",
            teachable_subjects=frozenset({"LAB"}),
            unavailable_slots=frozenset(),
            max_weekly_load=5,
        ),
    }
    classes = {
        "CS-A": ClassBatch(
            name="CS-A",
            strength=30,
            subject_codes=("MATH", "PHY", "LAB"),
            lunch_slot=3,
        )
    }
    rooms = {
        "LH-1": Room(name="LH-1", capacity=40, type="classroom"),
        "LAB-1": Room(name="LAB-1", capacity=35, type="lab"),
    }
    return Problem(
        name="minimal",
        time=time,
        subjects=subjects,
        teachers=teachers,
        classes=classes,
        rooms=rooms,
        seed=0,
    )


def make_session(class_name: str, subject_code: str, idx: int, teacher: str, is_lab: bool = False) -> Session:
    return Session(class_name=class_name, subject_code=subject_code,
                   session_index=idx, teacher_name=teacher, is_lab=is_lab)


def make_timetable(problem: Problem, assignments: dict) -> Timetable:
    return Timetable(problem=problem, assignment=assignments)


# ---------------------------------------------------------------------------
# H1 – Teacher clash
# ---------------------------------------------------------------------------

class TestH1TeacherClash:
    def test_no_clash(self, minimal_problem):
        s1 = make_session("CS-A", "MATH", 0, "Alice")
        s2 = make_session("CS-A", "PHY",  0, "Alice")
        tt = make_timetable(minimal_problem, {s1: (0, 1, "LH-1"), s2: (0, 2, "LH-1")})
        assert check_h1_teacher_clash(tt) is True

    def test_clash(self, minimal_problem):
        s1 = make_session("CS-A", "MATH", 0, "Alice")
        s2 = make_session("CS-A", "PHY",  0, "Alice")
        # Both at (day=0, period=1)
        tt = make_timetable(minimal_problem, {s1: (0, 1, "LH-1"), s2: (0, 1, "LH-1")})
        assert check_h1_teacher_clash(tt) is False


# ---------------------------------------------------------------------------
# H2 – Class clash
# ---------------------------------------------------------------------------

class TestH2ClassClash:
    def test_no_clash(self, minimal_problem):
        s1 = make_session("CS-A", "MATH", 0, "Alice")
        s2 = make_session("CS-A", "PHY",  0, "Alice")
        tt = make_timetable(minimal_problem, {s1: (0, 1, "LH-1"), s2: (1, 1, "LH-1")})
        assert check_h2_class_clash(tt) is True

    def test_clash(self, minimal_problem):
        s1 = make_session("CS-A", "MATH", 0, "Alice")
        s2 = make_session("CS-A", "PHY",  0, "Alice")
        tt = make_timetable(minimal_problem, {s1: (0, 1, "LH-1"), s2: (0, 1, "LH-1")})
        assert check_h2_class_clash(tt) is False


# ---------------------------------------------------------------------------
# H3 – Room clash
# ---------------------------------------------------------------------------

class TestH3RoomClash:
    def test_no_clash(self, minimal_problem):
        s1 = make_session("CS-A", "MATH", 0, "Alice")
        s2 = make_session("CS-A", "PHY",  0, "Alice")
        tt = make_timetable(minimal_problem, {s1: (0, 1, "LH-1"), s2: (0, 2, "LH-1")})
        assert check_h3_room_clash(tt) is True

    def test_clash(self, minimal_problem):
        s1 = make_session("CS-A", "MATH", 0, "Alice")
        s2 = make_session("CS-A", "PHY",  0, "Alice")
        tt = make_timetable(minimal_problem, {s1: (0, 1, "LH-1"), s2: (0, 1, "LH-1")})
        assert check_h3_room_clash(tt) is False


# ---------------------------------------------------------------------------
# H4 – Room capacity
# ---------------------------------------------------------------------------

class TestH4RoomCapacity:
    def test_ok(self, minimal_problem):
        s = make_session("CS-A", "MATH", 0, "Alice")  # strength 30, LH-1 cap 40
        tt = make_timetable(minimal_problem, {s: (0, 1, "LH-1")})
        assert check_h4_room_capacity(tt) is True

    def test_overflow(self, minimal_problem):
        # Add a tiny room
        import dataclasses
        tiny = Room(name="TINY", capacity=10, type="classroom")
        rooms = dict(minimal_problem.rooms)
        rooms["TINY"] = tiny
        p = dataclasses.replace(minimal_problem, rooms=rooms)
        s = make_session("CS-A", "MATH", 0, "Alice")  # strength 30 > 10
        tt = make_timetable(p, {s: (0, 1, "TINY")})
        assert check_h4_room_capacity(tt) is False


# ---------------------------------------------------------------------------
# H5 – Room type
# ---------------------------------------------------------------------------

class TestH5RoomType:
    def test_theory_in_classroom(self, minimal_problem):
        s = make_session("CS-A", "MATH", 0, "Alice")
        tt = make_timetable(minimal_problem, {s: (0, 1, "LH-1")})
        assert check_h5_room_type(tt) is True

    def test_lab_in_classroom(self, minimal_problem):
        s = make_session("CS-A", "LAB", 0, "Bob", is_lab=True)
        tt = make_timetable(minimal_problem, {s: (0, 1, "LH-1")})   # LH-1 is classroom
        assert check_h5_room_type(tt) is False

    def test_lab_in_lab_room(self, minimal_problem):
        s = make_session("CS-A", "LAB", 0, "Bob", is_lab=True)
        tt = make_timetable(minimal_problem, {s: (0, 1, "LAB-1")})
        assert check_h5_room_type(tt) is True


# ---------------------------------------------------------------------------
# H6 – Teacher availability
# ---------------------------------------------------------------------------

class TestH6TeacherAvailability:
    def test_available(self, minimal_problem):
        s = make_session("CS-A", "MATH", 0, "Alice")  # Alice unavailable (0,0)
        tt = make_timetable(minimal_problem, {s: (0, 1, "LH-1")})   # (0,1) is fine
        assert check_h6_teacher_availability(tt) is True

    def test_unavailable(self, minimal_problem):
        s = make_session("CS-A", "MATH", 0, "Alice")
        tt = make_timetable(minimal_problem, {s: (0, 0, "LH-1")})   # (0,0) is unavailable
        assert check_h6_teacher_availability(tt) is False


# ---------------------------------------------------------------------------
# H7 – Subject hours
# ---------------------------------------------------------------------------

class TestH7SubjectHours:
    def test_correct(self, minimal_problem):
        # MATH needs 3 sessions, PHY 2, LAB 1
        s1 = make_session("CS-A", "MATH", 0, "Alice")
        s2 = make_session("CS-A", "MATH", 1, "Alice")
        s3 = make_session("CS-A", "MATH", 2, "Alice")
        s4 = make_session("CS-A", "PHY",  0, "Alice")
        s5 = make_session("CS-A", "PHY",  1, "Alice")
        s6 = make_session("CS-A", "LAB",  0, "Bob", is_lab=True)
        asgn = {
            s1: (0, 1, "LH-1"), s2: (1, 1, "LH-1"), s3: (2, 1, "LH-1"),
            s4: (3, 1, "LH-1"), s5: (4, 1, "LH-1"),
            s6: (0, 1, "LAB-1"),
        }
        tt = make_timetable(minimal_problem, asgn)
        assert check_h7_subject_hours(tt) is True

    def test_missing_session(self, minimal_problem):
        # Only 2 of 3 required MATH sessions
        s1 = make_session("CS-A", "MATH", 0, "Alice")
        s2 = make_session("CS-A", "MATH", 1, "Alice")
        tt = make_timetable(minimal_problem, {s1: (0, 1, "LH-1"), s2: (1, 1, "LH-1")})
        assert check_h7_subject_hours(tt) is False


# ---------------------------------------------------------------------------
# H8 – Teacher load
# ---------------------------------------------------------------------------

class TestH8TeacherLoad:
    def test_within_limit(self, minimal_problem):
        sessions = [make_session("CS-A", "MATH", i, "Alice") for i in range(3)]
        asgn = {s: (i, 1, "LH-1") for i, s in enumerate(sessions)}
        tt = make_timetable(minimal_problem, asgn)
        assert check_h8_teacher_load(tt) is True  # Alice max=10

    def test_over_limit(self, minimal_problem):
        # Bob's max is 5; assign 6 sessions
        sessions = [make_session("CS-A", "LAB", i, "Bob", is_lab=True) for i in range(6)]
        asgn = {s: (i % 5, 1, "LAB-1") for i, s in enumerate(sessions)}
        tt = make_timetable(minimal_problem, asgn)
        assert check_h8_teacher_load(tt) is False


# ---------------------------------------------------------------------------
# H9 – Lunch slot
# ---------------------------------------------------------------------------

class TestH9LunchSlot:
    def test_not_in_lunch(self, minimal_problem):
        s = make_session("CS-A", "MATH", 0, "Alice")
        tt = make_timetable(minimal_problem, {s: (0, 1, "LH-1")})  # lunch=3
        assert check_h9_lunch_slot(tt) is True

    def test_in_lunch(self, minimal_problem):
        s = make_session("CS-A", "MATH", 0, "Alice")
        tt = make_timetable(minimal_problem, {s: (0, 3, "LH-1")})  # period 3 = lunch
        assert check_h9_lunch_slot(tt) is False


# ---------------------------------------------------------------------------
# S1 – Class gaps
# ---------------------------------------------------------------------------

class TestS1ClassGaps:
    def test_no_gap(self, minimal_problem):
        s1 = make_session("CS-A", "MATH", 0, "Alice")
        s2 = make_session("CS-A", "PHY",  0, "Alice")
        # Periods 1 and 2 — contiguous, no gap
        tt = make_timetable(minimal_problem, {s1: (0, 1, "LH-1"), s2: (0, 2, "LH-1")})
        assert penalty_s1_class_gaps(tt) == 0

    def test_gap(self, minimal_problem):
        s1 = make_session("CS-A", "MATH", 0, "Alice")
        s2 = make_session("CS-A", "PHY",  0, "Alice")
        # Periods 1 and 4 — gap of 2 (periods 2 and 3 are idle between 1 and 4)
        tt = make_timetable(minimal_problem, {s1: (0, 1, "LH-1"), s2: (0, 4, "LH-1")})
        assert penalty_s1_class_gaps(tt) == 2


# ---------------------------------------------------------------------------
# S2 – Subject spread
# ---------------------------------------------------------------------------

class TestS2SubjectSpread:
    def test_spread(self, minimal_problem):
        s1 = make_session("CS-A", "MATH", 0, "Alice")
        s2 = make_session("CS-A", "MATH", 1, "Alice")
        # Different days → no penalty
        tt = make_timetable(minimal_problem, {s1: (0, 1, "LH-1"), s2: (1, 1, "LH-1")})
        assert penalty_s2_subject_spread(tt) == 0

    def test_same_day(self, minimal_problem):
        s1 = make_session("CS-A", "MATH", 0, "Alice")
        s2 = make_session("CS-A", "MATH", 1, "Alice")
        # Same day → 1 penalty
        tt = make_timetable(minimal_problem, {s1: (0, 1, "LH-1"), s2: (0, 2, "LH-1")})
        assert penalty_s2_subject_spread(tt) == 1

    def test_lab_exempt(self, minimal_problem):
        s = make_session("CS-A", "LAB", 0, "Bob", is_lab=True)
        # Lab on same day twice would be exempt anyway; single lab = 0
        tt = make_timetable(minimal_problem, {s: (0, 1, "LAB-1")})
        assert penalty_s2_subject_spread(tt) == 0


# ---------------------------------------------------------------------------
# S3 – Lab consecutive
# ---------------------------------------------------------------------------

class TestS3LabConsecutive:
    def test_consecutive(self, minimal_problem):
        s = make_session("CS-A", "LAB", 0, "Bob", is_lab=True)
        # is_lab=True → slots_occupied_by returns [(day,p),(day,p+1)]
        tt = make_timetable(minimal_problem, {s: (0, 1, "LAB-1")})
        assert penalty_s3_lab_consecutive(tt) == 0

    def test_theory_ignored(self, minimal_problem):
        s = make_session("CS-A", "MATH", 0, "Alice")
        tt = make_timetable(minimal_problem, {s: (0, 1, "LH-1")})
        assert penalty_s3_lab_consecutive(tt) == 0


# ---------------------------------------------------------------------------
# S4 – Teacher preferences
# ---------------------------------------------------------------------------

class TestS4TeacherPreferences:
    def test_no_preference_violation(self, minimal_problem):
        s = make_session("CS-A", "MATH", 0, "Alice")  # Alice: no_first_period
        tt = make_timetable(minimal_problem, {s: (0, 1, "LH-1")})  # period 1 is fine
        assert penalty_s4_teacher_preferences(tt) == 0

    def test_first_period_violation(self, minimal_problem):
        s = make_session("CS-A", "MATH", 0, "Alice")
        tt = make_timetable(minimal_problem, {s: (0, 0, "LH-1")})  # period 0 violates
        assert penalty_s4_teacher_preferences(tt) > 0


# ---------------------------------------------------------------------------
# S5 – Teacher consecutive lectures
# ---------------------------------------------------------------------------

class TestS5TeacherConsecutive:
    def test_two_consecutive_ok(self, minimal_problem):
        s1 = make_session("CS-A", "MATH", 0, "Alice")
        s2 = make_session("CS-A", "PHY",  0, "Alice")
        # Periods 1 and 2 — run of 2, which is allowed
        tt = make_timetable(minimal_problem, {s1: (0, 1, "LH-1"), s2: (0, 2, "LH-1")})
        assert penalty_s5_teacher_consecutive(tt) == 0

    def test_three_consecutive_penalty(self, minimal_problem):
        s1 = make_session("CS-A", "MATH", 0, "Alice")
        s2 = make_session("CS-A", "MATH", 1, "Alice")
        s3 = make_session("CS-A", "PHY",  0, "Alice")
        # Periods 1, 2, 4 → run of 2 at 1-2 (ok), isolated at 4 → no penalty
        tt = make_timetable(minimal_problem, {
            s1: (0, 1, "LH-1"),
            s2: (0, 2, "LH-1"),
            s3: (0, 4, "LH-1"),
        })
        assert penalty_s5_teacher_consecutive(tt) == 0

    def test_long_run_penalised(self, minimal_problem):
        # Force a run of 3 by adding a 3rd consecutive session
        s1 = make_session("CS-A", "MATH", 0, "Alice")
        s2 = make_session("CS-A", "MATH", 1, "Alice")
        s3 = make_session("CS-A", "PHY",  0, "Alice")
        # Periods 1, 2, 4 → run of 2 ok; need 1,2,4 doesn't trigger
        # Use periods 0,1,2 — run of 3 → 1 extra period beyond 2 = 3 points
        tt = make_timetable(minimal_problem, {
            s1: (0, 0, "LH-1"),
            s2: (0, 1, "LH-1"),
            s3: (0, 2, "LH-1"),
        })
        assert penalty_s5_teacher_consecutive(tt) == 3
