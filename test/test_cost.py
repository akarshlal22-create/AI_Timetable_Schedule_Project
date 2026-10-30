"""tests/test_cost.py — Tests for core/cost.py.

Verifies that evaluate() returns zero violations for a hand-crafted valid
timetable, and that it correctly flags each type of hard violation.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.models import (
    ClassBatch, Problem, Room, Session, Subject, Teacher, TimeStructure, Timetable,
)
from core.cost import evaluate, conflicts, HARD_WEIGHT


# ---------------------------------------------------------------------------
# Minimal valid problem (same as in test_constraints)
# ---------------------------------------------------------------------------

def make_problem() -> Problem:
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
            unavailable_slots=frozenset({(4, 5)}),  # Friday last period
            max_weekly_load=10,
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
        "LH-1":  Room(name="LH-1",  capacity=40, type="classroom"),
        "LAB-1": Room(name="LAB-1", capacity=35, type="lab"),
    }
    return Problem(
        name="cost_test",
        time=time,
        subjects=subjects,
        teachers=teachers,
        classes=classes,
        rooms=rooms,
        seed=0,
    )


def s(cls, code, idx, teacher, is_lab=False):
    return Session(class_name=cls, subject_code=code, session_index=idx,
                   teacher_name=teacher, is_lab=is_lab)


# ---------------------------------------------------------------------------
# Perfect timetable — zero violations
# ---------------------------------------------------------------------------

class TestZeroCost:
    def test_valid_timetable_zero_cost(self):
        p = make_problem()
        # MATH: 3 sessions, PHY: 2 sessions, LAB: 1 session (lab occupies 2 periods)
        # All spread across different days, no clashes, correct rooms, no lunch/break overlap
        asgn = {
            s("CS-A", "MATH", 0, "Alice"):          (0, 1, "LH-1"),
            s("CS-A", "MATH", 1, "Alice"):          (1, 1, "LH-1"),
            s("CS-A", "MATH", 2, "Alice"):          (2, 1, "LH-1"),
            s("CS-A", "PHY",  0, "Alice"):          (3, 1, "LH-1"),
            s("CS-A", "PHY",  1, "Alice"):          (4, 1, "LH-1"),
            s("CS-A", "LAB",  0, "Bob", True):      (0, 4, "LAB-1"),  # periods 4 & 5
        }
        tt = Timetable(problem=p, assignment=asgn)
        result = evaluate(tt, complete=True)

        assert result.total_hard_violations == 0, result.summary()
        assert result.total_cost == result.total_soft_penalty, result.summary()
        # H7 must be satisfied (correct weekly hours for all subjects)
        assert result.hard["H7"] == 0

    def test_partial_timetable_no_h7_h8_check(self):
        p = make_problem()
        # Only 1 session assigned — incomplete, but complete=False should not flag H7
        asgn = {s("CS-A", "MATH", 0, "Alice"): (0, 1, "LH-1")}
        tt = Timetable(problem=p, assignment=asgn)
        result = evaluate(tt, complete=False)
        assert result.hard["H7"] == 0
        assert result.hard["H8"] == 0


# ---------------------------------------------------------------------------
# Each hard constraint fires when violated
# ---------------------------------------------------------------------------

class TestHardViolations:
    def test_h1_detected(self):
        p = make_problem()
        s1 = s("CS-A", "MATH", 0, "Alice")
        s2 = s("CS-A", "PHY",  0, "Alice")
        tt = Timetable(p, {s1: (0, 1, "LH-1"), s2: (0, 1, "LH-1")})  # same slot, same teacher
        result = evaluate(tt, complete=False)
        assert result.hard["H1"] == 1
        assert result.total_cost >= HARD_WEIGHT

    def test_h2_detected(self):
        p = make_problem()
        s1 = s("CS-A", "MATH", 0, "Alice")
        s2 = s("CS-A", "PHY",  0, "Alice")
        tt = Timetable(p, {s1: (0, 1, "LH-1"), s2: (0, 1, "LH-1")})
        result = evaluate(tt, complete=False)
        assert result.hard["H2"] == 1

    def test_h4_detected(self):
        import dataclasses
        p = make_problem()
        tiny = Room(name="TINY", capacity=5, type="classroom")
        rooms = dict(p.rooms)
        rooms["TINY"] = tiny
        p2 = dataclasses.replace(p, rooms=rooms)
        sess = s("CS-A", "MATH", 0, "Alice")
        tt = Timetable(p2, {sess: (0, 1, "TINY")})
        result = evaluate(tt, complete=False)
        assert result.hard["H4"] == 1

    def test_h5_detected(self):
        p = make_problem()
        sess = s("CS-A", "LAB", 0, "Bob", is_lab=True)
        tt = Timetable(p, {sess: (0, 1, "LH-1")})  # LH-1 is classroom, not lab
        result = evaluate(tt, complete=False)
        assert result.hard["H5"] == 1

    def test_h6_detected(self):
        p = make_problem()
        sess = s("CS-A", "MATH", 0, "Alice")
        tt = Timetable(p, {sess: (4, 5, "LH-1")})  # Alice unavailable on (4,5)
        result = evaluate(tt, complete=False)
        assert result.hard["H6"] == 1

    def test_h9_detected(self):
        p = make_problem()
        sess = s("CS-A", "MATH", 0, "Alice")
        tt = Timetable(p, {sess: (0, 3, "LH-1")})  # period 3 is lunch slot
        result = evaluate(tt, complete=False)
        assert result.hard["H9"] == 1


# ---------------------------------------------------------------------------
# Cost formula
# ---------------------------------------------------------------------------

class TestCostFormula:
    def test_cost_equals_hard_times_weight_plus_soft(self):
        p = make_problem()
        s1 = s("CS-A", "MATH", 0, "Alice")
        s2 = s("CS-A", "PHY",  0, "Alice")
        # Clash → H1 + H2 = 2 hard violations; may also have soft penalties
        tt = Timetable(p, {s1: (0, 1, "LH-1"), s2: (0, 1, "LH-1")})
        result = evaluate(tt, complete=False)
        expected = HARD_WEIGHT * result.total_hard_violations + result.total_soft_penalty
        assert result.total_cost == expected


# ---------------------------------------------------------------------------
# conflicts() helper
# ---------------------------------------------------------------------------

class TestConflicts:
    def test_no_conflicts_for_valid(self):
        p = make_problem()
        asgn = {
            s("CS-A", "MATH", 0, "Alice"): (0, 1, "LH-1"),
            s("CS-A", "PHY",  0, "Alice"): (1, 1, "LH-1"),
        }
        tt = Timetable(p, asgn)
        assert conflicts(tt) == []

    def test_detects_teacher_clash(self):
        p = make_problem()
        s1 = s("CS-A", "MATH", 0, "Alice")
        s2 = s("CS-A", "PHY",  0, "Alice")
        tt = Timetable(p, {s1: (0, 1, "LH-1"), s2: (0, 1, "LH-1")})
        result = conflicts(tt)
        reasons = [r for _, r in result]
        assert any("H1" in r for r in reasons)
