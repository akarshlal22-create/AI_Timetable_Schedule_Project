"""tests/test_editing.py — Tests for manual timetable editing, conflict preview, undo, and reset."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from algorithms.backtracking import BacktrackingConfig, BacktrackingSolver
from core.loader import load_problem_from_json
from core.models import ClassBatch, Problem, Room, Session, Timetable
from services.editing import MoveHistory, apply_move, preview_move


@pytest.fixture
def solved_fixture() -> tuple[Problem, Timetable]:
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "small.json")
    p = load_problem_from_json(data_path)
    solver = BacktrackingSolver(
        BacktrackingConfig(use_mrv=True, use_degree=True, use_forward_checking=True)
    )
    result = solver.solve(p, seed=42)
    assert result.timetable is not None
    return p, result.timetable


def test_move_into_free_valid_slot_creates_no_conflicts(solved_fixture):
    problem, timetable = solved_fixture

    # Find a theory lecture session
    lec_session = next(s for s in timetable.assignment if not s.is_lab)

    # Find a completely unoccupied slot across the entire timetable
    occupied_slots = set()
    for s, (d, per, rm) in timetable.assignment.items():
        occupied_slots.add((d, per))

    # Look for a valid free slot (not break, not teacher unavailable, not lunch)
    target_d = None
    target_p = None
    teacher = problem.teachers[lec_session.teacher_name]
    cls = problem.classes[lec_session.class_name]

    for d in range(problem.time.working_days):
        for p in range(problem.time.periods_per_day):
            if (d, p) not in occupied_slots:
                if p not in problem.time.break_periods and p != cls.lunch_slot:
                    if (d, p) not in teacher.unavailable_slots:
                        target_d = d
                        target_p = p
                        break
        if target_d is not None:
            break

    assert target_d is not None, "Could not find a free slot"

    # Classroom suitable for theory
    target_room = "LH-101"
    conflicts = preview_move(timetable, lec_session, target_d, target_p, target_room)
    assert conflicts == []


def test_move_onto_occupied_slots_flags_h1_h2_h3(solved_fixture):
    problem, timetable = solved_fixture

    sessions = list(timetable.assignment.keys())

    # Find two distinct sessions sharing the same teacher
    sess_t1 = None
    sess_t2 = None
    for i, s1 in enumerate(sessions):
        for s2 in sessions[i + 1:]:
            if s1.teacher_name == s2.teacher_name:
                sess_t1, sess_t2 = s1, s2
                break
        if sess_t1:
            break

    assert sess_t1 and sess_t2
    d2, p2, r2 = timetable.assignment[sess_t2]
    # Move sess_t1 to sess_t2's slot -> H1 teacher clash
    cfl_h1 = preview_move(timetable, sess_t1, d2, p2, "LH-102")
    assert any("H1" in c for c in cfl_h1)

    # Find two distinct sessions sharing the same class
    sess_c1 = None
    sess_c2 = None
    for i, s1 in enumerate(sessions):
        for s2 in sessions[i + 1:]:
            if s1.class_name == s2.class_name:
                sess_c1, sess_c2 = s1, s2
                break
        if sess_c1:
            break

    assert sess_c1 and sess_c2
    d2, p2, r2 = timetable.assignment[sess_c2]
    # Move sess_c1 to sess_c2's slot -> H2 class clash
    cfl_h2 = preview_move(timetable, sess_c1, d2, p2, "LH-102")
    assert any("H2" in c for c in cfl_h2)

    # Test H3 room clash: place session in an occupied room at the same slot
    target_sess = sessions[0]
    other_sess = sessions[1]
    od, op, oroom = timetable.assignment[other_sess]
    cfl_h3 = preview_move(timetable, target_sess, od, op, oroom)
    assert any("H3" in c for c in cfl_h3)


def test_too_small_room_flags_h4(solved_fixture):
    problem, timetable = solved_fixture

    # Add a small room (capacity 10) to the problem
    small_rooms = dict(problem.rooms)
    small_rooms["TINY-ROOM"] = Room(name="TINY-ROOM", capacity=10, type="classroom")
    prob_with_small_room = Problem(
        name=problem.name,
        time=problem.time,
        subjects=problem.subjects,
        teachers=problem.teachers,
        classes=problem.classes,
        rooms=small_rooms,
        seed=problem.seed,
    )
    tt = Timetable(problem=prob_with_small_room, assignment=dict(timetable.assignment))

    lec_sess = next(s for s in tt.assignment if not s.is_lab)
    cur_d, cur_p, _ = tt.assignment[lec_sess]

    # Move to TINY-ROOM
    cfls = preview_move(tt, lec_sess, cur_d, cur_p, "TINY-ROOM")
    assert any("H4" in c for c in cfls)


def test_lab_in_classroom_flags_h5(solved_fixture):
    problem, timetable = solved_fixture

    lab_sess = next(s for s in timetable.assignment if s.is_lab)
    cur_d, cur_p, _ = timetable.assignment[lab_sess]

    # Move lab to regular classroom LH-101
    cfls = preview_move(timetable, lab_sess, cur_d, cur_p, "LH-101")
    assert any("H5" in c for c in cfls)


def test_lab_overflow_past_end_of_day_flagged(solved_fixture):
    problem, timetable = solved_fixture
    lab_sess = next(s for s in timetable.assignment if s.is_lab)

    last_period = problem.time.periods_per_day - 1
    cfls = preview_move(timetable, lab_sess, 0, last_period, "CS-LAB")
    assert any("overflows" in c.lower() or "h3" in c.lower() for c in cfls)


def test_apply_move_immutability(solved_fixture):
    problem, timetable = solved_fixture

    orig_assignment_snapshot = dict(timetable.assignment)
    sess = next(iter(timetable.assignment.keys()))

    new_tt, _ = apply_move(timetable, sess, 0, 0, "LH-101")

    # Original timetable must not be modified
    assert timetable.assignment == orig_assignment_snapshot
    assert new_tt.assignment[sess] == (0, 0, "LH-101")
    assert timetable is not new_tt


def test_move_history_undo_and_reset(solved_fixture):
    problem, timetable = solved_fixture

    history = MoveHistory(timetable)
    assert not history.can_undo()
    assert history.current.assignment == timetable.assignment

    sess1 = list(timetable.assignment.keys())[0]
    sess2 = list(timetable.assignment.keys())[1]

    # First move
    tt1, _ = apply_move(history.current, sess1, 1, 1, "LH-101")
    history.push(tt1)
    assert history.can_undo()
    assert history.current.assignment[sess1] == (1, 1, "LH-101")

    # Second move
    tt2, _ = apply_move(history.current, sess2, 2, 2, "LH-102")
    history.push(tt2)
    assert history.current.assignment[sess2] == (2, 2, "LH-102")

    # Undo second move
    restored = history.undo()
    assert restored is not None
    assert history.current.assignment[sess2] == timetable.assignment[sess2]
    assert history.current.assignment[sess1] == (1, 1, "LH-101")
    assert history.can_undo()

    # Reset
    reset_tt = history.reset()
    assert reset_tt.assignment == timetable.assignment
    assert not history.can_undo()

