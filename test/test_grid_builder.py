"""tests/test_grid_builder.py — Tests for timetable grid views and styling."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd

from algorithms.backtracking import BacktrackingConfig, BacktrackingSolver
from core.loader import load_problem_from_json
from core.models import Problem, Timetable
from services.grid_builder import (
    build_grid_dataframe,
    get_day_labels,
    get_period_labels,
    style_timetable_grid,
)


@pytest.fixture
def solved_timetable() -> Timetable:
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "small.json")
    p = load_problem_from_json(data_path)
    solver = BacktrackingSolver(
        BacktrackingConfig(use_mrv=True, use_degree=True, use_forward_checking=True)
    )
    result = solver.solve(p, seed=42)
    assert result.timetable is not None
    return result.timetable


def test_labels_generation():
    day_labels = get_day_labels(5)
    assert day_labels == ["Mon", "Tue", "Wed", "Thu", "Fri"]
    period_labels = get_period_labels(6)
    assert period_labels == [f"Period {i}" for i in range(6)]


def test_class_view_grid_placement(solved_timetable: Timetable):
    cls_name = "CS-A"
    df, mask = build_grid_dataframe(solved_timetable, view_type="class", entity_name=cls_name)

    assert isinstance(df, pd.DataFrame)
    assert isinstance(mask, pd.DataFrame)
    assert df.shape == (6, 5)  # 6 periods x 5 days
    assert mask.shape == (6, 5)

    # Check lunch slot for CS-A (lunch_slot is 3)
    p = solved_timetable.problem
    lunch_slot = p.classes[cls_name].lunch_slot
    for day_col in df.columns:
        cell_val = str(df.loc[f"Period {lunch_slot}", day_col])
        assert "Lunch" in cell_val

    # Check that sessions assigned to CS-A appear in the grid
    class_sessions = solved_timetable.sessions_for_class(cls_name)
    assert len(class_sessions) > 0
    for sess, (d, per, rm) in class_sessions.items():
        day_col = df.columns[d]
        cell_val = str(df.loc[f"Period {per}", day_col])
        assert sess.subject_code in cell_val
        assert sess.teacher_name in cell_val
        assert rm in cell_val


def test_teacher_view_grid_placement(solved_timetable: Timetable):
    t_name = "Prof. Kumar"
    df, mask = build_grid_dataframe(solved_timetable, view_type="teacher", entity_name=t_name)

    assert df.shape == (6, 5)
    t_sessions = solved_timetable.sessions_for_teacher(t_name)
    assert len(t_sessions) > 0
    for sess, (d, per, rm) in t_sessions.items():
        day_col = df.columns[d]
        cell_val = str(df.loc[f"Period {per}", day_col])
        assert sess.subject_code in cell_val
        assert sess.class_name in cell_val
        assert rm in cell_val


def test_room_view_grid_placement(solved_timetable: Timetable):
    r_name = "CS-LAB"
    df, mask = build_grid_dataframe(solved_timetable, view_type="room", entity_name=r_name)

    assert df.shape == (6, 5)
    r_sessions = solved_timetable.sessions_for_room(r_name)
    assert len(r_sessions) > 0
    for sess, (d, per, _) in r_sessions.items():
        day_col = df.columns[d]
        cell_val = str(df.loc[f"Period {per}", day_col])
        assert sess.subject_code in cell_val
        assert sess.class_name in cell_val
        assert sess.teacher_name in cell_val


def test_lab_continuation_cells(solved_timetable: Timetable):
    # Find a lab session in the timetable
    lab_session = next(s for s in solved_timetable.assignment if s.is_lab)
    d, per, rm = solved_timetable.assignment[lab_session]

    df, _ = build_grid_dataframe(solved_timetable, view_type="class", entity_name=lab_session.class_name)
    day_col = df.columns[d]

    # Slot 1
    cell1 = str(df.loc[f"Period {per}", day_col])
    assert lab_session.subject_code in cell1
    assert "(cont.)" not in cell1

    # Slot 2 (continuation)
    cell2 = str(df.loc[f"Period {per + 1}", day_col])
    assert lab_session.subject_code in cell2
    assert "(cont.)" in cell2


def test_break_periods_labels(solved_timetable: Timetable):
    p = solved_timetable.problem
    for bp in p.time.break_periods:
        # Check in room view
        df, _ = build_grid_dataframe(solved_timetable, view_type="room", entity_name="LH-101")
        for day_col in df.columns:
            assert "Break" in str(df.loc[f"Period {bp}", day_col])


def test_grid_styler(solved_timetable: Timetable):
    df, mask = build_grid_dataframe(solved_timetable, view_type="class", entity_name="CS-A")
    styler = style_timetable_grid(df, mask)
    rendered = styler.to_html()
    assert isinstance(rendered, str)
    assert len(rendered) > 0

