"""tests/test_app_smoke.py — App smoke test using streamlit.testing.v1.AppTest."""

from __future__ import annotations

import os
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def test_app_smoke_generation():
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    # Confirm no exceptions on startup
    assert not at.exception
    assert at.session_state["app_problem"] is not None
    assert at.session_state["app_problem"].name == "small"

    # Select Improved Backtracking
    for sb in at.selectbox:
        if sb.label == "Algorithm":
            sb.select("Backtracking (MRV + Degree + Forward Checking)").run()
            break

    # Click Generate Timetable
    for b in at.button:
        if b.label == "Generate Timetable":
            b.click().run()
            break

    # Check that generation succeeded without unhandled exceptions
    assert not at.exception
    res = at.session_state["app_solver_result"]
    assert res is not None
    assert res.success is True
    assert res.timetable is not None
    assert res.hard_violations == 0

