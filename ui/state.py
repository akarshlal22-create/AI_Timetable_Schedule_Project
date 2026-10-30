"""ui/state.py — Centralized st.session_state access and management."""

from __future__ import annotations

from typing import List, Optional

import streamlit as st

from algorithms.base import SolverResult
from core.models import Problem, Session, Timetable
from services.editing import MoveHistory, apply_move

KEY_PROBLEM = "app_problem"
KEY_PROBLEM_SOURCE = "app_problem_source"
KEY_VALIDATION_ERRORS = "app_validation_errors"
KEY_SOLVER_RESULT = "app_solver_result"
KEY_MOVE_HISTORY = "app_move_history"


def init_session_state() -> None:
    """Initialize default keys in st.session_state if absent."""
    if KEY_PROBLEM not in st.session_state:
        st.session_state[KEY_PROBLEM] = None
    if KEY_PROBLEM_SOURCE not in st.session_state:
        st.session_state[KEY_PROBLEM_SOURCE] = ""
    if KEY_VALIDATION_ERRORS not in st.session_state:
        st.session_state[KEY_VALIDATION_ERRORS] = []
    if KEY_SOLVER_RESULT not in st.session_state:
        st.session_state[KEY_SOLVER_RESULT] = None
    if KEY_MOVE_HISTORY not in st.session_state:
        st.session_state[KEY_MOVE_HISTORY] = None


def get_problem() -> Optional[Problem]:
    """Return the currently loaded Problem."""
    return st.session_state.get(KEY_PROBLEM)


def set_problem(problem: Optional[Problem], source: str, errors: List[str]) -> None:
    """Set the active Problem and reset downstream timetables."""
    st.session_state[KEY_PROBLEM] = problem
    st.session_state[KEY_PROBLEM_SOURCE] = source
    st.session_state[KEY_VALIDATION_ERRORS] = errors
    # Loading new data clears previous timetable and move history
    st.session_state[KEY_SOLVER_RESULT] = None
    st.session_state[KEY_MOVE_HISTORY] = None


def get_validation_errors() -> List[str]:
    """Return current validation error strings."""
    return st.session_state.get(KEY_VALIDATION_ERRORS, [])


def get_solver_result() -> Optional[SolverResult]:
    """Return the active SolverResult."""
    return st.session_state.get(KEY_SOLVER_RESULT)


def set_solver_result(result: Optional[SolverResult]) -> None:
    """Save the SolverResult and initialize the MoveHistory."""
    st.session_state[KEY_SOLVER_RESULT] = result
    if result is not None and result.timetable is not None:
        st.session_state[KEY_MOVE_HISTORY] = MoveHistory(result.timetable)
    else:
        st.session_state[KEY_MOVE_HISTORY] = None


def get_current_timetable() -> Optional[Timetable]:
    """Return the current timetable (reflecting any manual edits) or None."""
    history: Optional[MoveHistory] = st.session_state.get(KEY_MOVE_HISTORY)
    if history is not None:
        return history.current
    res: Optional[SolverResult] = st.session_state.get(KEY_SOLVER_RESULT)
    if res is not None:
        return res.timetable
    return None


def get_move_history() -> Optional[MoveHistory]:
    """Return the active MoveHistory object."""
    return st.session_state.get(KEY_MOVE_HISTORY)


def apply_manual_edit(session: Session, new_day: int, new_period: int, new_room: str) -> List[str]:
    """Apply a move to the current timetable, updating history and returning conflict messages."""
    history = get_move_history()
    if history is None:
        return ["No active timetable to edit."]

    new_tt, cfls = apply_move(history.current, session, new_day, new_period, new_room)
    history.push(new_tt)
    return cfls


def undo_manual_edit() -> Optional[Timetable]:
    """Undo the last manual edit."""
    history = get_move_history()
    if history is not None:
        return history.undo()
    return None


def reset_manual_edits() -> Optional[Timetable]:
    """Reset back to the generated timetable."""
    history = get_move_history()
    if history is not None:
        return history.reset()
    return None
