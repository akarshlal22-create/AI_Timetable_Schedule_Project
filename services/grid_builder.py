"""services/grid_builder.py — Timetable to pandas DataFrame conversion for Class, Teacher, and Room views."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Set, Tuple

import pandas as pd
from pandas.io.formats.style import Styler

from core.cost import conflicts
from core.models import Session, Timetable

DAY_NAMES: List[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def get_day_labels(working_days: int) -> List[str]:
    """Return day column headers for *working_days*."""
    if working_days <= len(DAY_NAMES):
        return DAY_NAMES[:working_days]
    return [f"Day {d}" for d in range(working_days)]


def get_period_labels(periods_per_day: int) -> List[str]:
    """Return period row headers for *periods_per_day*."""
    return [f"Period {p}" for p in range(periods_per_day)]


def build_grid_dataframe(
    timetable: Timetable,
    view_type: Literal["class", "teacher", "room"],
    entity_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a (periods x days) DataFrame and matching conflict mask for *entity_name*.

    Args:
        timetable:   The current Timetable.
        view_type:   One of 'class', 'teacher', or 'room'.
        entity_name: Identifier for the class, teacher, or room.

    Returns:
        A tuple of ``(grid_df, conflict_mask_df)`` where conflict_mask_df is
        a boolean DataFrame of identical dimensions, True where a cell contains
        a hard-conflict session.
    """
    p = timetable.problem
    t = p.time
    days = get_day_labels(t.working_days)
    periods = get_period_labels(t.periods_per_day)

    # Initialize empty grid
    grid_data: List[List[List[str]]] = [
        [[] for _ in range(t.working_days)] for _ in range(t.periods_per_day)
    ]
    conflict_data: List[List[bool]] = [
        [False for _ in range(t.working_days)] for _ in range(t.periods_per_day)
    ]

    # Mark lunch and break periods
    cls_obj = p.classes.get(entity_name) if view_type == "class" else None
    for day_idx in range(t.working_days):
        for per_idx in range(t.periods_per_day):
            if cls_obj is not None and per_idx == cls_obj.lunch_slot:
                grid_data[per_idx][day_idx].append("Lunch")
            elif per_idx in t.break_periods:
                grid_data[per_idx][day_idx].append("Break")

    # Get set of all currently conflicted sessions from core/cost.py
    clashing_sessions: Set[Session] = {s for s, _ in conflicts(timetable)}

    # Place assigned sessions into cells
    for sess, (day, period, room) in timetable.assignment.items():
        # Filter by entity
        if view_type == "class" and sess.class_name != entity_name:
            continue
        if view_type == "teacher" and sess.teacher_name != entity_name:
            continue
        if view_type == "room" and room != entity_name:
            continue

        is_conflict = sess in clashing_sessions

        # Primary period cell label
        if view_type == "class":
            label = f"{sess.subject_code} ({sess.teacher_name}, {room})"
            cont_label = f"{sess.subject_code} (cont.) ({sess.teacher_name}, {room})"
        elif view_type == "teacher":
            label = f"{sess.subject_code} ({sess.class_name}, {room})"
            cont_label = f"{sess.subject_code} (cont.) ({sess.class_name}, {room})"
        else:  # room
            label = f"{sess.subject_code} ({sess.class_name}, {sess.teacher_name})"
            cont_label = f"{sess.subject_code} (cont.) ({sess.class_name}, {sess.teacher_name})"

        # Slot 1
        if 0 <= day < t.working_days and 0 <= period < t.periods_per_day:
            grid_data[period][day].append(label)
            if is_conflict:
                conflict_data[period][day] = True

        # Slot 2 for labs
        if sess.is_lab and (period + 1) < t.periods_per_day:
            if 0 <= day < t.working_days:
                grid_data[period + 1][day].append(cont_label)
                if is_conflict:
                    conflict_data[period + 1][day] = True

    # Flatten lists in cells to strings
    rendered_rows: List[List[str]] = []
    for per_idx in range(t.periods_per_day):
        row: List[str] = []
        for day_idx in range(t.working_days):
            cell_items = grid_data[per_idx][day_idx]
            if not cell_items:
                row.append("—")
            else:
                row.append(" / ".join(cell_items))
        rendered_rows.append(row)

    df = pd.DataFrame(rendered_rows, index=periods, columns=days)
    mask = pd.DataFrame(conflict_data, index=periods, columns=days)
    return df, mask


def style_timetable_grid(df: pd.DataFrame, mask: pd.DataFrame) -> Styler:
    """Apply styling to *df*: red highlighting for conflicts, muted text for breaks.

    Args:
        df:   Grid DataFrame.
        mask: Boolean DataFrame of identical shape (True where cell is clashing).

    Returns:
        A styled pandas Styler object ready for Streamlit rendering.
    """
    def apply_styles(data: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for r in range(len(data.index)):
            for c in range(len(data.columns)):
                if mask.iloc[r, c]:
                    styles.iloc[r, c] = (
                        "background-color: #FEE2E2; color: #991B1B; font-weight: 600;"
                    )
                elif data.iloc[r, c] in ("Break", "Lunch"):
                    styles.iloc[r, c] = "color: #94A3B8; font-style: italic;"
                elif data.iloc[r, c] == "—":
                    styles.iloc[r, c] = "color: #CBD5E1;"
        return styles

    return df.style.apply(apply_styles, axis=None)
