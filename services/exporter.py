"""services/exporter.py — Export timetable grids and flat session records to CSV and Excel."""

from __future__ import annotations

import io
from typing import List

import pandas as pd

from core.models import Timetable
from services.grid_builder import build_grid_dataframe, get_day_labels


def export_grid_csv(grid_df: pd.DataFrame) -> str:
    """Return the CSV string representation of the currently displayed grid."""
    return grid_df.to_csv(index=True)


def build_flat_sessions_dataframe(timetable: Timetable) -> pd.DataFrame:
    """Build a flat tabular DataFrame of all scheduled sessions.

    Columns:
        Class, Subject, Teacher, Room, Day, Period, Is_Lab
    """
    p = timetable.problem
    day_labels = get_day_labels(p.time.working_days)

    rows = []
    for sess, (day, period, room) in timetable.assignment.items():
        day_str = day_labels[day] if 0 <= day < len(day_labels) else f"Day {day}"
        rows.append({
            "Class": sess.class_name,
            "Subject": sess.subject_code,
            "Teacher": sess.teacher_name,
            "Room": room,
            "Day": day_str,
            "Day_Index": day,
            "Period": period,
            "Is_Lab": sess.is_lab,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["Class", "Day_Index", "Period"]).drop(columns=["Day_Index"])
    else:
        df = pd.DataFrame(columns=["Class", "Subject", "Teacher", "Room", "Day", "Period", "Is_Lab"])
    return df


def export_flat_csv(timetable: Timetable) -> str:
    """Return the CSV string of all sessions scheduled in *timetable*."""
    df = build_flat_sessions_dataframe(timetable)
    return df.to_csv(index=False)


def export_timetable_excel(timetable: Timetable) -> bytes:
    """Generate a multi-sheet Excel (.xlsx) workbook in-memory.

    Contains:
      - One sheet per ClassBatch
      - One sheet per Teacher
      - One sheet per Room
      - An 'All_Sessions' sheet with flat tabular records.

    Returns:
        Raw bytes of the .xlsx file.
    """
    p = timetable.problem
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # 1. Sheets per Class
        for cls_name in sorted(p.classes.keys()):
            grid_df, _ = build_grid_dataframe(timetable, view_type="class", entity_name=cls_name)
            # Excel sheet names maximum length is 31 characters
            sheet_title = f"Class_{cls_name}"[:31]
            grid_df.to_excel(writer, sheet_name=sheet_title)

        # 2. Sheets per Teacher
        for t_name in sorted(p.teachers.keys()):
            grid_df, _ = build_grid_dataframe(timetable, view_type="teacher", entity_name=t_name)
            clean_name = t_name.replace(":", "_").replace("/", "_").replace("\\", "_")
            sheet_title = f"T_{clean_name}"[:31]
            grid_df.to_excel(writer, sheet_name=sheet_title)

        # 3. Sheets per Room
        for r_name in sorted(p.rooms.keys()):
            grid_df, _ = build_grid_dataframe(timetable, view_type="room", entity_name=r_name)
            sheet_title = f"Room_{r_name}"[:31]
            grid_df.to_excel(writer, sheet_name=sheet_title)

        # 4. Flat table of all sessions
        flat_df = build_flat_sessions_dataframe(timetable)
        flat_df.to_excel(writer, sheet_name="All_Sessions", index=False)

    return output.getvalue()
