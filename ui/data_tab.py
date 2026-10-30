"""ui/data_tab.py — Data loading, validation, and read-only entity inspection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

from core.loader import load_problem_from_csv, load_problem_from_json
from core.models import Problem
from core.validation import validate_problem
from ui.state import get_problem, set_problem


def render_data_tab() -> None:
    """Render the Data input, validation, and entity preview tab."""
    st.subheader("Data Input")

    source_choice = st.radio(
        "Source",
        options=["Small sample", "Large sample", "Upload JSON", "Upload CSV files"],
        horizontal=True,
        label_visibility="collapsed",
    )

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    loaded_problem: Optional[Problem] = None
    errors: List[str] = []

    if source_choice == "Small sample":
        sample_path = data_dir / "small.json"
        if sample_path.exists():
            try:
                loaded_problem = load_problem_from_json(sample_path)
            except Exception as ex:
                errors.append(f"Failed to load small.json: {ex}")
        else:
            errors.append("data/small.json not found on disk.")

    elif source_choice == "Large sample":
        sample_path = data_dir / "large.json"
        if sample_path.exists():
            try:
                loaded_problem = load_problem_from_json(sample_path)
            except Exception as ex:
                errors.append(f"Failed to load large.json: {ex}")
        else:
            errors.append("data/large.json not found on disk.")

    elif source_choice == "Upload JSON":
        uploaded_json = st.file_uploader("Upload JSON problem file", type=["json"], label_visibility="collapsed")
        if uploaded_json is not None:
            try:
                with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False, encoding="utf-8") as tmp:
                    tmp.write(uploaded_json.getvalue().decode("utf-8"))
                    tmp.flush()
                    tmp_path = Path(tmp.name)
                loaded_problem = load_problem_from_json(tmp_path)
                tmp_path.unlink(missing_ok=True)
            except Exception as ex:
                errors.append(f"Invalid JSON file: {ex}")

    elif source_choice == "Upload CSV files":
        st.caption("Upload subjects.csv, teachers.csv, classes.csv, rooms.csv, time_structure.csv")
        uploaded_csvs = st.file_uploader(
            "Upload CSV files",
            type=["csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_csvs:
            expected = {"subjects.csv", "teachers.csv", "classes.csv", "rooms.csv", "time_structure.csv"}
            filenames = {f.name for f in uploaded_csvs}
            missing = expected - filenames
            if missing:
                errors.append(f"Missing required CSV file(s): {', '.join(sorted(missing))}")
            else:
                try:
                    with tempfile.TemporaryDirectory() as tmp_dir_str:
                        tmp_dir = Path(tmp_dir_str)
                        for uf in uploaded_csvs:
                            (tmp_dir / uf.name).write_bytes(uf.getvalue())
                        loaded_problem = load_problem_from_csv(tmp_dir, name="Uploaded CSV Problem")
                except Exception as ex:
                    errors.append(f"Error parsing CSV files: {ex}")

    # Validate problem if loaded
    if loaded_problem is not None:
        val_errors = validate_problem(loaded_problem)
        errors.extend(val_errors)

    # Check if problem changed or was newly selected
    current_problem = get_problem()
    should_update = (
        (loaded_problem is not None and loaded_problem != current_problem)
        or (loaded_problem is None and current_problem is not None and source_choice in ("Upload JSON", "Upload CSV files"))
    )

    if should_update:
        set_problem(loaded_problem, source_choice, errors)

    # Display errors or problem summary
    active_problem = get_problem()

    if errors:
        for err in errors:
            st.error(err)

    if active_problem is not None and not errors:
        p = active_problem
        t = p.time
        st.success(
            f"Loaded {p.name}: {len(p.subjects)} subjects, {len(p.teachers)} teachers, "
            f"{len(p.classes)} classes, {len(p.rooms)} rooms · "
            f"Grid: {t.working_days} days × {t.periods_per_day} periods/day"
        )

        st.divider()

        # Read-only entity table inspection
        inspect_entity = st.selectbox(
            "Inspect Entity Table",
            options=["Subjects", "Teachers", "Classes", "Rooms", "Time Structure"],
        )

        if inspect_entity == "Subjects":
            df = pd.DataFrame([
                {"Code": s.code, "Name": s.name, "Weekly Hours": s.weekly_hours, "Type": s.type}
                for s in p.subjects.values()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif inspect_entity == "Teachers":
            df = pd.DataFrame([
                {
                    "Teacher": t.name,
                    "Subjects": ", ".join(sorted(t.teachable_subjects)),
                    "Max Weekly Load": t.max_weekly_load,
                    "Unavailable Slots": ", ".join(f"({d},{per})" for d, per in sorted(t.unavailable_slots)) or "None",
                    "Preferences": str(t.preferences) if t.preferences else "None",
                }
                for t in p.teachers.values()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif inspect_entity == "Classes":
            df = pd.DataFrame([
                {
                    "Class": c.name,
                    "Strength": c.strength,
                    "Subject Codes": ", ".join(c.subject_codes),
                    "Lunch Slot": f"Period {c.lunch_slot}",
                }
                for c in p.classes.values()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif inspect_entity == "Rooms":
            df = pd.DataFrame([
                {"Room": r.name, "Capacity": r.capacity, "Type": r.type}
                for r in p.rooms.values()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

        elif inspect_entity == "Time Structure":
            df = pd.DataFrame([{
                "Working Days": t.working_days,
                "Periods Per Day": t.periods_per_day,
                "Global Breaks": ", ".join(f"Period {bp}" for bp in sorted(t.break_periods)) or "None",
            }])
            st.dataframe(df, use_container_width=True, hide_index=True)
