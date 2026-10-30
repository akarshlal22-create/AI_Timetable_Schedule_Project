"""ui/timetable_tab.py — Timetable grid views, conflict highlighting, manual editor, and exports."""

from __future__ import annotations

import streamlit as st

from core.cost import evaluate
from core.models import Session
from services.editing import apply_move, preview_move
from services.exporter import (
    export_flat_csv,
    export_grid_csv,
    export_timetable_excel,
)
from services.grid_builder import (
    build_grid_dataframe,
    get_day_labels,
    style_timetable_grid,
)
from ui.state import (
    apply_manual_edit,
    get_current_timetable,
    get_move_history,
    get_problem,
    reset_manual_edits,
    undo_manual_edit,
)


def render_timetable_tab() -> None:
    """Render the timetable grid display, manual editor, and export controls."""
    st.subheader("Timetable Views")

    problem = get_problem()
    current_tt = get_current_timetable()

    if problem is None:
        st.warning("Please load problem data in the Data tab first.")
        return

    if current_tt is None or not current_tt.assignment:
        st.info("No timetable generated yet. Run a solver in the Generate tab.")
        return

    # ------------------------------------------------------------------
    # 1. One-line status recomputed from core/cost.py
    # ------------------------------------------------------------------
    bd = evaluate(current_tt, complete=True)
    if bd.total_hard_violations == 0:
        st.success(
            f"Current Schedule: FEASIBLE · Hard Violations: 0 · "
            f"Soft Penalty: {bd.total_soft_penalty} · Cost: {bd.total_cost}"
        )
    else:
        st.warning(
            f"Current Schedule: CONFLICTS PRESENT · Hard Violations: {bd.total_hard_violations} · "
            f"Soft Penalty: {bd.total_soft_penalty} · Total Cost: {bd.total_cost}"
        )

    # ------------------------------------------------------------------
    # 2. View selector & entity selection
    # ------------------------------------------------------------------
    col_view, col_entity = st.columns([1, 2])

    with col_view:
        view_type = st.radio(
            "View By",
            options=["Class", "Teacher", "Room"],
            horizontal=True,
        )

    with col_entity:
        if view_type == "Class":
            entity_options = sorted(problem.classes.keys())
        elif view_type == "Teacher":
            entity_options = sorted(problem.teachers.keys())
        else:
            entity_options = sorted(problem.rooms.keys())

        selected_entity = st.selectbox(f"Select {view_type}", options=entity_options)

    # Build and style the grid DataFrame
    grid_df, conflict_mask = build_grid_dataframe(
        current_tt,
        view_type=view_type.lower(),  # type: ignore[arg-type]
        entity_name=selected_entity,
    )
    styled_grid = style_timetable_grid(grid_df, conflict_mask)
    st.dataframe(styled_grid, use_container_width=True)

    # ------------------------------------------------------------------
    # 3. Export Buttons (plain row)
    # ------------------------------------------------------------------
    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        st.download_button(
            label="Download Current Grid (CSV)",
            data=export_grid_csv(grid_df),
            file_name=f"{view_type.lower()}_{selected_entity}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_exp2:
        st.download_button(
            label="Download All Sessions (CSV)",
            data=export_flat_csv(current_tt),
            file_name="all_sessions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_exp3:
        st.download_button(
            label="Download Workbook (Excel .xlsx)",
            data=export_timetable_excel(current_tt),
            file_name="timetable_schedule.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.divider()

    # ------------------------------------------------------------------
    # 4. Manual Editing Section
    # ------------------------------------------------------------------
    st.subheader("Edit Session")

    # Filter sessions by class
    all_classes = sorted(problem.classes.keys())
    edit_class = st.selectbox("Class Filter", options=all_classes, key="edit_filter_class")

    # Get sessions for this class
    class_sessions = [s for s in current_tt.assignment.keys() if s.class_name == edit_class]
    class_sessions.sort(key=lambda x: (x.subject_code, x.session_index))

    if not class_sessions:
        st.info("No sessions available for this class.")
        return

    def session_format(s: Session) -> str:
        d, per, rm = current_tt.assignment[s]
        kind = "Lab (2 periods)" if s.is_lab else "Theory (1 period)"
        return f"{s.subject_code} #{s.session_index} [{kind}] — Current: Day {d}, Period {per}, Room {rm} (Teacher: {s.teacher_name})"

    selected_session = st.selectbox(
        "Select Session to Move",
        options=class_sessions,
        format_func=session_format,
        key="edit_selected_session",
    )

    cur_d, cur_p, cur_r = current_tt.assignment[selected_session]

    col_d, col_p, col_r = st.columns(3)
    t = problem.time
    day_labels = get_day_labels(t.working_days)

    with col_d:
        target_day = st.selectbox(
            "Target Day",
            options=list(range(t.working_days)),
            format_func=lambda d: f"{day_labels[d]} (Day {d})",
            index=cur_d if cur_d < t.working_days else 0,
            key="edit_target_day",
        )

    with col_p:
        target_period = st.selectbox(
            "Target Period",
            options=list(range(t.periods_per_day)),
            format_func=lambda p_idx: f"Period {p_idx}",
            index=cur_p if cur_p < t.periods_per_day else 0,
            key="edit_target_period",
        )

    with col_r:
        all_rooms = sorted(problem.rooms.keys())
        room_idx = all_rooms.index(cur_r) if cur_r in all_rooms else 0
        target_room = st.selectbox(
            "Target Room (All Rooms)",
            options=all_rooms,
            index=room_idx,
            key="edit_target_room",
        )

    # Conflict preview without applying
    preview_conflicts = preview_move(
        current_tt,
        selected_session,
        int(target_day),
        int(target_period),
        str(target_room),
    )

    if preview_conflicts:
        st.warning(f"Preview: Move creates {len(preview_conflicts)} hard conflict(s):")
        for c in preview_conflicts:
            st.write(f"• {c}")
    else:
        st.info("Preview: No hard-constraint conflicts for this slot.")

    # Action buttons: Apply, Undo, Reset
    col_apply, col_undo, col_reset = st.columns([1, 1, 1])

    with col_apply:
        if st.button("Apply Move", type="primary", use_container_width=True):
            apply_manual_edit(
                selected_session,
                int(target_day),
                int(target_period),
                str(target_room),
            )
            st.rerun()

    history = get_move_history()
    can_undo = history is not None and history.can_undo()

    with col_undo:
        if st.button("Undo", disabled=not can_undo, use_container_width=True):
            undo_manual_edit()
            st.rerun()

    with col_reset:
        if st.button("Reset to Original", disabled=not can_undo, use_container_width=True):
            reset_manual_edits()
            st.rerun()
