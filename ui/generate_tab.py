"""ui/generate_tab.py — Algorithm selection, generation triggers, and execution metrics."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.solver_registry import ALGORITHM_OPTIONS, create_solver
from ui.state import (
    get_problem,
    get_solver_result,
    get_validation_errors,
    set_solver_result,
)


def render_generate_tab() -> None:
    """Render the algorithm selection and generation controls."""
    st.subheader("Generate Timetable")

    problem = get_problem()
    val_errors = get_validation_errors()

    if problem is None:
        st.warning("Please load a problem in the Data tab first.")
        return

    if val_errors:
        st.error("Cannot generate timetable while blocking validation errors exist in the Data tab.")
        return

    col_algo, col_seed, col_time = st.columns([3, 1, 1])

    with col_algo:
        algo_choice = st.selectbox(
            "Algorithm",
            options=ALGORITHM_OPTIONS,
            index=2,  # Default to Min-Conflicts for lightning-fast solving on large datasets
            help="For large or complex problems (>100 sessions), local search (Min-Conflicts) solves in seconds, while Backtracking baseline explores an exponential search tree.",
        )

    with col_seed:
        seed = st.number_input("Random Seed", min_value=0, max_value=999999, value=42, step=1)

    with col_time:
        time_limit = st.number_input("Time Limit (s)", min_value=1.0, max_value=600.0, value=60.0, step=5.0)

    # Recommendations based on problem complexity
    num_sessions = sum(
        problem.subjects[code].weekly_hours
        for c in problem.classes.values()
        for code in c.subject_codes
        if code in problem.subjects
    )
    if num_sessions > 100:
        if "baseline" in algo_choice.lower():
            st.warning(
                f"⚠️ **Backtracking (baseline)** explores an unpruned exponential search space and will time out on this large problem ({num_sessions} sessions). "
                f"**Recommendation**: Select **Min-Conflicts** for instant solving (~1.4s) or **Backtracking (MRV + Forward Checking)**."
            )
        elif "simulated annealing" in algo_choice.lower() and "hybrid" not in algo_choice.lower():
            st.info(
                f"ℹ️ **Simulated Annealing** starts from a random assignment and may freeze with remaining violations on large problems. "
                f"**Recommendation**: Use **Min-Conflicts** or **Hybrid (Backtracking + SA)** for conflict-free timetables."
            )

    generate_clicked = st.button("Generate Timetable", type="primary")

    if generate_clicked:
        # Clear previous result while new run executes
        set_solver_result(None)
        with st.spinner(f"Solving CSP problem using {algo_choice} (limit {time_limit}s)..."):
            solver = create_solver(algo_choice, time_limit=float(time_limit))
            result = solver.solve(problem, seed=int(seed))
            set_solver_result(result)

    result = get_solver_result()
    if result is not None:
        st.divider()

        # Plain one-row status summary
        if result.success:
            st.success(
                f"Result: {result.status.upper()} · "
                f"Time: {result.time_seconds:.4f}s · "
                f"Nodes/Iterations: {result.nodes_expanded} · "
                f"Hard Violations: {result.hard_violations} · "
                f"Soft Penalty: {result.soft_penalty} · "
                f"Total Cost: {result.cost}"
            )
        else:
            st.warning(
                f"Result: {result.status.upper()} (Incomplete or Infeasible) · "
                f"Time: {result.time_seconds:.4f}s · "
                f"Nodes/Iterations: {result.nodes_expanded} · "
                f"Hard Violations: {result.hard_violations} · "
                f"Soft Penalty: {result.soft_penalty} · "
                f"Best Cost: {result.cost}"
            )

        # Hybrid specific phase metrics
        if "soft_before" in result.details and "soft_after" in result.details:
            s0 = result.details["soft_before"]
            sf = result.details["soft_after"]
            improvement = s0 - sf
            pct = (improvement / s0 * 100.0) if s0 > 0 else 0.0
            st.caption(
                f"Hybrid Refinement: Phase 1 Soft Penalty = {s0}  →  "
                f"Phase 2 Soft Penalty = {sf}  (reduced by {improvement} points, {pct:.1f}% improvement)"
            )

        # Cost vs. Iterations Chart for stochastic local search algorithms
        if len(result.cost_history) > 1:
            st.subheader("Cost vs. Iterations")
            chart_df = pd.DataFrame({"Cost": result.cost_history})
            st.line_chart(chart_df, color="#2563EB")
