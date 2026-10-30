"""app.py — Streamlit application entry point."""

from __future__ import annotations

import streamlit as st

from ui.data_tab import render_data_tab
from ui.generate_tab import render_generate_tab
from ui.state import init_session_state
from ui.style import apply_theme
from ui.timetable_tab import render_timetable_tab

st.set_page_config(
    page_title="AI Timetable Scheduler",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_theme()
init_session_state()

st.title("AI Timetable Scheduler")

tab_data, tab_generate, tab_timetable = st.tabs(["Data", "Generate", "Timetable"])

with tab_data:
    render_data_tab()

with tab_generate:
    render_generate_tab()

with tab_timetable:
    render_timetable_tab()
