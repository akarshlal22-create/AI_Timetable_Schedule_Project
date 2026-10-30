"""ui/style.py — Minimal styling for typography and clean spacing."""

from __future__ import annotations

import streamlit as st

_CUSTOM_CSS = """
<style>
/* Clean typography and tight spacing */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1E293B;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 100%;
}
h1, h2, h3, h4 {
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #0F172A;
}
hr {
    margin-top: 1rem;
    margin-bottom: 1rem;
    border-color: #E2E8F0;
}
</style>
"""


def apply_theme() -> None:
    """Inject minimal styling into the Streamlit app."""
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
