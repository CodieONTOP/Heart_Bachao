from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Heart risk explorer",
    page_icon=":material/favorite:",
    layout="wide",
)

PAGES_DIR = Path(__file__).resolve().parent / "app_pages"

pages = [
    st.Page(
        PAGES_DIR / "risk.py",
        title="Risk assessment",
        icon=":material/monitor_heart:",
        default=True,
    ),
    st.Page(
        PAGES_DIR / "batch.py",
        title="Batch scoring",
        icon=":material/table_view:",
    ),
    st.Page(
        PAGES_DIR / "insights.py",
        title="Model insights",
        icon=":material/insights:",
    ),
    st.Page(
        PAGES_DIR / "about.py",
        title="About & methods",
        icon=":material/info:",
    ),
]

page = st.navigation(pages, position="top")
page.run()
