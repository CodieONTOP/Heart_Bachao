from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Heart risk explorer",
    page_icon=":material/favorite:",
    layout="wide",
)

pages = [
    st.Page(
        "app_pages/risk.py",
        title="Risk assessment",
        icon=":material/monitor_heart:",
        default=True,
    ),
    st.Page(
        "app_pages/batch.py",
        title="Batch scoring",
        icon=":material/table_view:",
    ),
    st.Page(
        "app_pages/insights.py",
        title="Model insights",
        icon=":material/insights:",
    ),
    st.Page(
        "app_pages/about.py",
        title="About & methods",
        icon=":material/info:",
    ),
]

page = st.navigation(pages, position="top")
page.run()
