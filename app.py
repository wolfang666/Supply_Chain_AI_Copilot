"""
app.py — Supply Chain AI Copilot
Entry point. Uses st.navigation() to wire up the three pages.
All page logic lives in pages/1_assistant.py, 2_analytics.py, 3_explorer.py.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

pg = st.navigation([
    st.Page("pages/1_assistant.py",  title="Assistant",   icon="💬", default=True),
    st.Page("pages/2_analytics.py",  title="Analytics",   icon="📊"),
    st.Page("pages/3_explorer.py",   title="Data Explorer", icon="🔍"),
])

pg.run()
