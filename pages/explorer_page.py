"""
pages/explorer_page.py
──────────────────────
Renders the Data Explorer tab: live-filtered dataframe with
warehouse, product, and delay-range controls.
"""

from __future__ import annotations
import sys
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd

from ui.components import section_title


def render(df: pd.DataFrame) -> None:
    """Render the full Data Explorer tab."""
    section_title("🔍 Raw Data Explorer")

    f1, f2, f3 = st.columns(3)
    with f1:
        wh_options = ["All"] + sorted(df["Warehouse"].unique().tolist())
        selected_wh = st.selectbox("Filter by Warehouse", wh_options)
    with f2:
        prod_options = ["All"] + sorted(df["Product"].unique().tolist())
        selected_prod = st.selectbox("Filter by Product", prod_options)
    with f3:
        max_delay = int(df["delay_days"].max())
        delay_cap = st.slider("Max Delay (days)", 0, max_delay, max_delay)

    filtered = df.copy()
    if selected_wh != "All":
        filtered = filtered[filtered["Warehouse"] == selected_wh]
    if selected_prod != "All":
        filtered = filtered[filtered["Product"] == selected_prod]
    filtered = filtered[filtered["delay_days"] <= delay_cap]

    st.markdown(
        f'<div style="font-size:.8rem;color:#64748b;margin-bottom:.5rem;">'
        f"Showing {len(filtered):,} of {len(df):,} orders"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        filtered.style.background_gradient(subset=["delay_days"], cmap="YlOrRd"),
        use_container_width=True,
        height=450,
    )
