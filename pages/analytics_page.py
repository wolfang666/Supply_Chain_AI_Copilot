"""
pages/analytics_page.py
────────────────────────
Renders the Analytics tab: five Plotly charts + three KPI tables.
"""

from __future__ import annotations
import sys
from pathlib import Path


_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st
import pandas as pd

from analytics import (
    avg_delay_per_warehouse,
    fastest_shipping_product,
    orders_delayed_more_than_n,
)
from ui.charts import (
    chart_avg_delay_per_warehouse,
    chart_fastest_products,
    chart_delay_heatmap,
    chart_monthly_volume,
    chart_top_delayed_destinations,
)
from ui.components import section_title

_CHART_CONFIG = {"displayModeBar": False}


def _chart_card(fig) -> None:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config=_CHART_CONFIG)
    st.markdown("</div>", unsafe_allow_html=True)


def render(df: pd.DataFrame) -> None:
    """Render the full Analytics tab."""
    section_title("📊 Supply Chain Analytics Dashboard")

    c1, c2 = st.columns(2)
    with c1:
        _chart_card(chart_avg_delay_per_warehouse(df))
    with c2:
        _chart_card(chart_fastest_products(df))

    c3, c4 = st.columns(2)
    with c3:
        _chart_card(chart_monthly_volume(df))
    with c4:
        _chart_card(chart_top_delayed_destinations(df))

    _chart_card(chart_delay_heatmap(df))

    section_title("📋 KPI Tables")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("**Avg Delay by Warehouse**")
        st.dataframe(avg_delay_per_warehouse(df), use_container_width=True, hide_index=True)
    with t2:
        st.markdown("**Product Speed Ranking**")
        st.dataframe(fastest_shipping_product(df), use_container_width=True, hide_index=True)
    with t3:
        delayed_df = orders_delayed_more_than_n(df)[["Order_ID", "Warehouse", "Product", "delay_days"]]
        st.markdown(f"**Orders Delayed >3d ({len(delayed_df)})**")
        st.dataframe(delayed_df.head(15), use_container_width=True, hide_index=True)
