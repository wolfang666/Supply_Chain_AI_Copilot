"""
pages/2_analytics.py — Visualization Dashboard
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared

import streamlit as st
from ui.styles import inject_css
from ui.components import render_hero, render_metric_grid, render_no_data_placeholder, section_title
from ui.charts import (
    chart_processing_vs_shipping_per_warehouse,
    chart_processing_vs_shipping_per_product,
    chart_delay_heatmap,
    chart_processing_heatmap,
    chart_monthly_volume,
    chart_top_delayed_destinations,
    chart_warehouse_breakdown,
)
from analytics import (
    avg_delay_per_warehouse,
    avg_processing_time_per_warehouse,
    avg_shipping_delay_per_warehouse,
    fastest_shipping_product,
    orders_delayed_more_than_n,
)
from data_processing import get_summary_stats

st.set_page_config(page_title="Analytics — Supply Chain", page_icon="📊", layout="wide")
inject_css()
_shared.init_session()

with st.sidebar:
    _shared.render_sidebar()

if st.session_state.df is None:
    render_no_data_placeholder()
    st.stop()

df = st.session_state.df
stats = get_summary_stats(df)
_CFG = {"displayModeBar": False}

render_hero("Visualization Dashboard", "Processing time, shipping delay and lead time across your supply chain.")
render_metric_grid([
    {"label": "Avg Processing Time", "value": f"{stats['avg_processing_time']}d", "sub": "order to dispatch",        "icon": "⚙️"},
    {"label": "Avg Shipping Delay",  "value": f"{stats['avg_shipping_delay']}d",  "sub": "dispatch to delivery",     "icon": "🚚"},
    {"label": "Avg Total Lead Time", "value": f"{stats['avg_total_lead_time']}d", "sub": "order to delivery",        "icon": "📦"},
    {"label": "Delayed Orders",      "value": f"{stats['delayed_orders']:,}",     "sub": f"{stats['delay_rate_pct']}% exceed threshold", "icon": "⚠️"},
])
st.markdown("---")


def _card(fig, key):
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config=_CFG, key=key)
    st.markdown("</div>", unsafe_allow_html=True)


section_title("Processing Time & Shipping Delay")
c1, c2 = st.columns(2)
with c1:
    _card(chart_processing_vs_shipping_per_warehouse(df), "an_wh")
with c2:
    _card(chart_processing_vs_shipping_per_product(df), "an_prod")

_card(chart_warehouse_breakdown(df), "an_stack")

section_title("Heatmaps")
c3, c4 = st.columns(2)
with c3:
    _card(chart_processing_heatmap(df), "an_proc_heat")
with c4:
    _card(chart_delay_heatmap(df), "an_ship_heat")

section_title("Volume & Destinations")
c5, c6 = st.columns(2)
with c5:
    _card(chart_monthly_volume(df), "an_monthly")
with c6:
    _card(chart_top_delayed_destinations(df), "an_dest")

st.markdown("---")
section_title("KPI tables")

t1, t2, t3 = st.columns(3)
with t1:
    st.markdown("**Processing time by warehouse**")
    st.dataframe(avg_processing_time_per_warehouse(df), use_container_width=True, hide_index=True)
with t2:
    st.markdown("**Shipping delay by warehouse**")
    st.dataframe(avg_shipping_delay_per_warehouse(df), use_container_width=True, hide_index=True)
with t3:
    delayed_df = orders_delayed_more_than_n(df)[["Order_ID", "Warehouse", "Product", "order_processing_time", "shipping_delay", "total_lead_time"]]
    st.markdown(f"**Orders with high shipping delay ({len(delayed_df)})**")
    st.dataframe(delayed_df.head(15), use_container_width=True, hide_index=True)
