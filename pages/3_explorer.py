"""
pages/3_explorer.py — Data Explorer
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared

import streamlit as st
from ui.styles import inject_css
from ui.components import (
    render_hero, render_metric_grid, render_no_data_placeholder, section_title,
)
from data_processing import get_summary_stats

st.set_page_config(page_title="Data Explorer — Supply Chain", page_icon="🔍", layout="wide")
inject_css()
_shared.init_session()

with st.sidebar:
    _shared.render_sidebar()

if st.session_state.df is None:
    render_no_data_placeholder()
    st.stop()

df = st.session_state.df
stats = get_summary_stats(df)

render_hero("Data Explorer", "Filter and inspect raw order records.")
render_metric_grid([
    {"label": "Total Orders",         "value": f"{stats['total_orders']:,}",          "sub": "all warehouses",             "icon": "📦"},
    {"label": "Avg Processing Time",  "value": f"{stats['avg_processing_time']}d",    "sub": "order to dispatch",          "icon": "⚙️"},
    {"label": "Avg Shipping Delay",   "value": f"{stats['avg_shipping_delay']}d",     "sub": "dispatch to delivery",       "icon": "🚚"},
    {"label": "Avg Total Lead Time",  "value": f"{stats['avg_total_lead_time']}d",    "sub": "order to delivery",          "icon": "📍"},
])
st.markdown("---")
section_title("Filter orders")

f1, f2, f3 = st.columns(3)
with f1:
    sel_wh = st.selectbox("Warehouse", ["All"] + sorted(df["Warehouse"].unique().tolist()), key="ex_wh")
with f2:
    sel_prod = st.selectbox("Product", ["All"] + sorted(df["Product"].unique().tolist()), key="ex_prod")
with f3:
    max_d = int(df["shipping_delay"].max())
    delay_cap = st.slider("Max delay (days)", 0, max_d, max_d, key="ex_delay")

filtered = df.copy()
if sel_wh != "All":
    filtered = filtered[filtered["Warehouse"] == sel_wh]
if sel_prod != "All":
    filtered = filtered[filtered["Product"] == sel_prod]
filtered = filtered[filtered["shipping_delay"] <= delay_cap]

st.markdown(
    f'<div style="font-size:.78rem;color:#475569;margin-bottom:.4rem;">'
    f"Showing {len(filtered):,} of {len(df):,} orders</div>",
    unsafe_allow_html=True,
)


def _color_delay(val):
    try:
        max_val = filtered["shipping_delay"].max() or 1
        ratio = min(float(val) / max_val, 1.0)
        r = int(255 * ratio)
        g = int(160 * (1 - ratio))
        return f"background-color:rgba({r},{g},60,0.3);color:#f1f5f9"
    except Exception:
        return ""


st.dataframe(
    filtered.style.applymap(_color_delay, subset=["shipping_delay"]),
    use_container_width=True,
    height=480,
)
