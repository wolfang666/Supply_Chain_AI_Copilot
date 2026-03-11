"""
pages/1_assistant.py — AI Assistant
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared

import streamlit as st
from config import is_groq_configured
from analytics import generate_auto_insights
from ui.styles import inject_css
from ui.components import (
    render_hero, render_metric_grid, render_chat_history,
    render_insight_cards, render_executive_summary,
    render_no_data_placeholder, section_title,
)
from ui.charts import detect_chart_intent, get_chart_for_intent
from data_processing import get_summary_stats

st.set_page_config(page_title="Assistant", page_icon="💬", layout="wide")
inject_css()
_shared.init_session()

with st.sidebar:
    _shared.render_sidebar()

if st.session_state.df is None:
    render_no_data_placeholder()
    st.stop()

df = st.session_state.df
rag = st.session_state.rag
stats = get_summary_stats(df)

render_hero("Supply Chain AI Copilot", "Ask questions about your supply chain data.")
render_metric_grid([
    {"label": "Total Orders",         "value": f"{stats['total_orders']:,}",          "sub": "all warehouses",             "icon": "📦"},
    {"label": "Avg Processing Time",  "value": f"{stats['avg_processing_time']}d",    "sub": "order to dispatch",          "icon": "⚙️"},
    {"label": "Avg Shipping Delay",   "value": f"{stats['avg_shipping_delay']}d",     "sub": "dispatch to delivery",       "icon": "🚚"},
    {"label": "Avg Total Lead Time",  "value": f"{stats['avg_total_lead_time']}d",    "sub": "order to delivery",          "icon": "📍"},
])
st.markdown("---")


def _ask(question: str) -> str:
    if not is_groq_configured():
        return "AI assistant is not configured. Add GROQ_API_KEY to .env and restart."
    if not rag.is_ready():
        return "Data index not ready."
    try:
        return rag.answer(question)
    except Exception as exc:
        return f"Something went wrong: {exc}"


left_col, right_col = st.columns([3, 2])

with left_col:
    render_chat_history(st.session_state.chat_history)

    question = st.text_input(
        "question", placeholder="Ask a question about your supply chain…",
        label_visibility="collapsed", key="chat_input",
    )
    b1, b2, b3 = st.columns([2, 2, 1])
    with b1:
        send_clicked = st.button("Ask", use_container_width=True, key="btn_ask")
    with b2:
        insights_clicked = st.button("Generate insights", use_container_width=True, key="btn_insights")
    with b3:
        if st.button("Clear", use_container_width=True, key="btn_clear"):
            st.session_state.chat_history = []
            st.rerun()

    if send_clicked and question.strip():
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Thinking…"):
            st.session_state.chat_history.append({"role": "assistant", "content": _ask(question)})
        st.rerun()

    last_q = next(
        (m["content"] for m in reversed(st.session_state.chat_history) if m["role"] == "user"), ""
    )
    
    if insights_clicked:
        section_title("Key insights")
        render_insight_cards(generate_auto_insights(df))
        if is_groq_configured():
            with st.spinner("Generating summary…"):
                try:
                    render_executive_summary(rag.generate_insights_narrative(df))
                except Exception as e:
                    st.error(str(e))

with right_col:
    section_title("Suggested questions")
    for i, q in enumerate([
        "Which warehouse has the highest average delay?",
        "What is the fastest shipping product?",
        "How many orders were delayed more than 3 days?",
        "Which destination has the worst shipping times?",
        "Compare delay performance across all warehouses.",
        "What percentage of orders are significantly delayed?",
        "Show me the delay breakdown by product and warehouse.",
        "What is the overall average shipping delay?",
    ]):
        if st.button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.spinner("Thinking…"):
                st.session_state.chat_history.append({"role": "assistant", "content": _ask(q)})
            st.rerun()
