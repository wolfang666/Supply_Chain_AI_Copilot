"""
pages/chat_page.py
──────────────────
Renders the Assistant tab.
"""

from __future__ import annotations
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st
import pandas as pd

from config import is_groq_configured
from analytics import generate_auto_insights
from ui.components import (
    render_chat_history,
    render_insight_cards,
    render_executive_summary,
    section_title,
)
from ui.charts import detect_chart_intent, get_chart_for_intent

_SUGGESTED = [
    "Which warehouse has the highest average delay?",
    "What is the fastest shipping product?",
    "How many orders were delayed more than 3 days?",
    "Which destination has the worst shipping times?",
    "Compare delay performance across all warehouses.",
    "What percentage of orders are significantly delayed?",
    "Show me the delay breakdown by product and warehouse.",
    "What is the overall average shipping delay?",
]


def _ask(question: str, rag, df: pd.DataFrame) -> str:
    if not is_groq_configured():
        return (
            "AI assistant is not configured. "
            "Add your API key to the `.env` file and restart the app. "
            "You can still explore the Analytics and Data tabs."
        )
    if not rag.is_ready():
        return "Data index is not ready — please reload the dataset."
    try:
        return rag.answer(question)
    except Exception as exc:
        return f"Something went wrong: {exc}"


def render(df: pd.DataFrame, rag) -> None:
    left_col, right_col = st.columns([3, 2])

    with left_col:
        render_chat_history(st.session_state.chat_history)

        question = st.text_input(
            "question",
            placeholder="Ask a question about your supply chain…",
            label_visibility="collapsed",
            key="chat_input",
        )

        btn1, btn2, btn3 = st.columns([2, 2, 1])
        with btn1:
            send_clicked = st.button("Ask", use_container_width=True)
        with btn2:
            insights_clicked = st.button("Generate insights", use_container_width=True)
        with btn3:
            if st.button("Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if send_clicked and question.strip():
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Thinking…"):
                answer = _ask(question, rag, df)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        # Auto-chart
        last_q = next(
            (m["content"] for m in reversed(st.session_state.chat_history) if m["role"] == "user"),
            "",
        )
        if last_q:
            intent = detect_chart_intent(last_q)
            if intent:
                fig = get_chart_for_intent(intent, df)
                if fig:
                    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.markdown("</div>", unsafe_allow_html=True)

        # Insights
        if insights_clicked:
            section_title("Key insights")
            render_insight_cards(generate_auto_insights(df))
            if is_groq_configured():
                with st.spinner("Generating summary…"):
                    try:
                        render_executive_summary(rag.generate_insights_narrative(df))
                    except Exception as exc:
                        st.error(str(exc))

    with right_col:
        section_title("Suggested questions")
        for q in _SUGGESTED:
            if st.button(q, key=f"sq_{q[:20]}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Thinking…"):
                    answer = _ask(q, rag, df)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()
