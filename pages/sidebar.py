"""
pages/sidebar.py
────────────────
Sidebar: dataset upload, stats, and AI status indicator.
Sample dataset is auto-loaded on first run.
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
from data_processing import load_dataset, clean_dataset, get_summary_stats
from rag_engine import RAGEngine
from ui.components import render_sidebar_stats


def _load_and_index(df: pd.DataFrame, rag: RAGEngine) -> None:
    st.session_state.df = df
    st.session_state.chat_history = []
    with st.spinner("Indexing data…"):
        rag.build_index(df)
    st.session_state.index_built = True


def render(rag: RAGEngine) -> None:
    if not st.session_state.get("index_built"):
        try:
            df = load_dataset()
            _load_and_index(df, rag)
        except Exception as exc:
            st.error(f"Could not auto-load dataset: {exc}")

    st.markdown("""
    <div style="padding:.4rem 0 1.2rem;">
      <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1rem;
                  font-weight:800;color:#f1f5f9;letter-spacing:-.01em;">
        Supply Chain
      </div>
      <div style="font-size:.72rem;color:#374151;margin-top:1px;font-weight:500;
                  text-transform:uppercase;letter-spacing:.06em;">
        Analytics
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="sidebar-label">Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your own CSV", type=["csv"], label_visibility="collapsed")

    if uploaded:
        try:
            df = clean_dataset(pd.read_csv(uploaded))
            _load_and_index(df, rag)
            st.success("Dataset loaded")
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.get("df") is not None:
        st.markdown("---")
        st.markdown('<div class="sidebar-label">Overview</div>', unsafe_allow_html=True)
        render_sidebar_stats(get_summary_stats(st.session_state.df))

        st.markdown("---")
        st.markdown('<div class="sidebar-label">Preview</div>', unsafe_allow_html=True)
        st.dataframe(st.session_state.df.head(8), use_container_width=True, height=190)

    st.markdown("---")
    if is_groq_configured():
        st.markdown("""
        <div style="display:flex;align-items:center;gap:6px;font-size:.75rem;color:#64748b;">
          <span style="width:6px;height:6px;background:#10b981;border-radius:50%;display:inline-block;"></span>
          AI assistant ready
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size:.75rem;color:#64748b;line-height:1.6;">
          <span style="width:6px;height:6px;background:#f59e0b;border-radius:50%;
          display:inline-block;margin-right:5px;"></span>
          Add <code style="background:#1f2937;padding:1px 4px;border-radius:3px;
          color:#94a3b8;">GROQ_API_KEY</code> to .env to enable AI chat.
        </div>
        """, unsafe_allow_html=True)
