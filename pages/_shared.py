"""
pages/_shared.py
────────────────
Shared bootstrap imported by every page file.
Handles sys.path, session state init, RAG engine, and sidebar rendering.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC  = _ROOT / "src"
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st
import pandas as pd

from data_processing import clean_dataset
from rag_engine import RAGEngine
from config import is_groq_configured


def init_session() -> None:
    """Initialise session state defaults. No auto-load."""
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("index_built", False)
    st.session_state.setdefault("df", None)

    if "rag" not in st.session_state:
        st.session_state.rag = RAGEngine()


def load_and_index(df: pd.DataFrame) -> None:
    st.session_state.df = df
    st.session_state.chat_history = []
    with st.spinner("Indexing data…"):
        st.session_state.rag.build_index(df)
    st.session_state.index_built = True


def render_sidebar() -> None:
    """Sidebar: only dataset upload + API key notice."""

    st.markdown('<div class="sidebar-label">Dataset</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        try:
            load_and_index(clean_dataset(pd.read_csv(uploaded)))
            st.success("Dataset loaded")
        except Exception as e:
            st.error(str(e))

    if st.session_state.df is None:
        st.markdown("""
        <div style="font-size:.78rem;color:#475569;margin-top:.6rem;line-height:1.6;">
          Upload a CSV with columns:<br>
          <code style="color:#64748b;">Order_ID, Warehouse, Product,<br>
          Order_Date, Ship_Date, Destination</code>
        </div>""", unsafe_allow_html=True)

    if not is_groq_configured():
        st.markdown("---")
        st.markdown("""
        <div style="font-size:.75rem;color:#64748b;line-height:1.6;">
          Add <code style="background:#1f2937;padding:1px 4px;border-radius:3px;
          color:#94a3b8;">GROQ_API_KEY</code> to <code style="background:#1f2937;
          padding:1px 4px;border-radius:3px;color:#94a3b8;">.env</code>
          to enable AI chat.
        </div>""", unsafe_allow_html=True)
