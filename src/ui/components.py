"""
ui/components.py
────────────────
Reusable HTML/Streamlit component builders.
"""

from __future__ import annotations
import re
import streamlit as st


# ── Page header ───────────────────────────────────────────────────────────────

def render_hero(title: str, subtitle: str, badges: list[str] = None) -> None:
    st.markdown(f"""
    <div class="page-header">
      <div class="page-title">{title}</div>
      <p class="page-sub">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ── Metric cards ─────────────────────────────────────────────────────────────

def metric_card_html(label: str, value: str, sub: str = "", icon: str = "") -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-icon">{icon}</div>
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>"""


def render_metric_grid(cards: list[dict]) -> None:
    inner = "".join(
        metric_card_html(c["label"], c["value"], c.get("sub", ""), c.get("icon", ""))
        for c in cards
    )
    st.markdown(f'<div class="metric-grid">{inner}</div>', unsafe_allow_html=True)


# ── Section title ─────────────────────────────────────────────────────────────

def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


# ── Chat ──────────────────────────────────────────────────────────────────────

def render_chat_history(messages: list[dict]) -> None:
    html = '<div class="chat-container">'

    if not messages:
        html += """
        <div class="chat-empty">
          <div class="chat-empty-icon">💬</div>
          <div class="chat-empty-text">Ask a question to get started</div>
          <div class="chat-empty-hint">Try: "Which warehouse has the highest delay?"</div>
        </div>"""
    else:
        for msg in messages:
            if msg["role"] == "user":
                html += f"""
                <div class="chat-bubble-user">
                  <div class="chat-bubble-user-inner">{msg['content']}</div>
                </div>"""
            else:
                html += f"""
                <div class="chat-bubble-ai">
                  <div class="chat-avatar">✦</div>
                  <div class="chat-bubble-ai-inner">{msg['content']}</div>
                </div>"""

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ── Insight cards ─────────────────────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    """Convert **bold** markdown to <strong> tags for use inside HTML."""
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


# Accent colour per card type
_INSIGHT_ACCENT = {
    "danger":  "#ef4444",
    "success": "#10b981",
    "warning": "#f59e0b",
    "info":    "#4f46e5",
}

# Icon stripped from text, used as the visual dot colour signal
_ICON_MAP = {
    "🔴": "danger",
    "🟢": "success",
    "⚡": "success",
    "⚠️": "warning",
    "🚚": "info",
    "🐢": "warning",
    "📍": "info",
    "📊": "info",
}


def render_insight_cards(insights: list[str]) -> None:
    # Render as a 2-column grid
    cols = st.columns(2)
    for i, ins in enumerate(insights):
        kind = next((v for k, v in _ICON_MAP.items() if k in ins), "info")
        accent = _INSIGHT_ACCENT[kind]

        # Strip leading emoji for cleaner text
        text = ins
        for icon in _ICON_MAP:
            text = text.replace(icon, "").strip()
        text = _md_to_html(text)

        with cols[i % 2]:
            st.markdown(f"""
            <div style="
              background:var(--card);
              border:1px solid var(--border);
              border-left:3px solid {accent};
              border-radius:10px;
              padding:.85rem 1.1rem;
              margin-bottom:.65rem;
              font-size:.85rem;
              color:#cbd5e1;
              line-height:1.6;
            ">{text}</div>""", unsafe_allow_html=True)


def render_executive_summary(narrative: str) -> None:
    st.markdown(f"""
    <div class="summary-box">
      <div class="summary-label">Summary</div>
      <div class="summary-text">{narrative}</div>
    </div>""", unsafe_allow_html=True)


# ── No-data placeholder ───────────────────────────────────────────────────────

def render_no_data_placeholder() -> None:
    st.markdown("""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:12px;
                padding:3rem 2rem;text-align:center;margin-top:2rem;">
      <div style="font-size:2rem;margin-bottom:.8rem;opacity:.4;">📦</div>
      <div style="font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:.4rem;">No data loaded</div>
      <div style="color:#475569;font-size:.85rem;max-width:380px;margin:0 auto;line-height:1.6;">
        Use the sidebar to load the sample dataset or upload a CSV file.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar stats ─────────────────────────────────────────────────────────────

def render_sidebar_stats(stats: dict) -> None:
    rows = [
        ("Total Orders",    f"{stats['total_orders']:,}",   ""),
        ("Avg Delay",       f"{stats['avg_delay_days']} days", ""),
        ("Delayed >3d",     f"{stats['delay_rate_pct']}%",  "color:#f59e0b"),
        ("Warehouses",      str(stats['warehouses']),        ""),
        ("Products",        str(stats['products']),          ""),
        ("Date Range",      f"{stats['date_range_start']} → {stats['date_range_end']}", "font-size:.78rem"),
    ]
    rows_html = "".join(f"""
    <div class="stat-row">
      <div class="sidebar-label">{label}</div>
      <div class="sidebar-value" style="{style}">{value}</div>
    </div>""" for label, value, style in rows)

    st.markdown(f'<div class="sidebar-stat-grid">{rows_html}</div>', unsafe_allow_html=True)
