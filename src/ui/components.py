"""
ui/components.py
────────────────
Reusable HTML/Streamlit component builders.
"""

from __future__ import annotations

import re
import textwrap

import streamlit as st


# ── Page header ───────────────────────────────────────────────────────────────

def render_hero(title: str, subtitle: str, badges: list[str] = None) -> None:
    st.markdown(
        textwrap.dedent(
            f"""
            <div class="page-header">
              <div class="page-title">{title}</div>
              <p class="page-sub">{subtitle}</p>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


# ── Metric cards ─────────────────────────────────────────────────────────────

def metric_card_html(
    label: str,
    value: str,
    sub: str = "",
    icon: str = "",
) -> str:
    return textwrap.dedent(
        f"""
        <div class="metric-card">
          <div class="metric-icon">{icon}</div>
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-sub">{sub}</div>
        </div>
        """
    ).strip()


def render_metric_grid(cards: list[dict]) -> None:
    inner = "".join(
        metric_card_html(
            c["label"],
            c["value"],
            c.get("sub", ""),
            c.get("icon", ""),
        )
        for c in cards
    )

    st.markdown(
        f'<div class="metric-grid">{inner}</div>',
        unsafe_allow_html=True,
    )


# ── Section title ─────────────────────────────────────────────────────────────

def section_title(text: str) -> None:
    st.markdown(
        f'<div class="section-title">{text}</div>',
        unsafe_allow_html=True,
    )


# ── Chat ──────────────────────────────────────────────────────────────────────

def _safe_user_text(text: str) -> str:
    """
    Escape user-supplied text so it cannot break the surrounding HTML.
    Only minimal escaping is applied: angle brackets and ampersands.
    """
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _fix_escaped_markdown(text: str) -> str:
    """
    The LLM sometimes emits escaped Markdown markers like \\*\\*bold\\*\\*.
    Restore them so st.markdown renders them correctly.
    """

    # Un-escape \*\* → **
    text = re.sub(
        r"\\\*\\\*(.+?)\\\*\\\*",
        r"**\1**",
        text,
    )

    # Un-escape \* → *
    text = re.sub(
        r"\\\*(.+?)\\\*",
        r"*\1*",
        text,
    )

    return text


def render_chat_history(messages: list[dict]) -> None:
    """
    Render the chat history inside a fixed-height scrollable container.

    User messages are rendered using custom HTML bubbles with escaped
    text so user content cannot break the surrounding page HTML.

    Assistant messages are rendered through Streamlit's Markdown engine
    so Markdown tables, bold text, lists, etc. render correctly.

    A real Streamlit container is used for scrolling. This avoids
    opening a <div> in one st.markdown() call and closing it in another,
    which can cause malformed DOM/layout behavior in Streamlit.
    """

    # ------------------------------------------------------------------
    # Empty state
    # ------------------------------------------------------------------

    if not messages:
        st.markdown(
            textwrap.dedent(
                """
                <div class="chat-container">
                  <div class="chat-empty">
                    <div class="chat-empty-icon">💬</div>
                    <div class="chat-empty-text">
                      Ask a question to get started
                    </div>
                    <div class="chat-empty-hint">
                      Try: "Which warehouse has the highest delay?"
                    </div>
                  </div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )
        return

    # ------------------------------------------------------------------
    # Actual scrollable chat area
    # ------------------------------------------------------------------
    #
    # IMPORTANT:
    # Do not create <div class="chat-container"> here and then put
    # separate Streamlit elements inside it. Streamlit renders each
    # st.markdown() independently, so the HTML wrapper cannot reliably
    # contain the later elements.
    #
    # st.container(height=520) creates a real scrollable Streamlit
    # container.
    # ------------------------------------------------------------------

    with st.container(
        height=520,
        border=False,
    ):

        for msg in messages:

            # ==========================================================
            # USER MESSAGE
            # ==========================================================

            if msg["role"] == "user":

                safe_text = _safe_user_text(
                    msg["content"]
                )

                st.markdown(
                    textwrap.dedent(
                        f"""
                        <div class="chat-bubble-user">
                          <div class="chat-bubble-user-inner">
                            {safe_text}
                          </div>
                        </div>
                        """
                    ).strip(),
                    unsafe_allow_html=True,
                )

            # ==========================================================
            # ASSISTANT MESSAGE
            # ==========================================================

            else:

                # ------------------------------------------------------
                # AI avatar
                # ------------------------------------------------------

                st.markdown(
                    """
                    <div class="chat-bubble-ai">
                      <div class="chat-avatar">✦</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ------------------------------------------------------
                # AI content
                # ------------------------------------------------------
                #
                # Render through Streamlit Markdown so that:
                #
                #   **bold**
                #   tables
                #   lists
                #   paragraphs
                #
                # are handled correctly.
                # ------------------------------------------------------

                content = _fix_escaped_markdown(
                    msg["content"]
                )

                # Remove accidental HTML emitted by the LLM.
                #
                # This prevents things such as:
                #
                #     </div></div>
                #
                # from corrupting the page structure.
                content = re.sub(
                    r"<[^>]+>",
                    "",
                    content,
                )

                st.markdown(content)


# ── Insight cards ─────────────────────────────────────────────────────────────

def _md_to_html(text: str) -> str:
    """Convert **bold** markdown to <strong> tags for use inside HTML."""
    return re.sub(
        r"\*\*(.+?)\*\*",
        r"<strong>\1</strong>",
        text,
    )


# Accent colour per card type
_INSIGHT_ACCENT = {
    "danger": "#ef4444",
    "success": "#10b981",
    "warning": "#f59e0b",
    "info": "#4f46e5",
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
        kind = next(
            (
                v
                for k, v in _ICON_MAP.items()
                if k in ins
            ),
            "info",
        )

        accent = _INSIGHT_ACCENT[kind]

        # Strip leading emoji for cleaner text
        text = ins

        for icon in _ICON_MAP:
            text = text.replace(
                icon,
                "",
            ).strip()

        text = _md_to_html(text)

        with cols[i % 2]:
            st.markdown(
                textwrap.dedent(
                    f"""
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
                    ">{text}</div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )


def render_executive_summary(narrative: str) -> None:
    st.markdown(
        textwrap.dedent(
            f"""
            <div class="summary-box">
              <div class="summary-label">Summary</div>
              <div class="summary-text">{narrative}</div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


# ── No-data placeholder ───────────────────────────────────────────────────────

def render_no_data_placeholder() -> None:
    st.markdown(
        textwrap.dedent(
            """
            <div style="background:var(--card);border:1px solid var(--border);border-radius:12px;
                        padding:3rem 2rem;text-align:center;margin-top:2rem;">
              <div style="font-size:2rem;margin-bottom:.8rem;opacity:.4;">📦</div>
              <div style="font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:.4rem;">
                No data loaded
              </div>
              <div style="color:#475569;font-size:.85rem;max-width:380px;margin:0 auto;line-height:1.6;">
                Use the sidebar to load the sample dataset or upload a CSV file.
              </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )


# ── Sidebar stats ─────────────────────────────────────────────────────────────

def render_sidebar_stats(stats: dict) -> None:
    rows = [
        (
            "Total Orders",
            f"{stats['total_orders']:,}",
            "",
        ),
        (
            "Avg Delay",
            f"{stats['avg_delay_days']} days",
            "",
        ),
        (
            "Delayed >3d",
            f"{stats['delay_rate_pct']}%",
            "color:#f59e0b",
        ),
        (
            "Warehouses",
            str(stats["warehouses"]),
            "",
        ),
        (
            "Products",
            str(stats["products"]),
            "",
        ),
        (
            "Date Range",
            f"{stats['date_range_start']} → {stats['date_range_end']}",
            "font-size:.78rem",
        ),
    ]

    rows_html = "".join(
        textwrap.dedent(
            f"""
            <div class="stat-row">
              <div class="sidebar-label">{label}</div>
              <div class="sidebar-value" style="{style}">{value}</div>
            </div>
            """
        ).strip()
        for label, value, style in rows
    )

    st.markdown(
        f'<div class="sidebar-stat-grid">{rows_html}</div>',
        unsafe_allow_html=True,
    )