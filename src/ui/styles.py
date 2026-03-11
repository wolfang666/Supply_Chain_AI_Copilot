"""
ui/styles.py
────────────
All custom CSS for the Supply Chain AI Copilot dashboard.
Injected once via inject_css() at app startup.
"""

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@700;800&display=swap');

/* ── Design tokens ── */
:root {
  --bg:      #07090f;
  --surface: #0d1117;
  --card:    #111827;
  --border:  #1f2937;
  --accent:  #4f46e5;
  --accent2: #0ea5e9;
  --success: #10b981;
  --warning: #f59e0b;
  --danger:  #ef4444;
  --text:    #f1f5f9;
  --muted:   #64748b;
  --radius:  12px;
}

/* ── Base ── */
html, body, [class*="css"] { font-family:'Inter',sans-serif; color:var(--text); }
.main { background:var(--bg); }
.block-container { padding:1.5rem 2.2rem 3rem; max-width:1380px; }
section[data-testid="stSidebar"] { background:var(--surface)!important; border-right:1px solid var(--border); }
#MainMenu { visibility:hidden; }
footer { visibility:hidden; }
/* keep header visible so sidebar collapse/expand arrow works */
header[data-testid="stHeader"] { background:transparent!important; }

/* always show the sidebar collapse/expand toggle button */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
  visibility: visible !important;
  display: flex !important;
  opacity: 1 !important;
}
.stDeployButton { display:none; }

/* ── Page header ── */
.page-header {
  padding: 1.6rem 0 1.2rem;
  margin-bottom: 1.4rem;
  border-bottom: 1px solid var(--border);
}
.page-title {
  font-family:'Plus Jakarta Sans',sans-serif;
  font-size: 1.65rem;
  font-weight: 800;
  color: var(--text);
  margin: 0 0 .25rem;
  letter-spacing: -.02em;
}
.page-sub {
  color: var(--muted);
  font-size: .88rem;
  margin: 0;
  font-weight: 400;
}

/* ── Metric grid ── */
.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:.9rem; margin-bottom:1.4rem; }
.metric-card {
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:1.1rem 1.3rem;
  position:relative;
  overflow:hidden;
  transition:border-color .15s;
}
.metric-card:hover { border-color:#374151; }
.metric-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:var(--accent); opacity:.7;
}
.metric-icon  { position:absolute; top:1rem; right:1.1rem; font-size:1.1rem; opacity:.35; }
.metric-label { font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; margin-bottom:.45rem; font-weight:500; }
.metric-value { font-size:1.85rem; font-weight:700; color:var(--text); line-height:1; }
.metric-sub   { font-size:.74rem; color:var(--muted); margin-top:.3rem; }

/* ── Section titles ── */
.section-title {
  font-size:.8rem;
  font-weight:600;
  color:var(--muted);
  text-transform:uppercase;
  letter-spacing:.08em;
  margin:1.4rem 0 .9rem;
  display:flex;
  align-items:center;
  gap:8px;
}
.section-title::after { content:''; flex:1; height:1px; background:var(--border); margin-left:4px; }

/* ── Chat ── */
.chat-container {
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:1.2rem 1.4rem;
  min-height:140px;
  max-height:480px;
  overflow-y:auto;
  margin-bottom:.9rem;
}
.chat-empty {
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  padding:2.5rem 1rem; color:var(--muted); text-align:center;
}
.chat-empty-icon { font-size:1.6rem; margin-bottom:.6rem; opacity:.5; }
.chat-empty-text { font-size:.88rem; font-weight:500; color:#475569; }
.chat-empty-hint { font-size:.78rem; color:#334155; margin-top:.3rem; }

.chat-bubble-user { display:flex; justify-content:flex-end; margin-bottom:.9rem; }
.chat-bubble-user-inner {
  background:var(--accent);
  border-radius:12px 12px 3px 12px;
  padding:.7rem 1.1rem;
  max-width:72%;
  font-size:.88rem;
  color:#fff;
  line-height:1.55;
}
.chat-bubble-ai { display:flex; justify-content:flex-start; margin-bottom:.9rem; gap:9px; }
.chat-avatar {
  width:28px; height:28px;
  background:var(--border);
  border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  font-size:.75rem; flex-shrink:0; margin-top:2px;
  border:1px solid #374151;
}
.chat-bubble-ai-inner {
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:3px 12px 12px 12px;
  padding:.75rem 1.1rem;
  max-width:80%;
  font-size:.88rem;
  color:var(--text);
  line-height:1.65;
}

/* ── Insight cards ── */
.insight-card {
  background:var(--card);
  border:1px solid var(--border);
  border-left:3px solid var(--accent);
  border-radius:var(--radius);
  padding:.85rem 1.2rem;
  margin-bottom:.5rem;
  font-size:.875rem;
  line-height:1.6;
  color:var(--text);
}
.insight-card.warning { border-left-color:var(--warning); }
.insight-card.success { border-left-color:var(--success); }
.insight-card.danger  { border-left-color:var(--danger); }

/* ── Summary box ── */
.summary-box {
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:1.2rem 1.5rem;
  margin-top:.9rem;
}
.summary-label {
  font-size:.7rem; text-transform:uppercase; letter-spacing:.08em;
  color:var(--muted); margin-bottom:.5rem; font-weight:600;
}
.summary-text { font-size:.9rem; color:var(--text); line-height:1.72; }

/* ── Chart card wrapper ── */
.chart-card {
  background:var(--card);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:.4rem;
  margin-bottom:.9rem;
}

/* ── Sidebar ── */
.sidebar-label { font-size:.68rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin-bottom:.3rem; font-weight:500; }
.sidebar-value { font-size:.9rem; font-weight:600; color:var(--text); }
.sidebar-stat-grid { display:grid; gap:.6rem; }
.stat-row { padding:.5rem 0; border-bottom:1px solid var(--border); }
.stat-row:last-child { border-bottom:none; }

/* ── Buttons ── */
.stButton>button {
  background:var(--accent)!important;
  color:#fff!important; border:none!important; border-radius:8px!important;
  padding:.5rem 1.2rem!important; font-family:'Inter',sans-serif!important;
  font-weight:500!important; font-size:.875rem!important;
  transition:opacity .15s!important;
  box-shadow:none!important;
}
.stButton>button:hover { opacity:.88!important; }

/* ── Inputs ── */
.stTextInput>div>div>input {
  background:var(--card)!important; border:1px solid var(--border)!important;
  border-radius:8px!important; color:var(--text)!important;
  font-family:'Inter',sans-serif!important; font-size:.9rem!important;
}
.stTextInput>div>div>input:focus { border-color:#374151!important; box-shadow:0 0 0 2px rgba(79,70,229,.15)!important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background:var(--card)!important; border-radius:8px!important;
  padding:3px!important; gap:2px!important; border:1px solid var(--border)!important;
}
.stTabs [data-baseweb="tab"] {
  background:transparent!important; color:var(--muted)!important;
  border-radius:6px!important; font-weight:500!important; font-size:.875rem!important;
}
.stTabs [aria-selected="true"] { background:var(--accent)!important; color:#fff!important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#374151; }
</style>
"""


def inject_css() -> None:
    """Inject all custom CSS into the Streamlit page. Call once at startup."""
    st.markdown(_CSS, unsafe_allow_html=True)
