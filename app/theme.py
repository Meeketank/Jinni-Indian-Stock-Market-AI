"""Visual system: colours, global CSS (desktop + mobile), splash screen, Plotly template."""

from __future__ import annotations

import base64
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent / "assets"

BG = "#0A0E14"
PANEL = "#111823"
PANEL_2 = "#162030"
BORDER = "#222D3F"
TEXT = "#E6EAF2"
MUTED = "#8A94A6"
GOLD = "#F5B841"
UP = "#22C55E"
DOWN = "#EF4444"
BLUE = "#4C8DFF"
SERIES = ["#4C8DFF", "#F5B841", "#22C55E", "#E86AF0", "#2EC4B6", "#FF8A4C", "#A78BFA", "#94A3B8"]


def logo_data_uri() -> str:
    svg = (ASSETS / "logo.svg").read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(svg).decode()


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
:root {{
  --bg:{BG}; --panel:{PANEL}; --panel2:{PANEL_2}; --border:{BORDER}; --text:{TEXT};
  --muted:{MUTED}; --gold:{GOLD}; --up:{UP}; --down:{DOWN}; --blue:{BLUE};
}}
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', system-ui, sans-serif; }}
.stApp {{ background: radial-gradient(1200px 600px at 10% -10%, #14213a 0%, var(--bg) 55%) fixed; }}
.block-container {{ padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
h1 {{ font-weight: 800 !important; }}
code, .mono {{ font-family: 'JetBrains Mono', ui-monospace, monospace; }}
#MainMenu, footer {{ visibility: hidden; }}

/* ---------- cards ---------- */
.kpi-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 6px 0 14px; }}
.kpi {{ background: linear-gradient(180deg, var(--panel2), var(--panel)); border:1px solid var(--border);
        border-radius: 14px; padding: 12px 14px; min-width: 0; }}
.kpi .lbl {{ color: var(--muted); font-size: 0.74rem; text-transform: uppercase; letter-spacing: .06em; font-weight:600;
             white-space: nowrap; overflow:hidden; text-overflow: ellipsis; }}
.kpi .val {{ font-size: 1.35rem; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums;
             white-space: nowrap; overflow:hidden; text-overflow: ellipsis; }}
.kpi .sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 1px; font-variant-numeric: tabular-nums; }}
.up {{ color: var(--up) !important; }} .down {{ color: var(--down) !important; }} .gold {{ color: var(--gold) !important; }}

.panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 14px; padding: 14px 16px; margin-bottom: 12px; }}
.badge {{ display:inline-block; padding: 3px 10px; border-radius: 999px; font-weight: 700; font-size: .78rem; letter-spacing: .04em; }}
.b-sbuy {{ background: rgba(34,197,94,.18); color: #4ADE80; border:1px solid rgba(34,197,94,.45); }}
.b-buy  {{ background: rgba(34,197,94,.10); color: #86EFAC; border:1px solid rgba(34,197,94,.30); }}
.b-hold {{ background: rgba(245,184,65,.12); color: var(--gold); border:1px solid rgba(245,184,65,.35); }}
.b-sell {{ background: rgba(239,68,68,.10); color: #FCA5A5; border:1px solid rgba(239,68,68,.30); }}
.b-ssell{{ background: rgba(239,68,68,.18); color: #F87171; border:1px solid rgba(239,68,68,.45); }}
.b-demo {{ background: rgba(76,141,255,.14); color: #93B8FF; border:1px solid rgba(76,141,255,.4); }}

.section-title {{ font-size: .8rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; font-weight: 700; margin: 18px 0 8px; }}
.hint {{ color: var(--muted); font-size: .85rem; }}

/* ---------- ticker tape ---------- */
.tape {{ overflow:hidden; white-space:nowrap; border:1px solid var(--border); border-radius: 12px; background: var(--panel);
         padding: 8px 0; margin-bottom: 14px; -webkit-mask-image: linear-gradient(90deg, transparent, #000 6%, #000 94%, transparent); }}
.tape-inner {{ display:inline-block; animation: tape 40s linear infinite; }}
.tape span.item {{ margin: 0 22px; font-size: .88rem; font-variant-numeric: tabular-nums; }}
.tape b {{ color: var(--text); margin-right: 6px; }}
@keyframes tape {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}

/* ---------- hero ---------- */
.hero {{ border: 1px solid var(--border); border-radius: 20px; padding: 26px 26px 22px; margin-bottom: 16px;
         background: radial-gradient(500px 220px at 85% 0%, rgba(245,184,65,.16), transparent 70%),
                     linear-gradient(135deg, #111b2e 0%, #0c121c 100%); }}
.hero h1 {{ margin: 0; font-size: 2.1rem; line-height: 1.15; }}
.hero p {{ color: var(--muted); margin: 8px 0 0; font-size: 1rem; max-width: 720px; }}
.hero .brand {{ display:flex; align-items:center; gap: 12px; margin-bottom: 12px; }}
.hero .brand img {{ width: 40px; height: 40px; }}
.hero .brand span {{ font-weight: 800; letter-spacing: .28em; color: var(--gold); font-size: .85rem; }}

.nav-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(210px,1fr)); gap: 10px; }}
.nav-card {{ background: var(--panel); border:1px solid var(--border); border-radius: 14px; padding: 14px; }}
.nav-card .t {{ font-weight: 700; margin-bottom: 4px; }}
div[data-testid="stPageLink"] a {{ border:1px solid var(--border); border-radius: 12px; background: var(--panel); padding: 10px 12px; }} .nav-card .d {{ color: var(--muted); font-size: .85rem; }}

/* ---------- splash ---------- */
#jinni-splash {{ position: fixed; inset: 0; z-index: 999999; display:flex; flex-direction:column; align-items:center;
                 justify-content:center; background: radial-gradient(600px 400px at 50% 40%, #16233b, {BG});
                 animation: splash-out .6s ease 2.1s forwards; }}
#jinni-splash img {{ width: 92px; height: 92px; animation: pop 1s cubic-bezier(.2,.9,.3,1.3) both; }}
#jinni-splash .name {{ margin-top: 18px; font-weight: 800; font-size: 2rem; letter-spacing: .4em; color: {TEXT};
                       animation: rise .8s ease .35s both; padding-left: .4em; }}
#jinni-splash .tag {{ margin-top: 6px; color: {MUTED}; font-size: .9rem; letter-spacing: .12em; text-transform: uppercase;
                      animation: rise .8s ease .55s both; text-align:center; padding: 0 16px; }}
#jinni-splash .bar {{ margin-top: 26px; width: 180px; height: 3px; border-radius: 3px; background: {BORDER}; overflow:hidden; }}
#jinni-splash .bar i {{ display:block; height:100%; width: 0; background: linear-gradient(90deg, {GOLD}, #FFD978);
                        animation: load 1.8s ease .2s forwards; }}
@keyframes pop {{ from {{ transform: scale(.4); opacity: 0; }} to {{ transform: scale(1); opacity: 1; }} }}
@keyframes rise {{ from {{ transform: translateY(12px); opacity: 0; }} to {{ transform: none; opacity: 1; }} }}
@keyframes load {{ to {{ width: 100%; }} }}
@keyframes splash-out {{ to {{ opacity: 0; visibility: hidden; pointer-events: none; }} }}

/* ---------- widgets ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; overflow-x: auto; scrollbar-width: none; }}
.stTabs [data-baseweb="tab"] {{ padding: 8px 12px; white-space: nowrap; }}
div[data-testid="stMetric"] {{ background: var(--panel); border:1px solid var(--border); border-radius: 12px; padding: 10px 12px; }}
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{ border-radius: 10px; font-weight: 600; min-height: 42px; }}
div[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; }}
.disclaimer {{ color: var(--muted); font-size: .75rem; border-top: 1px solid var(--border); padding-top: 10px; margin-top: 28px; }}

/* ---------- mobile ---------- */
@media (max-width: 640px) {{
  .block-container {{ padding: 3.6rem .75rem 2rem; }}
  .hero {{ padding: 18px 16px; border-radius: 16px; }}
  .hero h1 {{ font-size: 1.5rem; }}
  .hero p {{ font-size: .9rem; }}
  .kpi-grid {{ grid-template-columns: repeat(2, minmax(0,1fr)); gap: 8px; }}
  .kpi {{ padding: 10px 11px; border-radius: 12px; }}
  .kpi .val {{ font-size: 1.08rem; }}
  .kpi .lbl {{ font-size: .66rem; }}
  .nav-grid {{ grid-template-columns: 1fr 1fr; gap: 8px; }}
  .nav-card {{ padding: 11px; }} .nav-card .d {{ font-size: .76rem; }}
  h2 {{ font-size: 1.3rem !important; }} h3 {{ font-size: 1.1rem !important; }}
  .tape span.item {{ margin: 0 14px; font-size: .8rem; }}
  .stTabs [data-baseweb="tab"] {{ padding: 6px 9px; font-size: .85rem; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def splash_once() -> None:
    """Animated opening screen, shown once per browser session, then fades away."""
    if st.session_state.get("_splash_shown"):
        return
    st.session_state["_splash_shown"] = True
    st.markdown(
        f"""<div id="jinni-splash"><img src="{logo_data_uri()}" alt="JINNI logo"/>
        <div class="name">JINNI</div>
        <div class="tag">Indian market risk &amp; analytics terminal</div>
        <div class="bar"><i></i></div></div>""",
        unsafe_allow_html=True,
    )


def register_plotly_template() -> None:
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=12),
        colorway=SERIES,
        xaxis=dict(gridcolor="#1A2433", zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor="#1A2433", zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED)),
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11)),
        margin=dict(l=8, r=8, t=40, b=8),
        hoverlabel=dict(bgcolor=PANEL_2, bordercolor=BORDER, font=dict(color=TEXT)),
        title=dict(font=dict(size=14, color=TEXT), x=0, xanchor="left"),
    )
    pio.templates["jinni"] = t
    pio.templates.default = "jinni"
