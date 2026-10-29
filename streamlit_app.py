"""JINNI — Indian market risk & analytics terminal.

Run:  streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="JINNI · Market Risk Terminal",
    page_icon=str(ROOT / "assets" / "icon.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)

from app.theme import inject_css, register_plotly_template, splash_once  # noqa: E402

inject_css()
register_plotly_template()
splash_once()
st.logo(str(ROOT / "assets" / "logo.svg"), size="large")

pages = {
    "Markets": [
        st.Page("app/views/home.py", title="Home", icon=":material/space_dashboard:", default=True),
        st.Page("app/views/analyzer.py", url_path="analyzer", title="Analyzer", icon=":material/candlestick_chart:"),
        st.Page("app/views/screener.py", url_path="screener", title="Screener", icon=":material/filter_list:"),
    ],
    "Portfolio": [
        st.Page("app/views/portfolio.py", url_path="risk", title="Risk", icon=":material/shield:"),
        st.Page("app/views/optimizer.py", url_path="optimizer", title="Optimizer", icon=":material/tune:"),
        st.Page("app/views/lab.py", url_path="lab", title="Lab", icon=":material/science:"),
        st.Page("app/views/settings.py", url_path="settings", title="Settings", icon=":material/settings:"),
    ],
}
st.navigation(pages, position="top").run()
