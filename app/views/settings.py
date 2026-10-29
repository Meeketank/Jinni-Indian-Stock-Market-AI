"""Settings, watchlist and saved analyses."""

import streamlit as st

from app import components as ui
from app import storage
from jinni.universe import display_name

st.title("Settings")

s = ui.settings()
with st.container(border=True):
    st.markdown("**Assumptions**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state["rf"] = st.number_input(
            "Risk-free rate (annual %)", 0.0, 15.0, s["rf"] * 100, 0.05,
            help="Used for Sharpe/Sortino/alpha. Default ≈ India 91-day T-bill / 10Y G-Sec.") / 100
    with c2:
        st.session_state["capital"] = st.number_input("Default capital (₹)", 10_000.0, 1e10, s["capital"], 50_000.0,
                                                      format="%.0f")
    with c3:
        st.session_state["risk_pct"] = st.number_input("Risk per trade (%)", 0.1, 5.0, s["risk_pct"] * 100, 0.1) / 100

with st.container(border=True):
    st.markdown("**Watchlist**")
    wl = storage.watchlist()
    if not wl:
        st.caption("Empty. Add stocks from the Stock Analyzer.")
    for t in wl:
        a, b = st.columns([4, 1], vertical_alignment="center")
        a.markdown(display_name(t))
        if b.button("Remove", key=f"rm_{t}", width="stretch"):
            storage.remove_watch(t)
            st.rerun()

with st.container(border=True):
    st.markdown("**Saved analyses**")
    log = storage.recent_analyses()
    if log.empty:
        st.caption("Nothing saved yet.")
    else:
        st.dataframe(log, width="stretch", hide_index=True)

with st.container(border=True):
    st.markdown("**Data**")
    if st.button("Clear cached market data", icon=":material/refresh:"):
        st.cache_data.clear()
        st.toast("Cache cleared. Data will reload.")
    st.caption("Prices are cached for 15 minutes to stay within Yahoo Finance rate limits. "
               "Set the environment variable JINNI_DEMO=1 to force demo data.")
