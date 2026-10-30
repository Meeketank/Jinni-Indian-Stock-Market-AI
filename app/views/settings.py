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
    st.markdown("**Risk rules** (every trade plan is checked against these)")
    r = dict(ui.rules())
    a, b, c = st.columns(3)
    with a:
        r["max_loss_per_trade"] = st.number_input("Max loss per trade (₹)", 100.0, 1e8, float(r["max_loss_per_trade"]),
                                                  1000.0, format="%.0f", help="Rupees lost if the stop fills. Decides size.")
        r["min_reward_risk"] = st.number_input("Min reward : risk", 0.5, 5.0, float(r["min_reward_risk"]), 0.25)
    with b:
        r["max_position_pct"] = st.slider("Max position (% of capital)", 1, 50, int(r["max_position_pct"] * 100)) / 100
        r["max_sector_pct"] = st.slider("Max sector (% of capital)", 5, 100, int(r["max_sector_pct"] * 100)) / 100
    with c:
        r["cash_buffer_pct"] = st.slider("Cash buffer (%)", 0, 50, int(r["cash_buffer_pct"] * 100)) / 100
        r["max_open_trades"] = int(st.number_input("Max open trades", 1, 100, int(r["max_open_trades"])))
        r["min_probability"] = st.slider("Min P(target first) (%)", 0, 90, int(r["min_probability"] * 100)) / 100
    r["require_stop"] = st.toggle("Require a stop-loss", value=bool(r["require_stop"]))
    if st.button("Save rules", type="primary", icon=":material/save:"):
        st.session_state["rules"] = r
        storage.save_setting("rules", r)
        st.toast("Rules saved")

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
