"""Home: opening screen + market pulse (indices, regime, breadth, sectors, movers)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import components as ui
from app.theme import DOWN, GOLD, MUTED, UP, logo_data_uri
from jinni import indicators as ind
from jinni.universe import BENCHMARK, INDICES, MACRO, NIFTY50, SECTOR_INDICES, display_name

st.markdown(
    f"""<div class="hero"><div class="brand"><img src="{logo_data_uri()}"/><span>JINNI</span></div>
    <h1>Indian market risk &amp; analytics terminal</h1>
    <p>Multi-factor stock ratings, portfolio VaR and stress tests, optimisation and
    walk-forward-validated models for NSE &amp; BSE in one place.</p></div>""",
    unsafe_allow_html=True,
)

with st.spinner("Loading market pulse…"):
    idx = ui.prices(tuple(list(INDICES) + list(MACRO)), "1y")
    sec = ui.prices(tuple(SECTOR_INDICES), "1y")
    stocks = ui.prices(tuple(NIFTY50), "1y")

ui.data_source_notice(idx)

# ---------------------------------------------------------------- tape
tape = []
for t in list(INDICES) + list(MACRO):
    if t in idx.close:
        s = idx.close[t].dropna()
        if len(s) > 1:
            tape.append((display_name(t), float(s.iloc[-1]), float(s.iloc[-1] / s.iloc[-2] - 1)))
for t in list(NIFTY50)[:20]:
    if t in stocks.close:
        s = stocks.close[t].dropna()
        if len(s) > 1:
            tape.append((display_name(t), float(s.iloc[-1]), float(s.iloc[-1] / s.iloc[-2] - 1)))
if tape:
    ui.ticker_tape(tape)

# ---------------------------------------------------------------- regime
nifty = idx.close[BENCHMARK].dropna() if BENCHMARK in idx.close else None
items = []
if nifty is not None and len(nifty) > 200:
    last, prev = nifty.iloc[-1], nifty.iloc[-2]
    s50, s200 = ind.sma(nifty, 50).iloc[-1], ind.sma(nifty, 200).iloc[-1]
    if last > s50 > s200:
        regime, rcls = "Bull trend", "up"
    elif last < s50 < s200:
        regime, rcls = "Bear trend", "down"
    else:
        regime, rcls = "Range / transition", "gold"
    items.append(("NIFTY 50", f"{last:,.0f}", ui.pct(last / prev - 1, sign=True), ui.tone(last - prev)))
    items.append(("Market regime", regime, f"50-DMA {s50:,.0f} · 200-DMA {s200:,.0f}", rcls))
    vol = ind.realized_vol(nifty, 20).iloc[-1]
    items.append(("NIFTY 20d vol", ui.pct(vol, 1), "annualised realised"))
if "^INDIAVIX" in idx.close:
    vix = idx.close["^INDIAVIX"].dropna()
    if len(vix) > 1:
        lvl = vix.iloc[-1]
        mood = "Calm" if lvl < 13 else "Normal" if lvl < 18 else "Elevated" if lvl < 24 else "Fear"
        items.append(("India VIX", f"{lvl:.2f}", f"{mood} · {ui.pct(lvl / vix.iloc[-2] - 1, sign=True)}",
                      "down" if lvl >= 18 else "up"))
c = stocks.close
if not c.empty and len(c) > 50:
    above50 = (c.iloc[-1] > c.rolling(50).mean().iloc[-1]).mean()
    above200 = (c.iloc[-1] > c.rolling(200, min_periods=150).mean().iloc[-1]).mean()
    adv = (c.iloc[-1] > c.iloc[-2]).sum()
    dec = (c.iloc[-1] < c.iloc[-2]).sum()
    items.append(("Breadth (>50-DMA)", ui.pct(above50, 0), f">200-DMA {ui.pct(above200, 0)}",
                  "up" if above50 >= 0.5 else "down"))
    items.append(("Advance / Decline", f"{adv} / {dec}", "NIFTY 50 today", "up" if adv >= dec else "down"))
ui.kpis(items)

# ---------------------------------------------------------------- chart
col1, col2 = st.columns([3, 2], gap="medium")
with col1:
    if nifty is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nifty.index, y=nifty, name="NIFTY 50", line=dict(color=GOLD, width=2),
                                 fill="tozeroy", fillcolor="rgba(245,184,65,0.06)"))
        fig.add_trace(go.Scatter(x=nifty.index, y=ind.sma(nifty, 50), name="50-DMA",
                                 line=dict(color="#4C8DFF", width=1.2)))
        fig.add_trace(go.Scatter(x=nifty.index, y=ind.sma(nifty, 200), name="200-DMA",
                                 line=dict(color=MUTED, width=1.2, dash="dot")))
        lo, hi = nifty.min(), nifty.max()
        fig.update_yaxes(range=[lo * 0.97, hi * 1.02])
        fig.update_layout(title="NIFTY 50 · 1 year", hovermode="x unified")
        ui.plot(fig, 340)

with col2:
    rows = []
    for t, name in SECTOR_INDICES.items():
        if t in sec.close:
            s = sec.close[t].dropna()
            if len(s) > 22:
                rows.append({"Sector": name, "1D": s.iloc[-1] / s.iloc[-2] - 1,
                             "1W": s.iloc[-1] / s.iloc[-6] - 1, "1M": s.iloc[-1] / s.iloc[-22] - 1,
                             "3M": s.iloc[-1] / s.iloc[max(-64, -len(s))] - 1})
    if rows:
        sdf = pd.DataFrame(rows).set_index("Sector").sort_values("1M", ascending=False)
        z = sdf.values * 100
        lim = max(1.0, float(np.nanmax(np.abs(z))))
        fig = go.Figure(go.Heatmap(
            z=z, x=sdf.columns, y=sdf.index, zmin=-lim, zmax=lim,
            colorscale=[[0, DOWN], [0.5, "#1A2433"], [1, UP]], showscale=False,
            text=[[f"{v:+.1f}%" for v in r] for r in z], texttemplate="%{text}",
            hovertemplate="%{y} %{x}: %{text}<extra></extra>",
        ))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="Sector rotation")
        ui.plot(fig, 340)

# ---------------------------------------------------------------- movers
if not c.empty and len(c) > 22:
    chg = pd.DataFrame({
        "Price": c.iloc[-1],
        "1D": c.iloc[-1] / c.iloc[-2] - 1,
        "1M": c.iloc[-1] / c.iloc[-22] - 1,
        "From 52w high": c.iloc[-1] / c.max() - 1,
    }).dropna(subset=["1D"])
    chg.index = [display_name(t) for t in chg.index]
    ui.section("NIFTY 50 movers")
    g, l_ = st.columns(2, gap="medium")
    fmt = {"Price": st.column_config.NumberColumn(format="₹%.2f"),
           "1D": st.column_config.NumberColumn(format="percent"),
           "1M": st.column_config.NumberColumn(format="percent"),
           "From 52w high": st.column_config.NumberColumn(format="percent")}
    with g:
        st.markdown("**Top gainers**")
        st.dataframe(chg.nlargest(5, "1D"), column_config=fmt, width="stretch")
    with l_:
        st.markdown("**Top losers**")
        st.dataframe(chg.nsmallest(5, "1D"), column_config=fmt, width="stretch")

# ---------------------------------------------------------------- nav
ui.section("Workspaces")
cards = [
    ("app/views/analyzer.py", "Stock Analyzer", ":material/candlestick_chart:",
     "Factor rating, chart, trade plan and measured signal history."),
    ("app/views/screener.py", "Screener", ":material/filter_list:",
     "Rank the NIFTY 50 or your own list by five factors."),
    ("app/views/portfolio.py", "Portfolio Risk", ":material/shield:",
     "VaR / CVaR, stress tests, beta, risk contribution, liquidity."),
    ("app/views/optimizer.py", "Optimizer", ":material/tune:",
     "Max-Sharpe, min-vol, risk-parity with sector caps and trade list."),
    ("app/views/lab.py", "Strategy Lab", ":material/science:",
     "Backtests with Indian costs, walk-forward ML validation."),
]
for row in (cards[:3], cards[3:]):
    cols = st.columns(3)
    for col, (target, title, icon, desc) in zip(cols, row):
        with col:
            st.page_link(target, label=title, icon=icon, width="stretch")
            st.caption(desc)

ui.disclaimer()
