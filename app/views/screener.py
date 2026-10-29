"""Cross-sectional screener over the NIFTY 50 or a custom list."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import components as ui
from app import storage
from app.theme import DOWN, GOLD, UP
from jinni import indicators as ind
from jinni import signals
from jinni.universe import BENCHMARK, NIFTY50, display_name, normalize_symbol, sector_of

st.title("Screener")

with st.container(border=True):
    c1, c2 = st.columns([1, 2])
    with c1:
        universe = st.radio("Universe", ["NIFTY 50", "Watchlist", "Custom"], horizontal=True)
    with c2:
        custom = ""
        if universe == "Custom":
            custom = st.text_area("Symbols (comma or space separated)", "IRCTC, TATAELXSI, POLYCAB, DIXON, HAL, BEL",
                                  height=70)
if universe == "NIFTY 50":
    tickers = list(NIFTY50)
elif universe == "Watchlist":
    tickers = storage.watchlist()
    if not tickers:
        st.info("Your watchlist is empty. Add stocks from the Stock Analyzer.")
        st.stop()
else:
    tickers = [normalize_symbol(s) for s in custom.replace(",", " ").split() if s.strip()]
    if not tickers:
        st.stop()


@st.cache_data(ttl=900, show_spinner=False)
def screen(tickers: tuple[str, ...]) -> tuple[pd.DataFrame, str]:
    pdata = ui.prices(tuple(tickers) + (BENCHMARK,), "2y")
    bench = pdata.close.get(BENCHMARK)
    rows = []
    for t in tickers:
        if t not in pdata.close:
            continue
        df = pdata.ohlcv(t)
        if len(df) < 210:
            continue
        sc = signals.factor_scores(df, bench).iloc[-1]
        c = df["Close"]
        plan = signals.trade_plan(df, "LONG")
        rows.append({
            "Symbol": display_name(t), "ticker": t, "Sector": sector_of(t) if t in NIFTY50 else "—",
            "Price": c.iloc[-1], "1D": c.iloc[-1] / c.iloc[-2] - 1, "1M": c.iloc[-1] / c.iloc[-22] - 1,
            "6M": c.iloc[-1] / c.iloc[-127] - 1, "Score": sc["Composite"],
            "Rating": signals.rating_from_score(sc["Composite"]), "Trend": sc["Trend"], "Momentum": sc["Momentum"],
            "Timing": sc["Timing"], "Volume": sc["Volume"], "Risk": sc["Risk"],
            "RSI": ind.rsi(c).iloc[-1], "Vol 20d": ind.realized_vol(c).iloc[-1],
            "Stop": plan.stop, "T1": plan.targets[0], "Upside T1": plan.targets[0] / plan.entry - 1,
        })
    return pd.DataFrame(rows), pdata.source


with st.spinner(f"Scoring {len(tickers)} stocks…"):
    res, source = screen(tuple(tickers))
if source == "demo":
    st.markdown('<span class="badge b-demo">DEMO DATA</span>', unsafe_allow_html=True)
if res.empty:
    st.warning("No symbols had enough history (need ~10 months).")
    st.stop()

with st.expander("Filters", expanded=False, icon=":material/tune:"):
    f1, f2, f3 = st.columns(3)
    with f1:
        ratings = st.multiselect("Rating", ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"],
                                 default=["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"])
    with f2:
        sectors = st.multiselect("Sector", sorted(res["Sector"].unique()))
    with f3:
        min_score = st.slider("Min score", 0, 100, 0)
        max_rsi = st.slider("Max RSI", 30, 100, 100)
view = res[res["Rating"].isin(ratings) & (res["Score"] >= min_score) & (res["RSI"] <= max_rsi)]
if sectors:
    view = view[view["Sector"].isin(sectors)]
view = view.sort_values("Score", ascending=False)

counts = res["Rating"].value_counts()
ui.kpis([
    ("Screened", f"{len(res)}", f"{len(view)} match filters"),
    ("Buy / Strong buy", f"{counts.get('BUY', 0) + counts.get('STRONG BUY', 0)}", "", "up"),
    ("Hold", f"{counts.get('HOLD', 0)}", ""),
    ("Sell / Strong sell", f"{counts.get('SELL', 0) + counts.get('STRONG SELL', 0)}", "", "down"),
    ("Median score", f"{res['Score'].median():.0f}", "universe breadth"),
])

st.dataframe(
    view.drop(columns=["ticker"]).set_index("Symbol"),
    width="stretch", height=min(640, 38 + 35 * len(view)),
    column_config={
        "Price": st.column_config.NumberColumn(format="₹%.2f"),
        "1D": st.column_config.NumberColumn(format="percent"),
        "1M": st.column_config.NumberColumn(format="percent"),
        "6M": st.column_config.NumberColumn(format="percent"),
        "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
        **{k: st.column_config.NumberColumn(format="%.0f") for k in ["Trend", "Momentum", "Timing", "Volume", "Risk",
                                                                       "RSI"]},
        "Vol 20d": st.column_config.NumberColumn(format="percent"),
        "Stop": st.column_config.NumberColumn(format="₹%.2f"),
        "T1": st.column_config.NumberColumn(format="₹%.2f"),
        "Upside T1": st.column_config.NumberColumn(format="percent"),
    },
)
st.download_button("Download CSV", view.to_csv(index=False).encode(), "jinni_screen.csv", "text/csv",
                   icon=":material/download:")

ui.section("Momentum vs risk map")
fig = go.Figure(go.Scatter(
    x=res["Vol 20d"], y=res["6M"], mode="markers+text", text=res["Symbol"], textposition="top center",
    textfont=dict(size=9),
    marker=dict(size=np.clip(res["Score"] / 4, 6, 26), color=res["Score"], cmin=20, cmax=80,
                colorscale=[[0, DOWN], [0.5, GOLD], [1, UP]], line=dict(width=0),
                colorbar=dict(title="Score", thickness=10)),
    hovertemplate="%{text}<br>6M %{y:.1%}<br>Vol %{x:.1%}<extra></extra>",
))
fig.update_layout(xaxis_title="20-day volatility (ann.)", yaxis_title="6-month return")
fig.update_xaxes(tickformat=".0%")
fig.update_yaxes(tickformat=".0%")
ui.plot(fig, 460)

if universe == "NIFTY 50":
    ui.section("Average score by sector")
    by_sec = res.groupby("Sector")["Score"].mean().sort_values(ascending=False)
    ui.plot(ui.colored_bar((by_sec - 50) / 100, "Sector score vs neutral (50)", fmt=".0%"), 380)

pick = st.selectbox("Open in Stock Analyzer", view["ticker"], format_func=display_name)
if st.button("Analyze", icon=":material/insights:"):
    st.session_state["analyze_ticker"] = pick
    st.switch_page("app/views/analyzer.py")
ui.disclaimer()
