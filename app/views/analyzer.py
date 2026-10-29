"""Single-stock deep analysis."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app import components as ui
from app import storage
from app.theme import BLUE, DOWN, GOLD, MUTED, UP
from jinni import indicators as ind
from jinni import events
from jinni import forecast as fc
from jinni import setups as jsetups
from jinni import signals
from jinni.universe import BENCHMARK, NIFTY50, display_name, normalize_symbol

st.title("Stock Analyzer")

options = list(NIFTY50) + ["Other…"]
default = st.session_state.get("analyze_ticker", "RELIANCE.NS")
c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
with c1:
    pick = st.selectbox("Stock", options, index=options.index(default) if default in options else len(options) - 1,
                        format_func=lambda t: t if t == "Other…" else f"{display_name(t)} · {NIFTY50.get(t, '')}")
    if pick == "Other…":
        raw = st.text_input("Any NSE/BSE symbol (e.g. IRCTC, TATAELXSI, 500325.BO)",
                            value="" if default in NIFTY50 else default)
        ticker = normalize_symbol(raw) if raw else None
    else:
        ticker = pick
with c2:
    period = st.selectbox("History", ["1y", "2y", "5y", "10y"], index=1)

if not ticker:
    st.info("Enter a symbol to analyse.")
    st.stop()
st.session_state["analyze_ticker"] = ticker

with st.spinner(f"Analysing {display_name(ticker)}…"):
    pdata = ui.prices((ticker, BENCHMARK), "10y" if period == "10y" else "5y")
if ticker not in pdata.close or pdata.close[ticker].dropna().shape[0] < 60:
    st.error(f"No price history found for **{ticker}**. Check the symbol (NSE symbols end in .NS, BSE in .BO).")
    st.stop()
ui.data_source_notice(pdata)

df_full = pdata.ohlcv(ticker)
bench = pdata.close[BENCHMARK] if BENCHMARK in pdata.close else None
scores_full = signals.factor_scores(df_full, bench)
n_view = {"1y": 252, "2y": 504, "5y": 1260, "10y": 2520}[period]
df = ind.add_all(df_full).iloc[-n_view:]
scores = scores_full.iloc[-n_view:]
last = scores.iloc[-1]
fund_score, fund_parts = signals.fundamental_score(ui.info(ticker))
composite = float(last["Composite"])
if fund_score is not None:
    composite = 0.8 * composite + 0.2 * fund_score
rating = signals.rating_from_score(composite)

price = float(df["Close"].iloc[-1])
chg = price / float(df["Close"].iloc[-2]) - 1
meta = ui.info(ticker)
name = meta.get("longName") or meta.get("shortName") or display_name(ticker)

st.markdown(
    f"### {name} &nbsp;{ui.rating_badge(rating)}"
    f"<br><span class='hint'>{ticker} · {meta.get('sector') or NIFTY50.get(ticker, '')} · "
    f"as of {df.index[-1]:%d %b %Y}</span>",
    unsafe_allow_html=True,
)
hi52 = df_full["Close"].iloc[-252:].max()
lo52 = df_full["Close"].iloc[-252:].min()
vol = float(df["VOL20"].iloc[-1])
ui.kpis([
    ("Last price", f"₹{price:,.2f}", ui.pct(chg, sign=True), ui.tone(chg)),
    ("JINNI score", f"{composite:.0f}/100", rating),
    ("52-week range", f"₹{lo52:,.0f}–{hi52:,.0f}", f"{ui.pct(price / hi52 - 1, 1)} from high"),
    ("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}", "Wilder"),
    ("Volatility (20d)", ui.pct(vol, 1), "annualised"),
    ("ATR (14)", f"₹{df['ATR'].iloc[-1]:,.2f}", ui.pct(df['ATR'].iloc[-1] / price, 2) + " of price"),
])

tab_chart, tab_odds, tab_factors, tab_plan, tab_fund, tab_evidence, tab_news = st.tabs(
    ["Chart", "Return odds", "Factors", "Trade plan", "Fundamentals", "Signal history", "News"])

# ------------------------------------------------------------------ chart
with tab_chart:
    overlays = st.multiselect("Overlays", ["SMA20", "SMA50", "SMA200", "EMA21", "Bollinger", "VWAP20"],
                              default=["SMA50", "SMA200"])
    lower = st.radio("Lower panel", ["Volume", "RSI", "MACD", "ADX"], horizontal=True, index=1)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 name="Price", increasing_line_color=UP, decreasing_line_color=DOWN,
                                 increasing_fillcolor=UP, decreasing_fillcolor=DOWN), 1, 1)
    palette = {"SMA20": "#A78BFA", "SMA50": BLUE, "SMA200": GOLD, "EMA21": "#2EC4B6", "VWAP20": "#E86AF0"}
    for o in overlays:
        if o == "Bollinger":
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], line=dict(color=MUTED, width=1), name="BB upper"), 1, 1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_lo"], line=dict(color=MUTED, width=1), name="BB lower",
                                     fill="tonexty", fillcolor="rgba(138,148,166,0.07)"), 1, 1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df[o], name=o, line=dict(color=palette[o], width=1.4)), 1, 1)
    if lower == "Volume":
        colors = np.where(df["Close"] >= df["Open"], UP, DOWN)
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume", opacity=0.7), 2, 1)
    elif lower == "RSI":
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color=GOLD, width=1.3)), 2, 1)
        for y in (30, 70):
            fig.add_hline(y=y, line=dict(color=MUTED, dash="dot", width=1), row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
    elif lower == "MACD":
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogram",
                             marker_color=np.where(df["MACD_hist"] >= 0, UP, DOWN)), 2, 1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color=BLUE, width=1.2)), 2, 1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal",
                                 line=dict(color=GOLD, width=1.2)), 2, 1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["ADX"], name="ADX", line=dict(color=GOLD, width=1.4)), 2, 1)
        fig.add_trace(go.Scatter(x=df.index, y=df["+DI"], name="+DI", line=dict(color=UP, width=1)), 2, 1)
        fig.add_trace(go.Scatter(x=df.index, y=df["-DI"], name="-DI", line=dict(color=DOWN, width=1)), 2, 1)
        fig.add_hline(y=25, line=dict(color=MUTED, dash="dot", width=1), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    ui.plot(fig, 560)

    ui.section("Technical events")
    for e in signals.detect_events(df_full):
        st.markdown(f"- {e}")
    lv = ind.pivot_levels(df_full)
    sr = ind.support_resistance(df_full["Close"])
    ui.kpis([(k, f"₹{v:,.2f}") for k, v in lv.items()] + [
        ("Swing support", f"₹{sr['support']:,.2f}" if sr["support"] else "—"),
        ("Swing resistance", f"₹{sr['resistance']:,.2f}" if sr["resistance"] else "—"),
    ])

# --------------------------------------------------------------- odds
with tab_odds:
    st.markdown("Chance of the price **touching** each gain within each window (no stop), averaged from 3,000 "
                "simulated paths and the stock's own history.")
    mat = fc.odds_matrix(df_full)
    fig = go.Figure(go.Heatmap(z=mat.values, x=mat.columns, y=mat.index, zmin=0, zmax=1,
                               colorscale=[[0, "#111823"], [0.5, "#3B5B99"], [1, UP]], showscale=False,
                               text=[[f"{v:.0%}" for v in r] for r in mat.values], texttemplate="%{text}"))
    fig.update_yaxes(autorange="reversed")
    ui.plot(fig, 300)
    o1, o2 = st.columns(2)
    with o1:
        q_t = st.select_slider("Goal", [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20], value=0.05,
                               format_func=lambda x: f"+{x:.1%}".replace(".0%", "%"))
    with o2:
        q_h = st.selectbox("Within", list(fc.HORIZONS), index=2)
    hz = fc.HORIZONS[q_h]
    stop_q = fc.auto_stop_pct(df_full, hz, q_t)
    fq = fc.forecast_stock(ticker, df_full, bench, hz, q_t, stop_q, None, n_paths=4000)
    ui.kpis([
        ("Target first", ui.pct(fq.p_target, 0), f"₹{fq.target_price:,.2f}", "up"),
        ("Stop first", ui.pct(fq.p_stop, 0), f"₹{fq.stop_price:,.2f} (−{stop_q:.1%} auto)", "down"),
        ("Expected return", ui.pct(fq.ev, 2, sign=True), "after costs", ui.tone(fq.ev)),
        ("Grade", fq.grade, fq.verdict),
    ])
    act_now = jsetups.active_setups(df_full)
    st.caption(("Active setups: " + ", ".join(act_now) + ". ") if act_now else "No chart setup active today. ")
    st.page_link("app/views/predictor.py", label="Compare against every stock in Return Finder",
                 icon=":material/target:")

# ---------------------------------------------------------------- factors
with tab_factors:
    fac = last[list(signals.WEIGHTS)]
    cats = list(fac.index) + ([ "Fundamentals"] if fund_score is not None else [])
    vals = list(fac.values) + ([fund_score] if fund_score is not None else [])
    fc1, fc2 = st.columns([1, 1], gap="medium")
    with fc1:
        fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], fill="toself",
                                        line=dict(color=GOLD), fillcolor="rgba(245,184,65,0.18)", name=ticker))
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], gridcolor="#1A2433",
                                                                              tickfont=dict(color=MUTED, size=9)),
                                     angularaxis=dict(gridcolor="#1A2433")), showlegend=False,
                          title="Factor profile (0–100)")
        ui.plot(fig, 360)
    with fc2:
        hist_fig = go.Figure()
        for k, col in zip(signals.WEIGHTS, ["#4C8DFF", "#F5B841", "#2EC4B6", "#A78BFA", "#94A3B8"]):
            hist_fig.add_trace(go.Scatter(x=scores.index, y=scores[k], name=k, line=dict(width=1, color=col),
                                          visible="legendonly"))
        hist_fig.add_trace(go.Scatter(x=scores.index, y=scores["Composite"], name="Composite",
                                      line=dict(width=2.2, color=GOLD)))
        for y, c in ((60, UP), (42, DOWN)):
            hist_fig.add_hline(y=y, line=dict(color=c, dash="dot", width=1))
        hist_fig.update_layout(title="Score over time", yaxis=dict(range=[0, 100]), hovermode="x unified")
        ui.plot(hist_fig, 360)
    weights_txt = ", ".join(f"{k} {int(w * 100)}%" for k, w in signals.WEIGHTS.items())
    st.caption(f"Technical weights: {weights_txt}. "
               + ("Final score = 80% technical + 20% fundamentals." if fund_score is not None
                  else "No fundamentals available, so the score is technical only."))

# --------------------------------------------------------------- trade plan
with tab_plan:
    settings = ui.settings()
    p1, p2, p3 = st.columns(3)
    with p1:
        default_side = 1 if rating in ("SELL", "STRONG SELL") else 0
        side = st.radio("Side", ["LONG", "SHORT"], index=default_side, horizontal=True)
    with p2:
        capital = st.number_input("Capital (₹)", min_value=10_000.0, value=float(settings["capital"]),
                                  step=50_000.0, format="%.0f")
    with p3:
        risk_pct = st.slider("Risk per trade (% of capital)", 0.25, 3.0, settings["risk_pct"] * 100, 0.25) / 100
    plan = signals.trade_plan(df_full, side, capital, risk_pct)
    rows = pd.DataFrame(plan.as_rows()).set_index("Level")
    st.dataframe(rows, width="stretch", column_config={
        "Price": st.column_config.NumberColumn(format="₹%.2f"),
        "Move %": st.column_config.NumberColumn(format="%+.2f%%"),
        "R multiple": st.column_config.NumberColumn(format="%.1fR"),
    })
    ui.kpis([
        ("Position size", f"{plan.qty:,} shares", ui.inr(plan.qty * plan.entry)),
        ("Capital at risk", ui.inr(plan.capital_at_risk), ui.pct(plan.capital_at_risk / capital, 2) + " of capital"),
        ("Risk / share", f"₹{plan.risk_per_share:,.2f}", ui.pct(plan.risk_per_share / plan.entry) + " stop distance"),
    ])
    if side == "SHORT":
        st.caption("Shorting delivery equity is not allowed in India; use stock futures/options (F&O names only) "
                   "or treat this as an exit plan.")
    st.caption("Stop = tighter of 2×ATR or just beyond the nearest swing level. Targets at 1.5R / 2.5R / 4R, "
               "with T1 pulled to the nearest resistance (support for shorts) when that comes first.")

# ------------------------------------------------------------- fundamentals
with tab_fund:
    meta = ui.info(ticker)
    if not meta:
        st.info("Fundamentals are unavailable for this symbol right now.")
    else:
        def g(k, f=lambda v: v):
            v = meta.get(k)
            try:
                return f(v) if v is not None else "—"
            except Exception:
                return "—"
        dy = meta.get("dividendYield")
        # Yahoo reports dividendYield in percent units (e.g. 0.45 = 0.45%) since 2024
        dy_txt = f"{dy:.2f}%" if isinstance(dy, (int, float)) else "—"
        ui.kpis([
            ("Market cap", g("marketCap", ui.inr)),
            ("P/E (TTM)", g("trailingPE", lambda v: f"{v:.1f}")),
            ("P/B", g("priceToBook", lambda v: f"{v:.2f}")),
            ("ROE", g("returnOnEquity", lambda v: ui.pct(v, 1))),
            ("Debt / Equity", g("debtToEquity", lambda v: f"{v / 100:.2f}")),
            ("Dividend yield", dy_txt),
            ("Profit margin", g("profitMargins", lambda v: ui.pct(v, 1))),
            ("Earnings growth", g("earningsGrowth", lambda v: ui.pct(v, 1, sign=True))),
        ])
        if fund_parts:
            ui.plot(ui.colored_bar(pd.Series(fund_parts) / 100 - 0.5, "Fundamental sub-scores (centred at 50)",
                                   fmt=".0%"), 260)
            st.caption("Bars show each sub-score relative to a neutral 50.")

# ----------------------------------------------------------------- evidence
with tab_evidence:
    st.markdown("How did this stock actually move after each rating in the past? "
                "Use this to judge whether the scoring rule has worked **for this stock**.")
    horizon = st.select_slider("Forward window (trading days)", [5, 10, 20, 60], value=20)
    ev = signals.score_history_stats(scores_full["Composite"], df_full["Close"], horizon)
    if ev.empty:
        st.info("Not enough history.")
    else:
        st.dataframe(ev, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.2f") for c in ev.columns if c != "Days"})
        ui.plot(ui.colored_bar(ev[f"Avg {horizon}d return %"] / 100, f"Average {horizon}-day forward return by rating",
                               fmt=".1%"), 280)
        st.caption("In-sample check of a rules-based score (no fitting). Overlapping windows, so day counts overstate "
                   "independent observations. See Strategy Lab for cost-adjusted backtests.")

# ------------------------------------------------------------------- news
with tab_news:
    for w in events.event_warnings(ticker, 21):
        st.warning(w, icon=":material/event:")
    news = events.headlines(ticker)
    if news.empty:
        st.caption("No headlines available right now.")
    for _, n in news.iterrows():
        tags = f" <span class='badge b-hold'>{n['Tags']}</span>" if n["Tags"] else ""
        when = n["Time"].strftime("%d %b %H:%M") if n["Time"] is not None else ""
        st.markdown(f"<span class='hint'>{when}</span> [{n['Headline']}]({n['Link']}){tags}", unsafe_allow_html=True)

# ------------------------------------------------------------------ actions
a1, a2 = st.columns(2)
with a1:
    if st.button("Add to watchlist", icon=":material/star:", width="stretch"):
        storage.add_watch(ticker)
        st.toast(f"{display_name(ticker)} added to watchlist")
with a2:
    if st.button("Save analysis", icon=":material/save:", width="stretch"):
        plan = signals.trade_plan(df_full, "LONG")
        storage.log_analysis({
            "ticker": ticker, "price": price, "rating": rating, "composite": composite,
            "trend": float(last["Trend"]), "momentum": float(last["Momentum"]), "timing": float(last["Timing"]),
            "volume": float(last["Volume"]), "risk": float(last["Risk"]), "fundamental": fund_score,
            "stop": plan.stop, "target1": plan.targets[0], "target2": plan.targets[1], "target3": plan.targets[2],
        })
        st.toast("Analysis saved")
ui.disclaimer()
