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
from jinni import patterns as jpat
from jinni import risk
from jinni import seasonality as jseason
from jinni import sentiment
from jinni import signals
from jinni import ta
from app import charts as jcharts
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

tab_360, tab_chart, tab_pat, tab_ind, tab_odds, tab_season, tab_plan, tab_fund, tab_news = st.tabs(
    ["360° view", "Chart", "Patterns", "Indicators", "Return odds", "Seasonality", "Trade plan", "Fundamentals",
     "News & sentiment"])
tab_factors, tab_evidence = tab_360, tab_ind


@st.cache_data(ttl=900, show_spinner=False)
def _scan(tk: str, n: int, src: str):
    d = ui.prices((tk, BENCHMARK), "10y" if n > 1260 else "5y").ohlcv(tk)
    return jpat.scan(d), ta.summary(d)


scan, (ta_tbl, ta_verdict) = _scan(ticker, n_view, pdata.source)
news = events.headlines(ticker)
news_s = sentiment.score_news(news)

# ---------------------------------------------------------------- 360 view
with tab_360:
    active = scan["chart"] + scan["harmonic"] + scan["wave"] + scan["divergence"] + scan["trend"]
    recent_c = [p for p in scan["candles"] if p.end >= df_full.index[-3]]
    bull_n = sum(p.bias > 0 for p in active + recent_c)
    bear_n = sum(p.bias < 0 for p in active + recent_c)
    f1m = fc.forecast_stock(ticker, df_full, bench, 21, 0.05, fc.auto_stop_pct(df_full, 21, 0.05), None, n_paths=3000)
    cal = jseason.calendar_effects(df_full["Close"])
    this_month = pd.Timestamp.today().month_name()
    m_row = cal["Month"].loc[this_month] if this_month in cal["Month"].index else None
    parts = {
        "Technicals (87 indicators)": (ta_verdict["Overall"]["score"] + 1) * 50,
        "Factor score": float(last["Composite"]),
        "Patterns": 50 + 50 * (bull_n - bear_n) / max(bull_n + bear_n, 1),
        "Forecast (+5% in 1 month)": float(np.clip(50 + f1m.ev * 1000, 0, 100)),
    }
    if fund_score is not None:
        parts["Fundamentals"] = fund_score
    if np.isfinite(news_s["score"]):
        parts["News sentiment"] = 50 + 50 * news_s["score"]
    overall = float(np.mean(list(parts.values())))
    lab = signals.rating_from_score(overall)
    st.markdown(f"#### JINNI 360° score: **{overall:.0f}/100** {ui.rating_badge(lab)}", unsafe_allow_html=True)
    fig = go.Figure(go.Bar(x=list(parts.values()), y=list(parts.keys()), orientation="h",
                           marker_color=[UP if v >= 60 else DOWN if v < 42 else GOLD for v in parts.values()],
                           text=[f"{v:.0f}" for v in parts.values()], textposition="outside"))
    fig.add_vline(x=50, line=dict(color=MUTED, dash="dot"))
    fig.update_layout(xaxis=dict(range=[0, 105]), title="What each lens says (50 = neutral)")
    fig.update_yaxes(autorange="reversed")
    ui.plot(fig, 300)
    c_past, c_now, c_fut = st.columns(3, gap="medium")
    r1y = df_full["Close"].iloc[-1] / df_full["Close"].iloc[-min(253, len(df_full))] - 1
    with c_past:
        st.markdown("**Past**")
        ui.kpis([
            ("1-year return", ui.pct(r1y, 1, sign=True), "", ui.tone(r1y)),
            (f"{this_month} history", ui.pct(m_row["Avg %"] / 100 * 21, 1, sign=True) if m_row is not None else "—",
             f"up {m_row['Up %']:.0%} of days" if m_row is not None else ""),
            ("Max drawdown (5y)", ui.pct(risk.max_drawdown(df_full["Close"].pct_change())["max_drawdown"], 0), "",
             "down"),
        ])
    with c_now:
        st.markdown("**Present**")
        ui.kpis([
            ("Technical summary", ta_verdict["Overall"]["label"],
             f"{ta_verdict['Overall']['buy']} buy · {ta_verdict['Overall']['sell']} sell"),
            ("Patterns", f"{bull_n} bullish · {bear_n} bearish", "active chart + recent candles"),
            ("News tone", news_s["label"], f"{news_s['n']} headlines"),
        ])
    with c_fut:
        st.markdown("**Future**")
        ev_w = events.event_warnings(ticker, 21)
        ui.kpis([
            ("+5% within 1 month", ui.pct(f1m.p_target, 0), f"before a −{f1m.stop_pct:.1%} stop"),
            ("Median in 1 month", ui.pct(f1m.sim.terminal_pct[50], 1, sign=True),
             f"90%: {f1m.sim.terminal_pct[5]:+.0%} to {f1m.sim.terminal_pct[95]:+.0%}"),
            ("Events ahead", f"{len(ev_w)}" if ev_w else "None known", ev_w[0][:40] if ev_w else ""),
        ])
    st.caption("The 360° score averages independent lenses so no single signal dominates. Patterns and indicators "
               "describe the present; only the Forecast lens is a tested probability.")

# ------------------------------------------------------------------ chart
with tab_chart:
    by_cat = {c: [k for k, v in ta.CATALOG.items() if v.cat == c] for c in ta.CATEGORIES}
    ov_all = [k for k, v in ta.CATALOG.items() if v.kind == "overlay"]
    pn_all = [k for k, v in ta.CATALOG.items() if v.kind == "panel"]
    cc1, cc2 = st.columns(2)
    with cc1:
        overlays = st.multiselect(f"Overlays ({len(ov_all)} available)", ov_all, default=["EMA 20", "SMA 50", "SMA 200"])
    with cc2:
        panels = st.multiselect(f"Indicator panels ({len(pn_all)} available)", pn_all, default=["RSI 14", "MACD"],
                                max_selections=4)
    layers = st.multiselect("Auto-detected patterns to draw", jcharts.LAYERS,
                            default=["Candlestick patterns", "Chart patterns", "Trendlines", "Support/resistance zones"])
    z1, z2 = st.columns([3, 1], vertical_alignment="bottom")
    with z1:
        zoom = st.segmented_control("Zoom", ["3M", "6M", "1Y", "2Y", "5Y"], default="6M")
    with z2:
        heikin = st.toggle("Heikin-Ashi", value=False)
    view_n = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260}.get(zoom or "6M", 126)
    fig = jcharts.build(df_full, overlays, panels, layers, scan, heikin, view=min(view_n, len(df_full)))
    ui.plot(fig, 520 + 150 * len(panels))
    st.caption("▲/▼ markers = bullish/bearish candlestick patterns; shaded bands = support (green) / resistance (red) "
               "zones; ✓ = breakout confirmed by a close. Hover a marker for its name.")
    ui.section("Technical events")
    for e in signals.detect_events(df_full):
        st.markdown(f"- {e}")
    ui.section("Pivot levels (all methods)")
    st.dataframe(ta.pivots_all(df_full), width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="₹%.2f") for c in
                                ["Classic", "Fibonacci", "Camarilla", "Woodie", "DeMark"]})
    vp = ta.volume_profile(df_full)
    ui.kpis([("Volume POC (1y)", f"₹{vp['POC']:,.2f}", "most-traded price"),
             ("Value area high", f"₹{vp['VAH']:,.2f}", "70% of volume"),
             ("Value area low", f"₹{vp['VAL']:,.2f}")])

# --------------------------------------------------------------- patterns
with tab_pat:
    all_p = scan["chart"] + scan["harmonic"] + scan["wave"] + scan["divergence"] + scan["trend"] + scan["gap"]
    if not all_p and not scan["candles"]:
        st.info("No patterns detected right now.")
    rows = [{"Pattern": p.name, "Type": p.kind.title(), "Bias": p.bias_text, "Status": p.status,
             "Since": p.start.date() if hasattr(p.start, "date") else p.start, "Target ₹": p.target,
             "Invalidation ₹": p.invalidation, "What it means": p.note} for p in all_p]
    rows += [{"Pattern": p.name, "Type": "Candle", "Bias": p.bias_text, "Status": "printed",
              "Since": p.end.date(), "Target ₹": None, "Invalidation ₹": None, "What it means": ""}
             for p in scan["candles"]]
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
            "Target ₹": st.column_config.NumberColumn(format="₹%.2f"),
            "Invalidation ₹": st.column_config.NumberColumn(format="₹%.2f")})
    ui.section("Did these patterns work on this stock before?")
    hz_p = st.select_slider("Look-ahead (trading days)", [5, 10, 20], value=10, key="pat_h")
    cs = jpat.candle_stats(df_full, scan["candle_frame"], hz_p)
    cs = cs[cs["Times"] > 0].sort_values("Times", ascending=False)
    st.markdown(f"**Candlestick patterns**: average {hz_p}-day move after each, on {display_name(ticker)}'s own "
                f"history (stock's normal {hz_p}-day average: {cs.attrs['base_mean']:+.2%}).")
    st.dataframe(cs, width="stretch", column_config={
        f"Avg {hz_p}d": st.column_config.NumberColumn(format="percent"),
        "Up %": st.column_config.NumberColumn(format="percent"),
        "Worked %": st.column_config.NumberColumn(format="percent", help="Moved in the pattern's direction"),
        "Edge vs avg": st.column_config.NumberColumn(format="percent"),
        "Last seen": st.column_config.DateColumn()})

    @st.cache_data(ttl=3600, show_spinner=False)
    def _ph(tk, h, src):
        return jpat.pattern_history(ui.prices((tk, BENCHMARK), "5y").ohlcv(tk), h)

    with st.spinner("Replaying chart patterns through history…"):
        ph = _ph(ticker, max(hz_p, 10), pdata.source)
    st.markdown("**Chart & harmonic patterns**, replayed through history using only information available at the time:")
    st.dataframe(ph, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="percent") for c in ph.columns if c != "Times"})
    st.caption("Small counts are noise. A pattern that 'worked' 50% of the time on one stock has no edge there.")

# ------------------------------------------------------------- indicators
with tab_ind:
    v = ta_verdict
    g_cols = st.columns(3)
    for col, cat in zip(g_cols, ["Overall", "Moving averages", "Momentum"]):
        with col:
            sc = v[cat]["score"]
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=sc * 100, number=dict(suffix="", valueformat=".0f"),
                title=dict(text=f"{cat}<br><span style='font-size:12px'>{v[cat]['label']}</span>"),
                gauge=dict(axis=dict(range=[-100, 100], tickvals=[-100, -50, 0, 50, 100]),
                           bar=dict(color=UP if sc > 0.1 else DOWN if sc < -0.1 else GOLD),
                           bgcolor="#111823", borderwidth=0,
                           steps=[dict(range=[-100, -50], color="rgba(239,68,68,.25)"),
                                  dict(range=[50, 100], color="rgba(34,197,94,.25)")])))
            ui.plot(fig, 220)
    ui.kpis([(cat, v[cat]["label"], f"{v[cat]['buy']} buy · {v[cat]['neutral']} neutral · {v[cat]['sell']} sell")
             for cat in ta.CATEGORIES])
    cat_f = st.multiselect("Show categories", ta.CATEGORIES, default=ta.CATEGORIES)
    show = ta_tbl[ta_tbl["Category"].isin(cat_f)]
    st.dataframe(show.drop(columns=["Votes"]), width="stretch", hide_index=True,
                 height=min(700, 38 + 35 * len(show)),
                 column_config={"Value": st.column_config.NumberColumn(format="%.2f")})
    st.caption(f"{len(ta.CATALOG)} indicators. Measurement-only indicators (ATR, volatility, width…) show values "
               "but don't vote. Indicator signals are descriptive, and many are correlated: 20 moving averages "
               "agreeing is one opinion, not twenty.")

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

# ------------------------------------------------------------ seasonality
with tab_season:
    cal = jseason.calendar_effects(df_full["Close"])
    st.markdown(f"How {display_name(ticker)} has behaved by calendar (last {len(df_full) // 252} years). "
                "Rows marked significant survive a correction for testing many rows at once.")
    s1, s2 = st.columns(2, gap="medium")
    for col, key in ((s1, "Month"), (s2, "Weekday")):
        with col:
            t = cal[key].dropna(subset=["Avg %"])
            fig = go.Figure(go.Bar(x=t.index, y=t["Avg %"], marker_color=[UP if x >= 0 else DOWN for x in t["Avg %"]],
                                   text=[f"{u:.0%} up" for u in t["Up %"]], textposition="outside"))
            fig.update_layout(title=f"Average daily return by {key.lower()} (%)")
            ui.plot(fig, 300)
    st.dataframe(cal["Special days"], width="stretch", column_config={
        "Up %": st.column_config.NumberColumn(format="percent"), "p-value": st.column_config.NumberColumn(format="%.3f")})
    up_ev = jseason.upcoming(df_full["Close"])
    if not up_ev.empty:
        ui.section("Coming up")
        st.dataframe(up_ev, width="stretch", column_config={"Past up %": st.column_config.NumberColumn(format="percent")})
    st.page_link("app/views/events.py", label="Test any event or date rule in the Event Lab", icon=":material/event:")

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
    if news_s["n"]:
        ui.kpis([("News tone", news_s["label"], f"score {news_s['score']:+.2f} (−1 to +1)",
                  ui.tone(news_s["score"])),
                 ("Positive headlines", f"{news_s['pos']}", "", "up"),
                 ("Negative headlines", f"{news_s['neg']}", "", "down")])
    if news.empty:
        st.caption("No headlines available right now.")
    for k_, n in news.iterrows():
        sc_ = news_s["scores"].get(k_, 0) if news_s["n"] else 0
        dot = "🟢" if sc_ > 0.1 else "🔴" if sc_ < -0.1 else "⚪"
        tags = f" <span class='badge b-hold'>{n['Tags']}</span>" if n["Tags"] else ""
        when = n["Time"].strftime("%d %b %H:%M") if n["Time"] is not None else ""
        st.markdown(f"{dot} <span class='hint'>{when}</span> [{n['Headline']}]({n['Link']}){tags}",
                    unsafe_allow_html=True)
    st.caption("Tone is scored with a finance word list (negation-aware). It measures how headlines read, "
               "not whether they are right.")

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
