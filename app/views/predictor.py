"""Return Finder: which stocks can deliver +X% within N days, with honest odds."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import components as ui
from app import storage
from app.theme import BLUE, DOWN, GOLD, MUTED, UP
from jinni import events, forecast as fc, riskcheck as rc, risk, setups
from jinni.universe import BENCHMARK, NIFTY50, display_name, normalize_symbol, sector_of

st.title("Return Finder")
st.markdown('<span class="hint">Pick a goal. JINNI estimates, for every stock, the odds of reaching it '
            "before the stop-loss, using three independent engines, and tells you plainly when it's a coin flip."
            "</span>", unsafe_allow_html=True)

# ------------------------------------------------------------------ goal
with st.container(border=True):
    g1, g2 = st.columns(2)
    with g1:
        tgt_choice = st.segmented_control("Target gain", ["2%", "5%", "10%", "15%", "Custom"], default="5%")
        if tgt_choice == "Custom" or tgt_choice is None:
            target = st.number_input("Target %", 0.5, 100.0, 7.0, 0.5) / 100
        else:
            target = float(tgt_choice.rstrip("%")) / 100
        hz_choice = st.segmented_control("Within", list(fc.HORIZONS) + ["Custom"], default="1 month")
        if hz_choice == "Custom" or hz_choice is None:
            horizon = int(st.number_input("Trading days", 2, 250, 15, 1))
        else:
            horizon = fc.HORIZONS[hz_choice]
    with g2:
        stop_mode = st.segmented_control("Stop-loss", ["Auto", "Fixed %", "None"], default="Auto",
                                         help="Auto: volatility-based stop (≥1×ATR), no wider than target ÷ 1.5. "
                                              "None: just the odds of touching the target.")
        stop_fixed = None
        if stop_mode == "Fixed %":
            stop_fixed = st.number_input("Stop %", 0.5, 50.0, 3.0, 0.5) / 100
        universe = st.selectbox("Search in", ["NIFTY 50", "My holdings", "Watchlist", "Custom list"])
        custom = ""
        if universe == "Custom list":
            custom = st.text_input("Symbols", "IRCTC, HAL, BEL, POLYCAB, DIXON, TATAELXSI, TRENT, ZYDUSLIFE")
        use_ml = st.toggle("Include ML engine (slower, more thorough)", value=True)
    go_btn = st.button("Find opportunities", type="primary", icon=":material/search:", width="stretch")

if universe == "NIFTY 50":
    tickers = list(NIFTY50)
elif universe == "My holdings":
    tickers = list(st.session_state.get("holdings", storage.load_holdings())["ticker"])
elif universe == "Watchlist":
    tickers = storage.watchlist()
else:
    tickers = [normalize_symbol(x) for x in custom.replace(",", " ").split() if x.strip()]
stop_mode = stop_mode or "Auto"


@st.cache_data(ttl=900, show_spinner=False)
def run_finder(tickers: tuple, horizon: int, target: float, stop_mode: str, stop_fixed, use_ml: bool):
    pdata = ui.prices(tuple(tickers) + (BENCHMARK,), "5y")
    bench = pdata.close.get(BENCHMARK)
    dfs = {t: pdata.ohlcv(t) for t in tickers if t in pdata.close and pdata.close[t].notna().sum() > 300}
    stops = {}
    for t, df in dfs.items():
        stops[t] = (fc.auto_stop_pct(df, horizon, target) if stop_mode == "Auto"
                    else stop_fixed if stop_mode == "Fixed %" else None)
    ml = None
    ml_err = None
    if use_ml and len(dfs) >= 3:
        try:
            ml = fc.ml_hit_model(dfs, bench, horizon, target, stops if stop_mode == "Auto" else stop_fixed)
        except ValueError as e:
            ml_err = str(e)
    out = {}
    for t, df in dfs.items():
        try:
            out[t] = fc.forecast_stock(t, df, bench, horizon, target, stops[t], ml, n_paths=4000)
        except ValueError:
            continue
    act = {t: setups.active_setups(df) for t, df in dfs.items()}
    base = np.nanmean([f.hist["all"]["p_target"] for f in out.values()]) if out else np.nan
    return out, act, ml, ml_err, base, pdata.source


key = (tuple(tickers), horizon, target, stop_mode, stop_fixed, use_ml)
if go_btn:
    if not tickers:
        st.warning("No stocks to search. Add some to your list first.")
        st.stop()
    with st.spinner(f"Running {len(tickers)} stocks through 3 engines…"):
        st.session_state["finder"] = (key, run_finder(*key))

if "finder" not in st.session_state:
    ui.section("How it works")
    st.markdown(
        "1. **Time Machine**: replays the next days 4,000 times from the stock's own daily bars, "
        "rescaled to today's volatility. The stop is checked before the target, and gaps fill at the open.\n"
        "2. **History replay**: the same plan tested on every past day, and on days that looked like today.\n"
        "3. **ML model**: trained across all stocks, validated walk-forward, calibrated. Its vote counts only as much "
        "as its measured skill.\n\n"
        "Then a **verdict**: a +X% target with a −S% stop needs to come first in S÷(X+S) of decided cases just to "
        "break even. JINNI compares the odds to that line, after costs."
    )
    ui.disclaimer()
    st.stop()

(k_tickers, horizon, target, stop_mode, stop_fixed, use_ml), (res, act, ml, ml_err, base, source) = \
    st.session_state["finder"]
if source == "demo":
    st.markdown('<span class="badge b-demo">DEMO DATA</span> <span class="hint">Synthetic prices: odds are '
                "illustrative.</span>", unsafe_allow_html=True)
if not res:
    st.error("No stock had enough history (about 15 months needed).")
    st.stop()

rows = []
for t, f in res.items():
    rows.append({
        "ticker": t, "Stock": display_name(t), "Grade": f.grade, "P(target first)": f.p_target,
        "P(stop first)": f.p_stop, "P(touch target)": f.p_touch, "Exp. return": f.ev,
        "Price": f.price, "Target ₹": f.target_price, "Stop ₹": f.stop_price,
        "Stop %": f.stop_pct, "Typical days": f.days_to_target, "Verdict": f.verdict,
        "Setups": ", ".join(act.get(t, [])), "Sector": sector_of(t),
    })
tbl = pd.DataFrame(rows)
gc = tbl["Grade"].value_counts()
hz_label = next((k for k, v in fc.HORIZONS.items() if v == horizon), f"{horizon} days")
stop_txt = "auto stop" if stop_mode == "Auto" else f"−{stop_fixed:.1%} stop" if stop_fixed else "no stop"
st.markdown(f"#### Goal: **+{target:.0%} within {hz_label}** · {stop_txt}")
best = tbl.loc[tbl["P(target first)"].idxmax()]
ui.kpis([
    ("Stocks scanned", f"{len(tbl)}", f"{len(k_tickers) - len(tbl)} skipped" if len(k_tickers) > len(tbl) else ""),
    ("Grade A / B", f"{gc.get('A', 0)} / {gc.get('B', 0)}", "positive expected value", "up"),
    ("Grade C / D", f"{gc.get('C', 0)} / {gc.get('D', 0)}", "break-even or worse", "down"),
    ("Best odds", ui.pct(best["P(target first)"], 0), best["Stock"]),
    ("Typical stock", ui.pct(base, 0), "historical hit rate for this goal"),
    ("ML engine", f"AUC {ml.auc:.2f}" if ml else "off",
     (f"weight {ml.weight:.0%}" if ml.weight > 0 else "no skill → 0 weight") if ml else (ml_err or "")),
])

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    sort_by = st.selectbox("Sort by", ["Exp. return", "P(target first)", "P(touch target)"])
with c2:
    grades = st.multiselect("Grades", ["A", "B", "C", "D"], default=["A", "B", "C", "D"])
with c3:
    only_setups = st.toggle("Only stocks with an active setup", value=False)
view = tbl[tbl["Grade"].isin(grades)]
if only_setups:
    view = view[view["Setups"] != ""]
view = view.sort_values(sort_by, ascending=False)
view.insert(0, "Rank", range(1, len(view) + 1))
st.dataframe(
    view.drop(columns=["ticker", "Sector"]).set_index("Rank"), width="stretch",
    height=min(640, 38 + 35 * len(view)),
    column_config={
        "P(target first)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
        "P(stop first)": st.column_config.NumberColumn(format="percent"),
        "P(touch target)": st.column_config.NumberColumn(format="percent",
                                                         help="Touches the target at any point, ignoring the stop"),
        "Exp. return": st.column_config.NumberColumn(format="percent", help="Per trade, after ~0.3% costs"),
        "Price": st.column_config.NumberColumn(format="₹%.2f"),
        "Target ₹": st.column_config.NumberColumn(format="₹%.2f"),
        "Stop ₹": st.column_config.NumberColumn(format="₹%.2f"),
        "Stop %": st.column_config.NumberColumn(format="percent"),
        "Typical days": st.column_config.NumberColumn(format="%.0f", help="Median days to target when it was hit"),
    },
)
b1, b2 = st.columns(2)
with b1:
    st.download_button("Download CSV", view.to_csv(index=False).encode(), "jinni_return_finder.csv", "text/csv",
                       icon=":material/download:", width="stretch")
with b2:
    if st.button("Track top 5 (A/B) in Journal", icon=":material/bookmark_add:", width="stretch"):
        n = 0
        for _, r in view[view["Grade"].isin(["A", "B"])].head(5).iterrows():
            f = res[r["ticker"]]
            n += storage.track_prediction({
                "entry_date": f.as_of, "ticker": f.ticker,
                "entry": f.price, "target_pct": f.target_pct, "stop_pct": f.stop_pct, "target_price": f.target_price,
                "stop_price": f.stop_price, "horizon": f.horizon, "p_target": f.p_target, "p_stop": f.p_stop,
                "grade": f.grade, "verdict": f.verdict, "qty": 0, "source": "finder-auto",
            })
        st.toast(f"Tracking {n} new predictions" if n else "No new A/B picks to track")

# ------------------------------------------------------------------ detail
ui.section("Deep dive & trade plan")
pick = st.selectbox("Stock", view["ticker"], format_func=lambda t: f"{display_name(t)} · grade {res[t].grade}")
f = res[pick]
pdata = ui.prices((pick, BENCHMARK), "5y")
df = pdata.ohlcv(pick)

st.markdown(f"### {display_name(pick)} &nbsp;<span class='badge {'b-sbuy' if f.grade == 'A' else 'b-buy' if f.grade == 'B' else 'b-hold' if f.grade == 'C' else 'b-ssell'}'>GRADE {f.grade}</span>"
            f"<br><span class='hint'>{f.verdict}</span>", unsafe_allow_html=True)
items = [
    ("Target first", ui.pct(f.p_target, 1), f"₹{f.target_price:,.2f} (+{f.target_pct:.1%})", "up"),
    ("Stop first", ui.pct(f.p_stop, 1), f"₹{f.stop_price:,.2f} (−{f.stop_pct:.1%})" if f.stop_price else "no stop",
     "down"),
    ("Touches target", ui.pct(f.p_touch, 1), "at any point"),
    ("Expected return", ui.pct(f.ev, 2, sign=True), "per trade, after costs", ui.tone(f.ev)),
]
if f.stop_pct:
    items.append(("Win share needed", ui.pct(f.breakeven, 0), f"got {ui.pct(f.win_share, 0)} of decided cases",
                  "up" if f.win_share > f.breakeven else "down"))
items.append(("Typical time", f"{f.days_to_target:.0f} days" if np.isfinite(f.days_to_target) else "—",
              "median when target hit"))
ui.kpis(items)
for n in f.notes:
    st.caption("• " + n)

ev_warn = events.event_warnings(pick, horizon)
for w in ev_warn:
    st.warning(w, icon=":material/event:")

tabs = st.tabs(["Price cone", "Engines", "History", "Odds table", "Trade plan", "News"])
with tabs[0]:
    fan = f.sim.fan
    last_d = df.index[-1]
    fut = pd.bdate_range(last_d, periods=len(fan) + 1)[1:]
    hist = df["Close"].iloc[-max(30, horizon * 2):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Price", line=dict(color=MUTED, width=1.5)))
    fig.add_trace(go.Scatter(x=fut, y=fan["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut, y=fan["p5"], fill="tonexty", fillcolor="rgba(76,141,255,0.12)",
                             line=dict(width=0), name="90% range"))
    fig.add_trace(go.Scatter(x=fut, y=fan["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut, y=fan["p25"], fill="tonexty", fillcolor="rgba(76,141,255,0.25)",
                             line=dict(width=0), name="50% range"))
    fig.add_trace(go.Scatter(x=fut, y=fan["p50"], name="Median path", line=dict(color=BLUE, width=2)))
    fig.add_hline(y=f.target_price, line=dict(color=UP, dash="dash"), annotation_text=f"Target ₹{f.target_price:,.0f}",
                  annotation_font_color=UP)
    if f.stop_price:
        fig.add_hline(y=f.stop_price, line=dict(color=DOWN, dash="dash"), annotation_text=f"Stop ₹{f.stop_price:,.0f}",
                      annotation_position="bottom right", annotation_font_color=DOWN)
    fig.update_layout(title=f"Where the price could be over the next {horizon} trading days", hovermode="x unified")
    ui.plot(fig, 420)
    tp = f.sim.terminal_pct
    st.caption(f"{f.sim.n_paths:,} simulated paths (seed {f.sim.seed}, repeatable). Return at day {horizon}: median "
               f"{tp[50]:+.1%}, 90% of paths between {tp[5]:+.1%} and {tp[95]:+.1%}. "
               "A simulation from the stock's own past, not a promise.")

with tabs[1]:
    eng = pd.Series(f.engines)
    fig = go.Figure(go.Bar(x=eng.values, y=eng.index, orientation="h",
                           marker_color=[GOLD, BLUE, "#E86AF0"][:len(eng)],
                           text=[f"{v:.0%}" for v in eng.values], textposition="outside"))
    fig.add_vline(x=f.p_target, line=dict(color=UP, dash="dot"), annotation_text=f"Blend {f.p_target:.0%}")
    fig.update_layout(title="P(target first) by engine", xaxis=dict(range=[0, 1], tickformat=".0%"))
    ui.plot(fig, 260)
    if ml:
        st.markdown(f"**ML engine check** (walk-forward, {ml.n_oos:,} out-of-sample predictions): AUC {ml.auc:.3f}, "
                    f"Brier {ml.brier:.3f}, base rate {ml.base_rate:.1%}. Blend weight **{ml.weight:.0%}** "
                    "(zero at AUC 0.52, full at 0.60).")
        rel = ml.reliability.rename(index=lambda i: f"Q{int(i) + 1}")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=rel.index, y=rel["Actual"], name="Actual hit rate", marker_color=BLUE))
        fig.add_trace(go.Scatter(x=rel.index, y=rel["Predicted"], name="Predicted", mode="lines+markers",
                                 line=dict(color=GOLD)))
        fig.update_layout(title="Calibration: predicted vs actual (by prediction quintile)", yaxis_tickformat=".0%")
        ui.plot(fig, 280)

with tabs[2]:
    h = f.hist
    cmp_ = pd.DataFrame({"All past days": h["all"], "Days like today": h["similar"], "Used (shrunk)": h["shrunk"]}).T
    st.dataframe(cmp_[["n", "p_target", "p_stop", "p_neither", "avg_ret"]], width="stretch", column_config={
        "n": st.column_config.NumberColumn("Days", format="%d"),
        "p_target": st.column_config.NumberColumn("Target first", format="percent"),
        "p_stop": st.column_config.NumberColumn("Stop first", format="percent"),
        "p_neither": st.column_config.NumberColumn("Neither", format="percent"),
        "avg_ret": st.column_config.NumberColumn("Avg return", format="percent"),
    })
    st.caption("“Days like today” = same side of the 200-DMA and a JINNI score within ±10. Small samples are "
               "shrunk toward all days. Windows overlap, so the independent sample is smaller than the day count.")
    sh = fc.setup_history(df, setups.setup_frame(df), horizon, target, f.stop_pct)
    active = act.get(pick, [])
    st.markdown("**Every past time each setup fired** " + (f"(active now: {', '.join(active)})" if active else
                                                           "(no setup active today)"))
    st.dataframe(sh, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="percent") for c in ["Target first", "Stop first", "Neither",
                                                                      "Avg return"]})

with tabs[3]:
    with st.spinner("Computing odds grid…"):
        mat = fc.odds_matrix(df)
    fig = go.Figure(go.Heatmap(z=mat.values, x=mat.columns, y=mat.index, zmin=0, zmax=1,
                               colorscale=[[0, "#111823"], [0.5, "#3B5B99"], [1, UP]], showscale=False,
                               text=[[f"{v:.0%}" for v in r] for r in mat.values], texttemplate="%{text}",
                               hovertemplate="+%{x} within %{y}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=f"{display_name(pick)}: chance of touching each gain (no stop)")
    ui.plot(fig, 300)
    st.caption("Touch probability: the high reaches the level at least once in the window. "
               "Average of the Time Machine and history.")

with tabs[4]:
    s = ui.settings()
    rules = ui.rules()
    hold = st.session_state.get("holdings", storage.load_holdings())
    hp = ui.prices(tuple(hold["ticker"]) + (pick,), "2y")
    last_px = hp.close.ffill().iloc[-1]
    hv = (hold.set_index("ticker")["quantity"] * last_px.reindex(hold["ticker"]).values).dropna()
    capital = st.number_input("Trading capital (₹)", 10_000.0, 1e10, float(s["capital"]), 50_000.0, format="%.0f",
                              help="Cash available for new trades. Change the default in Settings.")
    cash = capital
    if not f.stop_price:
        st.info("Pick a stop-loss (Auto or Fixed) to size a trade.")
    else:
        sz = rc.size_position(f.price, f.stop_price, capital, cash, rules)
        lim = pd.DataFrame({"Max shares": sz.limits}).assign(Decides=lambda d: d.index == sz.deciding_limit)
        st.dataframe(lim, width="stretch")
        if sz.refused:
            st.error(sz.refused)
        else:
            st.markdown(f"**{sz.qty:,} shares** (₹{sz.qty * f.price:,.0f}), decided by **{sz.deciding_limit.lower()}**. "
                        f"Loss at stop ₹{sz.qty * (f.price - f.stop_price):,.0f}; gain at target "
                        f"₹{sz.qty * (f.target_price - f.price):,.0f}.")
        pl = storage.predictions()
        pl = pl[pl["source"] == "plan"] if not pl.empty else pl
        open_n = int(sum(pd.Timestamp(r.entry_date) + pd.Timedelta(days=int(r.horizon * 1.4) + 1) >= pd.Timestamp.today()
                         for r in pl.itertuples())) if not pl.empty else 0
        sec_vals = pd.Series({t: v for t, v in hv.items()}).groupby(lambda t: sector_of(t)).sum().to_dict()
        gres = rc.gate(f.price, f.stop_price, f.target_price, sz.qty, sector_of(pick), f.p_target, capital, cash,
                       sec_vals, open_n, rules, ev_warn)
        icon = {"PASS": "✅", "WARN": "⚠️", "BLOCK": "⛔"}
        st.markdown(f"#### Risk check: {icon[gres.status]} {gres.status}")
        for stt, rule, detail in gres.checks:
            st.markdown(f"{icon[stt]} **{rule}**: {detail}")
        if sz.qty > 0:
            rets = risk.returns(hp.close).iloc[-504:]
            imp = rc.portfolio_impact(rets, hv, pick, sz.qty * f.price)
            imp_df = pd.DataFrame(imp).T
            st.markdown("**Portfolio impact** (your holdings + this trade)")
            st.dataframe(imp_df, width="stretch", column_config={
                "Value": st.column_config.NumberColumn(format="₹%.0f"),
                "VaR %": st.column_config.NumberColumn("1-day VaR 95%", format="percent"),
                "VaR ₹": st.column_config.NumberColumn(format="₹%.0f"),
                "Volatility": st.column_config.NumberColumn(format="percent"),
            })
        if st.button("Track this trade plan in Journal", type="primary", icon=":material/bookmark_add:",
                     disabled=bool(sz.refused) or gres.status == "BLOCK"):
            ok = storage.track_prediction({
                "entry_date": str(df.index[-1].date()), "ticker": pick, "entry": f.price,
                "target_pct": f.target_pct, "stop_pct": f.stop_pct, "target_price": f.target_price,
                "stop_price": f.stop_price, "horizon": horizon, "p_target": f.p_target, "p_stop": f.p_stop,
                "grade": f.grade, "verdict": f.verdict, "qty": sz.qty, "source": "plan",
            })
            st.toast("Tracked. See Journal." if ok else "Already tracked today.")
        st.caption("Paper tracking only: JINNI never places orders. Edit the rules in Settings.")

with tabs[5]:
    news = events.headlines(pick)
    if news.empty:
        st.caption("No headlines available right now.")
    else:
        for _, n in news.iterrows():
            tags = f" <span class='badge b-hold'>{n['Tags']}</span>" if n["Tags"] else ""
            when = n["Time"].strftime("%d %b %H:%M") if n["Time"] is not None else ""
            st.markdown(f"<span class='hint'>{when}</span> [{n['Headline']}]({n['Link']}){tags}",
                        unsafe_allow_html=True)

if st.button("Open full analysis", icon=":material/candlestick_chart:"):
    st.session_state["analyze_ticker"] = pick
    st.switch_page("app/views/analyzer.py")
ui.disclaimer()
