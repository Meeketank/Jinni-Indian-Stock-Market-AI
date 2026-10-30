"""Strategy Lab: rule backtests with costs + walk-forward ML validation."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import components as ui
from app.theme import BLUE, DOWN, GOLD, MUTED, SERIES, UP
from jinni import backtest as bt
from jinni import ml
from jinni.universe import BENCHMARK, NIFTY50, display_name, normalize_symbol

st.title("Strategy Lab")
tab_bt, tab_ml = st.tabs(["Backtest rules", "ML model (walk-forward)"])

# ------------------------------------------------------------------ backtest
with tab_bt:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        t = st.selectbox("Stock", list(NIFTY50), index=list(NIFTY50).index("RELIANCE.NS"), format_func=display_name)
    with c2:
        period = st.selectbox("Period", ["3y", "5y", "10y"], index=1)
    with c3:
        cost = st.number_input("Cost per side (bps)", 0.0, 100.0, bt.DEFAULT_COST_BPS, 1.0)
    chosen = st.multiselect("Strategies", list(bt.STRATEGIES), default=list(bt.STRATEGIES),
                            format_func=bt.STRATEGIES.get)
    with st.spinner("Backtesting…"):
        pdata = ui.prices((t, BENCHMARK), period)
    ui.data_source_notice(pdata)
    if t not in pdata.close:
        st.error("No data.")
    else:
        df = pdata.ohlcv(t)
        bench = pdata.close.get(BENCHMARK)
        results = {k: bt.run(df, k, bench, cost) for k in chosen}
        fig = go.Figure()
        for (k, r), color in zip(results.items(), SERIES):
            fig.add_trace(go.Scatter(x=r["equity"].index, y=r["equity"], name=bt.STRATEGIES[k].split(":")[0],
                                     line=dict(width=2.2 if k == "composite" else 1.4, color=color)))
        fig.update_layout(title=f"{display_name(t)}: growth of ₹1, net of costs", hovermode="x unified",
                          yaxis_type="log")
        ui.plot(fig, 400)
        rows = {}
        for k, r in results.items():
            s = r["stats"]
            rows[bt.STRATEGIES[k].split(":")[0]] = {
                "CAGR": s.get("CAGR"), "Volatility": s.get("Volatility"), "Sharpe": s.get("Sharpe"),
                "Max DD": s.get("Max drawdown"), "Calmar": s.get("Calmar"), "Trades": s.get("Trades"),
                "Win rate": s.get("Win rate"), "Time in mkt": s.get("Time in market"), "Costs": s.get("Costs paid"),
            }
        tbl = pd.DataFrame(rows).T
        st.dataframe(tbl, width="stretch", column_config={
            **{c: st.column_config.NumberColumn(format="percent") for c in
               ["CAGR", "Volatility", "Max DD", "Win rate", "Time in mkt", "Costs"]},
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Calmar": st.column_config.NumberColumn(format="%.2f"),
            "Trades": st.column_config.NumberColumn(format="%d"),
        })
        st.caption("Signals use the close of day t and trade on day t+1 (no look-ahead). Long-only, delivery. "
                   "Costs are charged on every entry and exit.")

# ------------------------------------------------------------------------ ML
with tab_ml:
    st.markdown(
        "Gradient-boosted trees predict whether a stock will be up over the next *h* days. Validation is "
        "**walk-forward with a purge gap**: train only on the past, test on the following unseen block, so the "
        "accuracy below is what you could have expected live. Compare it with the base rate. "
        "Daily stock direction is close to a coin flip, and an honest model shows that."
    )
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        pool = st.multiselect("Train on", list(NIFTY50),
                              default=["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS"],
                              format_func=display_name)
        extra = st.text_input("Extra symbols", "", placeholder="e.g. IRCTC, HAL")
    with m2:
        horizon = st.selectbox("Horizon (days)", [1, 5, 10, 20], index=1)
    with m3:
        target = st.selectbox("Target", ["Absolute (stock up)", "Relative (beats NIFTY)"])
    with m4:
        folds = st.selectbox("Folds", [3, 4, 5, 6], index=2)
    pool = pool + [normalize_symbol(x) for x in extra.replace(",", " ").split() if x.strip()]
    run = st.button("Train & validate", type="primary", icon=":material/model_training:")
    if run and pool:
        with st.spinner("Building features and running walk-forward validation…"):
            pdata = ui.prices(tuple(pool) + (BENCHMARK,), "10y")
            bench = pdata.close.get(BENCHMARK)
            dfs = {t: pdata.ohlcv(t) for t in pool if t in pdata.close}
            try:
                res = ml.walk_forward(dfs, bench, horizon=horizon, n_splits=folds,
                                      relative=target.startswith("Relative"))
                st.session_state["ml_result"] = (res, pdata.source, horizon)
            except ValueError as e:
                st.error(str(e))
    if "ml_result" in st.session_state:
        res, src, h = st.session_state["ml_result"]
        if src == "demo":
            st.markdown('<span class="badge b-demo">DEMO DATA</span> <span class="hint">Synthetic prices are random, '
                        "so ~0 edge is the correct result here.</span>", unsafe_allow_html=True)
        sm = res.summary
        ui.kpis([
            ("Out-of-sample accuracy", ui.pct(sm["accuracy"], 1), f"{sm['n_oos']:,} predictions"),
            ("Base rate", ui.pct(sm["base_rate"], 1), "always guess majority"),
            ("Edge", ui.pct(sm["edge"], 1, sign=True), "accuracy − base", ui.tone(sm["edge"])),
            ("AUC", ui.num(sm["auc"]), "0.5 = no skill"),
            ("Brier score", ui.num(sm["brier"], 3), "lower is better"),
        ])
        st.dataframe(res.folds, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.3f") for c in
            ["Accuracy", "Base rate", "Edge vs base", "AUC", "Precision (up)"]})
        b1, b2 = st.columns(2, gap="medium")
        with b1:
            bk = sm["buckets"]
            fig = go.Figure(go.Bar(x=bk.index.astype(str), y=bk["Avg_fwd_ret"],
                                   marker_color=[UP if v >= 0 else DOWN for v in bk["Avg_fwd_ret"]],
                                   text=[f"{v:.2%}" for v in bk["Avg_fwd_ret"]], textposition="outside"))
            fig.update_layout(title=f"Avg {h}-day return by predicted-probability quintile", yaxis_tickformat=".1%")
            ui.plot(fig, 320)
            st.caption("A useful model shows returns rising from Q1 to Q5.")
        with b2:
            imp = res.importance.head(10).iloc[::-1]
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=GOLD))
            fig.update_layout(title="Top features (permutation importance, AUC drop)")
            ui.plot(fig, 320)
        ui.section("Live probabilities (final model, trained on all data)")
        lp = res.latest.copy()
        lp.index = [display_name(t) for t in lp.index]
        lp["Signal"] = np.where(lp["prob_up"] >= 0.55, "Bullish", np.where(lp["prob_up"] <= 0.45, "Bearish", "Neutral"))
        st.dataframe(lp.sort_values("prob_up", ascending=False), width="stretch", column_config={
            "prob_up": st.column_config.ProgressColumn(f"P(up in {h}d)", min_value=0, max_value=1, format="%.2f"),
            "as_of": st.column_config.DateColumn("As of"),
        })
        if sm["edge"] < 0.01:
            st.info("The model shows little or no edge over the base rate on this setup. Treat its probabilities as "
                    "noise. This is normal for short-horizon direction prediction.", icon=":material/info:")
ui.disclaimer()
