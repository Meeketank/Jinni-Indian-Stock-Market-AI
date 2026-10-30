"""Portfolio construction: objectives, constraints, efficient frontier, whole-share trade list."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import components as ui
from app import storage
from app.theme import BLUE, GOLD, MUTED, SERIES, UP
from jinni import optimizer as opt
from jinni import risk
from jinni.universe import BENCHMARK, NIFTY50, display_name, normalize_symbol, sector_of

st.title("Optimizer")

s = ui.settings()
with st.container(border=True):
    c1, c2 = st.columns(2)
    with c1:
        source = st.radio("Universe", ["My holdings", "NIFTY 50", "Custom"], horizontal=True)
        if source == "Custom":
            raw = st.text_input("Symbols", "RELIANCE, HDFCBANK, INFY, ITC, LT, SUNPHARMA, TITAN, NTPC, BHARTIARTL")
            tickers = [normalize_symbol(x) for x in raw.replace(",", " ").split()]
        elif source == "NIFTY 50":
            tickers = list(NIFTY50)
        else:
            hold = st.session_state.get("holdings", storage.load_holdings())
            tickers = list(hold["ticker"])
        objective = st.selectbox("Objective", list(opt.OBJECTIVES), format_func=opt.OBJECTIVES.get)
        mu_method = st.selectbox("Expected returns", ["capm", "shrunk", "historical"],
                                 format_func={"capm": "CAPM equilibrium (beta × 7% ERP)",
                                              "shrunk": "Historical, shrunk 50% to the mean",
                                              "historical": "Historical (noisy)"}.get)
    with c2:
        max_w = st.slider("Max weight per stock", 0.05, 1.0, 0.15 if len(tickers) > 10 else 0.30, 0.01, format="%.2f")
        min_w = st.slider("Min weight per stock", 0.0, 0.05, 0.0, 0.005, format="%.3f")
        use_sector = st.checkbox("Cap sector weight", value=True)
        sector_cap = st.slider("Sector cap", 0.1, 1.0, 0.30, 0.05, format="%.2f") if use_sector else None
        lookback = st.selectbox("Estimation window", ["1y", "2y", "3y", "5y"], index=2)
        capital = st.number_input("Capital to deploy (₹)", min_value=10_000.0, value=float(s["capital"]),
                                  step=100_000.0, format="%.0f")

if len(tickers) < 2:
    st.info("Pick at least two stocks.")
    st.stop()

with st.spinner("Estimating covariance and solving…"):
    pdata = ui.prices(tuple(tickers) + (BENCHMARK,), "5y")
ui.data_source_notice(pdata)
n = {"1y": 252, "2y": 504, "3y": 756, "5y": 1260}[lookback]
avail = [t for t in tickers if t in pdata.close]
rets = risk.returns(pdata.close[avail]).iloc[-n:]
rets = rets.dropna(axis=1, thresh=int(0.9 * len(rets))).dropna()
if rets.shape[1] < 2:
    st.error("Not enough overlapping history for these stocks.")
    st.stop()
bench = risk.returns(pdata.close[BENCHMARK]).iloc[-n:] if BENCHMARK in pdata.close else None
cov = risk.ledoit_wolf_cov(rets)
mu = opt.expected_returns(rets, bench, mu_method, rf=s["rf"])
sectors = {t: sector_of(t) for t in rets.columns}
kw = dict(mu=mu, cov=cov, rf=s["rf"], min_w=min_w, max_w=max_w, sectors=sectors, sector_cap=sector_cap)
res = opt.optimize(rets, objective, **kw)
eq = opt.optimize(rets, "equal", **kw)

pr = risk.portfolio_returns(rets, res.weights)
pr_eq = risk.portfolio_returns(rets, eq.weights)
st_opt = risk.performance_stats(pr, bench, s["rf"])
ui.kpis([
    ("Expected return", ui.pct(res.exp_return, 1), f"vs {ui.pct(eq.exp_return, 1)} equal-wt"),
    ("Ex-ante volatility", ui.pct(res.volatility, 1), f"vs {ui.pct(eq.volatility, 1)} equal-wt"),
    ("Ex-ante Sharpe", ui.num(res.sharpe), f"vs {ui.num(eq.sharpe)} equal-wt"),
    ("Holdings", f"{(res.weights > 0).sum()}", f"eff. N {risk.effective_n(res.weights):.1f}"),
    ("Beta (backcast)", ui.num(st_opt.get("Beta"))),
    ("Max DD (backcast)", ui.pct(st_opt.get("Max drawdown"), 1), "", "down"),
])

t1, t2, t3, t4 = st.tabs(["Weights", "Efficient frontier", "Trade list", "Backcast"])

with t1:
    wv = res.weights[res.weights > 0].sort_values(ascending=False)
    rc = risk.risk_contributions(cov, res.weights).loc[wv.index]
    fig = go.Figure()
    labels = [display_name(t) for t in wv.index]
    fig.add_trace(go.Bar(y=labels, x=wv.values, name="Weight", orientation="h", marker_color=GOLD))
    fig.add_trace(go.Bar(y=labels, x=rc["% of risk"].values, name="Risk share", orientation="h", marker_color=BLUE))
    fig.update_layout(barmode="group", xaxis_tickformat=".0%", title=opt.OBJECTIVES[objective])
    fig.update_yaxes(autorange="reversed")
    ui.plot(fig, max(300, 30 * len(wv) + 80))
    sec_w = res.weights.groupby(pd.Series(sectors)).sum().sort_values(ascending=False)
    sec_w = sec_w[sec_w > 0]
    fig = go.Figure(go.Pie(labels=sec_w.index, values=sec_w.values, hole=0.6,
                           marker=dict(colors=SERIES, line=dict(color="#0A0E14", width=2)), textinfo="label+percent"))
    fig.update_layout(title="Sector mix", showlegend=False)
    ui.plot(fig, 320)

with t2:
    with st.spinner("Tracing frontier…"):
        fr = opt.efficient_frontier(rets, mu, cov, n_points=25, max_w=max_w, rf=s["rf"])
    fig = go.Figure()
    if not fr.empty:
        fig.add_trace(go.Scatter(x=fr["Volatility"], y=fr["Return"], mode="lines", name="Efficient frontier",
                                 line=dict(color=GOLD, width=2.5)))
    vols = np.sqrt(np.diag(cov.values) * 252)
    fig.add_trace(go.Scatter(x=vols, y=mu.reindex(rets.columns).values, mode="markers+text",
                             text=[display_name(t) for t in rets.columns], textposition="top center",
                             textfont=dict(size=9, color=MUTED), marker=dict(color=MUTED, size=7), name="Stocks"))
    for key, color, symbol in [("max_sharpe", UP, "star"), ("min_vol", BLUE, "diamond"),
                               ("risk_parity", "#E86AF0", "circle"), ("equal", "#FFFFFF", "x")]:
        r = opt.optimize(rets, key, **kw)
        fig.add_trace(go.Scatter(x=[r.volatility], y=[r.exp_return], mode="markers", name=opt.OBJECTIVES[key],
                                 marker=dict(color=color, size=14, symbol=symbol, line=dict(width=1, color="#0A0E14"))))
    if not fr.empty:
        best = fr.loc[fr["Sharpe"].idxmax()]
        xs = np.linspace(0, fr["Volatility"].max() * 1.1, 20)
        fig.add_trace(go.Scatter(x=xs, y=s["rf"] + best["Sharpe"] * xs, mode="lines", name="Capital market line",
                                 line=dict(color=MUTED, dash="dot", width=1)))
    fig.update_layout(xaxis_title="Volatility (ann.)", yaxis_title="Expected return (ann.)",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    ui.plot(fig, 480)
    st.caption("Frontier respects the per-stock cap (sector cap applies to the chosen portfolio only). "
               "Covariance uses Ledoit-Wolf shrinkage, which is far more stable than the raw sample estimate.")

with t3:
    last_px = pdata.close[rets.columns].ffill().iloc[-1]
    alloc, cash = opt.discrete_allocation(res.weights, last_px, capital)
    alloc.index = [display_name(t) for t in alloc.index]
    st.dataframe(alloc, width="stretch", column_config={
        "Price ₹": st.column_config.NumberColumn(format="₹%.2f"),
        "Value ₹": st.column_config.NumberColumn(format="₹%.0f"),
        "Actual weight": st.column_config.NumberColumn(format="percent"),
        "Target weight": st.column_config.NumberColumn(format="percent"),
    })
    st.caption(f"Whole shares only. Leftover cash: {ui.inr(cash)}.")
    if source == "My holdings":
        hold = st.session_state.get("holdings", storage.load_holdings()).set_index("ticker")["quantity"]
        tgt, _ = opt.discrete_allocation(res.weights, last_px, float((hold.reindex(last_px.index).fillna(0) * last_px).sum()))
        trades = opt.rebalance_trades(hold.reindex(rets.columns).fillna(0), tgt)
        trades.index = [display_name(t) for t in trades.index]
        st.markdown("**Rebalance from current holdings** (same total value)")
        st.dataframe(trades, width="stretch", column_config={
            "Price ₹": st.column_config.NumberColumn(format="₹%.2f"),
            "Notional ₹": st.column_config.NumberColumn(format="₹%.0f"),
            "Est. cost ₹": st.column_config.NumberColumn(format="₹%.0f"),
        })
        st.caption(f"Turnover {ui.inr(trades['Notional ₹'].sum())}, estimated costs {ui.inr(trades['Est. cost ₹'].sum())} "
                   "(≈15 bps per side: STT, exchange, stamp duty, GST, slippage). Selling may also trigger capital "
                   "gains tax (STCG 20% / LTCG 12.5% above ₹1.25 L).")
    st.download_button("Download allocation CSV", alloc.to_csv().encode(), "jinni_allocation.csv", "text/csv",
                       icon=":material/download:")

with t4:
    cum = pd.DataFrame({"Optimised": (1 + pr).cumprod(), "Equal weight": (1 + pr_eq).cumprod()})
    if bench is not None:
        cum["NIFTY 50"] = (1 + bench.reindex(pr.index).fillna(0)).cumprod()
    fig = go.Figure()
    for col, color in zip(cum.columns, [GOLD, BLUE, MUTED]):
        fig.add_trace(go.Scatter(x=cum.index, y=cum[col], name=col, line=dict(color=color, width=2 if col == "Optimised" else 1.4)))
    fig.update_layout(title="In-sample backcast (growth of ₹1)", hovermode="x unified")
    ui.plot(fig, 360)
    st.warning("This backcast uses the same data the optimiser saw, so it flatters the result. Judge portfolios by "
               "their risk profile (volatility, drawdown, concentration), not by this curve.", icon=":material/info:")
ui.disclaimer()
