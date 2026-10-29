"""Portfolio risk: holdings, P&L, VaR/CVaR, decomposition, stress tests, performance, liquidity."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import components as ui
from app import storage
from app.theme import BLUE, DOWN, GOLD, MUTED, SERIES, UP
from jinni import risk
from jinni.universe import BENCHMARK, STRESS_SCENARIOS, display_name, normalize_symbol, sector_of

st.title("Portfolio Risk")

if "holdings" not in st.session_state:
    st.session_state["holdings"] = storage.load_holdings()

with st.expander("Holdings", expanded=False, icon=":material/edit:"):
    st.caption("Edit quantities and average cost, add rows with any NSE symbol (e.g. TATAELXSI) or BSE (.BO). "
               "Sample portfolio shown until you save your own.")
    edited = st.data_editor(
        st.session_state["holdings"], num_rows="dynamic", width="stretch", key="holdings_editor",
        column_config={
            "ticker": st.column_config.TextColumn("Symbol", required=True),
            "quantity": st.column_config.NumberColumn("Qty", min_value=0, step=1, required=True),
            "avg_cost": st.column_config.NumberColumn("Avg cost ₹", min_value=0, format="%.2f"),
        },
    )
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Apply & save", type="primary", width="stretch", icon=":material/save:"):
            e = edited.copy()
            e["ticker"] = e["ticker"].astype(str).map(normalize_symbol)
            e = e.groupby("ticker", as_index=False).agg(
                quantity=("quantity", "sum"),
                avg_cost=("avg_cost", "mean"),
            )
            st.session_state["holdings"] = e
            storage.save_holdings(e)
            st.rerun()
    with b2:
        st.download_button("Export CSV", st.session_state["holdings"].to_csv(index=False).encode(),
                           "jinni_holdings.csv", "text/csv", width="stretch", icon=":material/download:")
    with b3:
        up = st.file_uploader("Import CSV (ticker,quantity,avg_cost)", type="csv", label_visibility="collapsed")
        if up is not None:
            imp = pd.read_csv(up)
            imp.columns = [c.strip().lower() for c in imp.columns]
            if {"ticker", "quantity"} <= set(imp.columns):
                imp["avg_cost"] = imp.get("avg_cost", np.nan)
                imp["ticker"] = imp["ticker"].astype(str).map(normalize_symbol)
                st.session_state["holdings"] = imp[["ticker", "quantity", "avg_cost"]]
                storage.save_holdings(st.session_state["holdings"])
                st.rerun()
            else:
                st.error("CSV needs columns: ticker, quantity (avg_cost optional).")

h = st.session_state["holdings"].dropna(subset=["ticker", "quantity"])
h = h[h["quantity"] > 0]
if h.empty:
    st.info("Add at least one holding.")
    st.stop()

s = ui.settings()
c1, c2, c3 = st.columns(3)
with c1:
    lookback = st.selectbox("Risk lookback", ["1y", "2y", "3y", "5y"], index=1)
with c2:
    conf = st.selectbox("VaR confidence", [0.95, 0.99, 0.975], format_func=lambda x: f"{x:.1%}".rstrip("0").rstrip("."))
with c3:
    horizon = st.selectbox("VaR horizon (days)", [1, 5, 10, 21], index=0)

tickers = tuple(h["ticker"])
with st.spinner("Pricing portfolio and running risk models…"):
    pdata = ui.prices(tickers + (BENCHMARK,), "10y")
ui.data_source_notice(pdata)
avail = [t for t in tickers if t in pdata.close]
if not avail:
    st.error("None of the holdings returned prices.")
    st.stop()

closes = pdata.close[avail]
last_px = closes.ffill().iloc[-1]
prev_px = closes.ffill().iloc[-2]
hh = h.set_index("ticker").loc[avail]
mv = hh["quantity"] * last_px
value = float(mv.sum())
w = mv / value
cost = (hh["quantity"] * hh["avg_cost"]).where(hh["avg_cost"].notna())
pnl = mv - cost
day_pnl = float((hh["quantity"] * (last_px - prev_px)).sum())

n_look = {"1y": 252, "2y": 504, "3y": 756, "5y": 1260}[lookback]
rets_all = risk.returns(closes)
rets = rets_all.iloc[-n_look:]
bench = risk.returns(pdata.close[BENCHMARK]) if BENCHMARK in pdata.close else None
bench_l = bench.iloc[-n_look:] if bench is not None else None
pr = risk.portfolio_returns(rets, w)
stats = risk.performance_stats(pr, bench_l, s["rf"])
cov = risk.ledoit_wolf_cov(rets.dropna(axis=1, thresh=int(0.8 * len(rets))).dropna())
w_cov = w.reindex(cov.index).fillna(0)
w_cov = w_cov / w_cov.sum()
rep = risk.var_report(rets[cov.index], w_cov, value, conf, horizon, n_sims=8000)
var_hist = rep.table.loc["Historical simulation"]

ui.kpis([
    ("Market value", ui.inr(value), f"{len(avail)} positions"),
    ("Unrealised P&L", ui.inr(pnl.sum()) if pnl.notna().any() else "—",
     ui.pct(pnl.sum() / cost.sum(), 2, sign=True) if cost.notna().any() and cost.sum() else "", ui.tone(pnl.sum())),
    ("Today's P&L", ui.inr(day_pnl), ui.pct(day_pnl / (value - day_pnl), 2, sign=True), ui.tone(day_pnl)),
    (f"VaR {conf:.0%} · {horizon}d", ui.inr(var_hist["VaR ₹"]), ui.pct(var_hist["VaR %"]) + " (historical)", "down"),
    (f"CVaR {conf:.0%} · {horizon}d", ui.inr(var_hist["CVaR ₹"]), "expected shortfall", "down"),
    ("Beta vs NIFTY", ui.num(stats.get("Beta")), f"R² {ui.num(stats.get('R²'))}"),
    ("Volatility", ui.pct(stats.get("Volatility"), 1), "annualised"),
    ("Sharpe", ui.num(stats.get("Sharpe")), f"rf {s['rf']:.1%}"),
])

tabs = st.tabs(["Exposure", "VaR & tails", "Risk decomposition", "Stress tests", "Performance", "Liquidity"])

# ---------------------------------------------------------------- exposure
with tabs[0]:
    tbl = pd.DataFrame({
        "Qty": hh["quantity"], "Avg cost": hh["avg_cost"], "LTP": last_px, "Value": mv, "Weight": w,
        "Day %": last_px / prev_px - 1, "P&L": pnl, "P&L %": pnl / cost,
        "Sector": [sector_of(t) for t in avail],
    })
    tbl.index = [display_name(t) for t in tbl.index]
    st.dataframe(tbl.sort_values("Value", ascending=False), width="stretch", column_config={
        "Avg cost": st.column_config.NumberColumn(format="₹%.2f"),
        "LTP": st.column_config.NumberColumn(format="₹%.2f"),
        "Value": st.column_config.NumberColumn(format="₹%.0f"),
        "Weight": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=1),
        "Day %": st.column_config.NumberColumn(format="percent"),
        "P&L": st.column_config.NumberColumn(format="₹%.0f"),
        "P&L %": st.column_config.NumberColumn(format="percent"),
    })
    e1, e2 = st.columns(2, gap="medium")
    with e1:
        sec = tbl.groupby("Sector")["Value"].sum().sort_values(ascending=False)
        fig = go.Figure(go.Pie(labels=sec.index, values=sec.values, hole=0.62, sort=False,
                               marker=dict(colors=SERIES, line=dict(color="#0A0E14", width=2)),
                               textinfo="percent", textposition="inside"))
        fig.update_layout(title="Sector allocation", showlegend=True,
                          annotations=[dict(text=ui.inr(value), showarrow=False, font=dict(size=15))])
        ui.plot(fig, 360)
    with e2:
        top_w = w.max()
        hhi_n = risk.effective_n(w)
        ui.kpis([
            ("Largest position", ui.pct(top_w, 1), display_name(w.idxmax())),
            ("Top 5 weight", ui.pct(w.nlargest(5).sum(), 1)),
            ("Effective # of bets", f"{hhi_n:.1f}", "1 / Σw² (Herfindahl)"),
            ("Diversification ratio", ui.num(risk.diversification_ratio(cov, w_cov)), "Σwσ / σp"),
        ])
        if top_w > 0.2:
            st.warning(f"Concentration: {display_name(w.idxmax())} is {top_w:.0%} of the portfolio. SEBI's mutual-fund "
                       "single-stock cap is 10%, a useful reference point.")
        big_sector = tbl.groupby("Sector")["Weight"].sum()
        if big_sector.max() > 0.35:
            st.warning(f"Sector concentration: {big_sector.idxmax()} is {big_sector.max():.0%} of the portfolio.")

# -------------------------------------------------------------------- VaR
with tabs[1]:
    tbl = rep.table.copy()
    st.dataframe(tbl, width="stretch", column_config={
        "VaR %": st.column_config.NumberColumn(format="percent"),
        "CVaR %": st.column_config.NumberColumn(format="percent"),
        "VaR ₹": st.column_config.NumberColumn(format="₹%.0f"),
        "CVaR ₹": st.column_config.NumberColumn(format="₹%.0f"),
    })
    st.caption(f"Loss not expected to be exceeded on {conf:.0%} of {horizon}-day periods (VaR), and the average loss "
               "when it is (CVaR). Historical uses overlapping real windows; Cornish-Fisher corrects the normal model "
               "for skew and fat tails; Monte Carlo draws Student-t (5 d.o.f.) shocks with Ledoit-Wolf covariance.")
    final = rep.paths[:, -1]
    v1, v2 = st.columns(2, gap="medium")
    with v1:
        fig = go.Figure(go.Histogram(x=final * value, nbinsx=80, marker_color=BLUE, opacity=0.85, name="Simulated"))
        fig.add_vline(x=-rep.table.loc["Monte Carlo (Student-t, LW cov)", "VaR ₹"], line=dict(color=DOWN, dash="dash"),
                      annotation_text="VaR", annotation_font_color=DOWN)
        fig.add_vline(x=-rep.table.loc["Monte Carlo (Student-t, LW cov)", "CVaR ₹"], line=dict(color=DOWN),
                      annotation_text="CVaR", annotation_position="bottom left", annotation_font_color=DOWN)
        fig.update_layout(title=f"Monte Carlo {horizon}-day P&L distribution (₹)", showlegend=False, bargap=0.02)
        ui.plot(fig, 340)
    with v2:
        bt = risk.var_backtest(pr, conf)
        if bt:
            ser = bt["series"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ser.index, y=ser["r"], mode="lines", name="Daily return",
                                     line=dict(color=MUTED, width=0.8)))
            fig.add_trace(go.Scatter(x=ser.index, y=-ser["var"], name="−VaR (rolling 250d)",
                                     line=dict(color=GOLD, width=1.4)))
            br = ser[ser["breach"]]
            fig.add_trace(go.Scatter(x=br.index, y=br["r"], mode="markers", name="Breach",
                                     marker=dict(color=DOWN, size=6)))
            fig.update_yaxes(tickformat=".1%")
            fig.update_layout(title="1-day VaR backtest")
            ui.plot(fig, 340)
            verdict = "passes" if bt["model_ok"] else "fails"
            st.caption(f"{bt['breaches']} breaches vs {bt['expected']:.0f} expected over {bt['observations']} days "
                       f"(rate {bt['breach_rate']:.2%}). Kupiec test p = {bt['kupiec_p']:.2f} → model **{verdict}** "
                       "at 5% significance.")
    ui.kpis([
        ("Skew", ui.num(stats.get("Skew")), "negative = crash-prone"),
        ("Excess kurtosis", ui.num(stats.get("Excess kurtosis")), "> 0 = fat tails"),
        ("Worst day", ui.pct(stats.get("Worst day")), "", "down"),
        ("Best day", ui.pct(stats.get("Best day")), "", "up"),
    ])

# ---------------------------------------------------------- decomposition
with tabs[2]:
    rc = risk.risk_contributions(cov, w_cov)
    rc.index = [display_name(t) for t in rc.index]
    rc = rc.sort_values("% of risk", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=rc.index, x=rc["Weight"], name="Capital weight", orientation="h", marker_color=BLUE))
    fig.add_trace(go.Bar(y=rc.index, x=rc["% of risk"], name="Risk contribution", orientation="h", marker_color=GOLD))
    fig.update_layout(barmode="group", title="Capital vs risk allocation", xaxis_tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    ui.plot(fig, max(320, 28 * len(rc) + 80))
    st.caption("A position whose gold bar is longer than its blue bar adds more risk than its capital share. "
               "Euler decomposition: contributions sum exactly to portfolio volatility.")
    d1, d2 = st.columns(2, gap="medium")
    with d1:
        corr = rets[cov.index].corr()
        labels = [display_name(t) for t in corr.index]
        fig = go.Figure(go.Heatmap(z=corr.values, x=labels, y=labels, zmin=-1, zmax=1,
                                   colorscale=[[0, BLUE], [0.5, "#111823"], [1, DOWN]],
                                   hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>"))
        avg_corr = corr.values[np.triu_indices_from(corr.values, 1)].mean() if len(corr) > 1 else np.nan
        fig.update_layout(title=f"Correlation (avg {avg_corr:.2f})")
        fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
        fig.update_yaxes(tickfont=dict(size=9), autorange="reversed")
        ui.plot(fig, 420)
    with d2:
        if bench_l is not None:
            fe = risk.factor_exposures(rets, bench_l)
            fe.index = [display_name(t) for t in fe.index]
            st.markdown("**Market factor exposure**")
            st.dataframe(fe.sort_values("Beta", ascending=False), width="stretch", column_config={
                "Beta": st.column_config.NumberColumn(format="%.2f"),
                "Correlation": st.column_config.NumberColumn(format="%.2f"),
                "Total vol": st.column_config.NumberColumn(format="percent"),
                "Idiosyncratic vol": st.column_config.NumberColumn(format="percent"),
                "Systematic % of var": st.column_config.NumberColumn(format="percent"),
            })
            sys_share = float((rc["Weight"].values * fe.reindex(rc.index)["Systematic % of var"].fillna(0).values).sum())
            st.caption(f"Roughly {sys_share:.0%} of a typical holding's variance is explained by the NIFTY. "
                       "The rest is stock-specific and can be diversified away.")

# ----------------------------------------------------------------- stress
with tabs[3]:
    st.markdown("**Historical replay**: today's weights held through past crises.")
    full_close = pdata.close[avail]
    nifty_px = pdata.close[BENCHMARK] if BENCHMARK in pdata.close else None
    stress = risk.historical_stress(full_close, w, STRESS_SCENARIOS, nifty_px)
    stress["P&L ₹"] = stress["Portfolio %"] / 100 * value
    st.dataframe(stress, width="stretch", column_config={
        "Portfolio %": st.column_config.NumberColumn(format="%+.2f%%"),
        "NIFTY %": st.column_config.NumberColumn(format="%+.2f%%"),
        "Coverage %": st.column_config.NumberColumn(format="%.0f%%",
                                                    help="Share of today's portfolio that existed in that window"),
        "P&L ₹": st.column_config.NumberColumn(format="₹%.0f"),
    })
    valid = stress.dropna(subset=["Portfolio %"])
    if not valid.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(y=valid.index, x=valid["Portfolio %"] / 100, name="Portfolio", orientation="h",
                             marker_color=GOLD))
        fig.add_trace(go.Bar(y=valid.index, x=valid["NIFTY %"] / 100, name="NIFTY 50", orientation="h",
                             marker_color=MUTED))
        fig.update_layout(barmode="group", xaxis_tickformat=".0%", title="Scenario returns")
        fig.update_yaxes(autorange="reversed")
        ui.plot(fig, 360)

    st.markdown("**Hypothetical shocks** via each holding's beta to NIFTY.")
    if bench_l is not None:
        betas = risk.factor_exposures(rets, bench_l)["Beta"]
        shock = st.slider("NIFTY move", -40, 20, -15, 1, format="%d%%") / 100
        port_move = risk.factor_shock(betas, w, shock)
        shocks = pd.Series({f"NIFTY {x:+.0%}": risk.factor_shock(betas, w, x) for x in (-0.3, -0.2, -0.1, -0.05, 0.05, 0.1)})
        ui.kpis([
            ("Portfolio move", ui.pct(port_move, 1, sign=True), ui.inr(port_move * value), ui.tone(port_move)),
            ("Portfolio beta", ui.num((w * betas.reindex(w.index).fillna(1)).sum())),
        ])
        ui.plot(ui.colored_bar(shocks, "Beta-implied P&L under market moves"), 280)
        st.caption("Beta shocks assume correlations hold. In real crises correlations rise, so historical replay "
                   "is usually the more conservative view.")

# ------------------------------------------------------------ performance
with tabs[4]:
    if bench_l is not None:
        cum = pd.DataFrame({"Portfolio": (1 + pr).cumprod(), "NIFTY 50": (1 + bench_l.reindex(pr.index).fillna(0)).cumprod()})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum.index, y=cum["Portfolio"], name="Portfolio (current weights)",
                                 line=dict(color=GOLD, width=2)))
        fig.add_trace(go.Scatter(x=cum.index, y=cum["NIFTY 50"], name="NIFTY 50", line=dict(color=MUTED, width=1.5)))
        fig.update_layout(title="Growth of ₹1 (backcast)", hovermode="x unified")
        ui.plot(fig, 340)
        dd = risk.drawdown_series(pr)
        fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color=DOWN, width=1),
                                   fillcolor="rgba(239,68,68,0.18)", name="Drawdown"))
        fig.update_layout(title="Drawdown", yaxis_tickformat=".0%")
        ui.plot(fig, 220)
        rb = risk.rolling_beta(pr, bench_l)
        if not rb.empty:
            fig = go.Figure(go.Scatter(x=rb.index, y=rb, line=dict(color=BLUE, width=1.5), name="Rolling beta"))
            fig.add_hline(y=1, line=dict(color=MUTED, dash="dot"))
            fig.update_layout(title="Rolling 3-month beta")
            ui.plot(fig, 220)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Calmar", "Beta", "Alpha (ann.)",
            "Tracking error", "Information ratio", "Up capture", "Down capture"]
    pct_keys = {"CAGR", "Volatility", "Max drawdown", "Alpha (ann.)", "Tracking error", "Up capture", "Down capture"}
    ui.kpis([(k, ui.pct(stats[k], 1) if k in pct_keys else ui.num(stats[k])) for k in keys if k in stats])
    st.caption("Backcast: today's weights applied over the lookback window, rebalanced daily. "
               "It shows how this mix behaves, not your actual historical returns.")

# --------------------------------------------------------------- liquidity
with tabs[5]:
    part = st.slider("Max participation in daily volume", 0.05, 0.5, 0.2, 0.05, format="%.2f")
    liq = risk.liquidity_profile(w, value, pdata.close[avail], pdata.volume[avail], part)
    liq.index = [display_name(t) for t in liq.index]
    st.dataframe(liq.sort_values("Days to exit", ascending=False), width="stretch", column_config={
        "Position ₹": st.column_config.NumberColumn(format="₹%.0f"),
        "20d ADV ₹": st.column_config.NumberColumn(format="₹%.0f"),
        "% of ADV": st.column_config.NumberColumn(format="%.3f%%"),
        "Days to exit": st.column_config.NumberColumn(format="%.2f"),
    })
    worst = liq["Days to exit"].max()
    st.caption(f"Whole portfolio can be exited in about {max(worst, 0):.2f} trading days at {part:.0%} of average daily "
               "value traded, without moving the market much.")

ui.disclaimer()
