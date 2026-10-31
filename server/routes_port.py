"""Screener, portfolio risk, optimizer, strategy lab and journal scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from jinni import backtest as bt
from jinni import forecast as fc
from jinni import indicators as ind
from jinni import optimizer as opt
from jinni import risk
from jinni import setups as jsetups
from jinni import signals
from jinni.universe import BENCHMARK, NIFTY50, STRESS_SCENARIOS, display_name, normalize_symbol, sector_of

from . import figs
from .core import (nifty50, ApiError, caption, callout, col, cols, doc, inr, kpis, memo, num, pct, plot, prices, reply, section,
                   source_notice, table, tabs, text, tone)

router = APIRouter()


def tick_list(s: str, limit: int = 60) -> list[str]:
    return list(dict.fromkeys(normalize_symbol(x) for x in s.replace(" ", ",").split(",") if x.strip()))[:limit]


# -------------------------------------------------------------- screener


@router.get("/screen")
def screen(tickers: str = "nifty50", ratings: str = "STRONG BUY,BUY,HOLD,SELL,STRONG SELL", min_score: int = 0,
           max_rsi: int = 100, sectors: str = "", setups: str = ""):
    tk = nifty50() if tickers.lower() == "nifty50" else tick_list(tickers)
    if not tk:
        raise ApiError(400, "No symbols.")
    pdx = prices(tk + [BENCHMARK], "2y", ttl=900)
    bench = pdx.close.get(BENCHMARK)
    rows = []
    for t in tk:
        if t not in pdx.close:
            continue
        df = pdx.ohlcv(t)
        if len(df) < 210:
            continue
        sc = signals.factor_scores(df, bench).iloc[-1]
        c = df["Close"]
        plan = signals.trade_plan(df, "LONG")
        rows.append({"t": t, "Symbol": display_name(t), "Sector": sector_of(t) if t in NIFTY50 else "—",
                     "Price": c.iloc[-1], "1D": c.iloc[-1] / c.iloc[-2] - 1, "1M": c.iloc[-1] / c.iloc[-22] - 1,
                     "6M": c.iloc[-1] / c.iloc[-127] - 1, "Score": sc["Composite"],
                     "Rating": signals.rating_from_score(sc["Composite"]), "Trend": sc["Trend"],
                     "Momentum": sc["Momentum"], "Timing": sc["Timing"], "Volume": sc["Volume"], "Risk": sc["Risk"],
                     "RSI": ind.rsi(c).iloc[-1], "Vol 20d": ind.realized_vol(c).iloc[-1], "Stop": plan.stop,
                     "T1": plan.targets[0], "Upside T1": plan.targets[0] / plan.entry - 1,
                     "Setups": ", ".join(jsetups.active_setups(df))})
    res = pd.DataFrame(rows)
    if res.empty:
        return reply(doc([callout("warn", "No symbols had enough history (need ~10 months).")]), ttl=60)
    want = [x for x in ratings.split(",") if x]
    view = res[res["Rating"].isin(want) & (res["Score"] >= min_score) & (res["RSI"] <= max_rsi)]
    if sectors:
        view = view[view["Sector"].isin([s for s in sectors.split(",") if s])]
    if setups:
        pick = [s for s in setups.split(",") if s]
        view = view[view["Setups"].apply(lambda x: any(p in x for p in pick))]
    view = view.sort_values("Score", ascending=False)
    cnt = res["Rating"].value_counts()
    blocks = source_notice(pdx.source)
    blocks.append(kpis([("Screened", f"{len(res)}", f"{len(view)} match filters"),
                        ("Buy / Strong buy", f"{cnt.get('BUY', 0) + cnt.get('STRONG BUY', 0)}", "", "up"),
                        ("Hold", f"{cnt.get('HOLD', 0)}", ""),
                        ("Sell / Strong sell", f"{cnt.get('SELL', 0) + cnt.get('STRONG SELL', 0)}", "", "down"),
                        ("Median score", f"{res['Score'].median():.0f}", "universe breadth")]))
    blocks.append(table(view, [col("Symbol", "Stock"), col("Score", "Score", "bar100"), col("Rating", "Rating", "badge"),
                               col("Price", "Price", "inr2"), col("1D", "1D", "spct2"), col("1M", "1M", "spct1"),
                               col("6M", "6M", "spct1"), col("Trend", "Trend", "n0"), col("Momentum", "Mom", "n0"),
                               col("Timing", "Timing", "n0"), col("Volume", "Vol", "n0"), col("Risk", "Risk", "n0"),
                               col("RSI", "RSI", "n0"), col("Vol 20d", "Vol 20d", "pct0"), col("Stop", "Stop", "inr2"),
                               col("T1", "T1", "inr2"), col("Upside T1", "Upside", "pct1"), col("Setups", "Setups"),
                               col("Sector", "Sector")], select={"key": "t", "action": "analyze"},
                        height=min(600, 44 + 38 * len(view))))
    sc_ = figs.fig([{"type": "scatter", "mode": "markers+text", "x": figs._v(res["Vol 20d"]), "y": figs._v(res["6M"]),
                     "text": list(res["Symbol"]), "textposition": "top center", "textfont": {"size": 9},
                     "marker": {"size": np.clip(res["Score"] / 4, 6, 26).tolist(), "color": figs._v(res["Score"], 1),
                                "cmin": 20, "cmax": 80, "line": {"width": 0},
                                "colorscale": [[0, figs.DOWN], [0.5, figs.GOLD], [1, figs.UP]],
                                "colorbar": {"title": "Score", "thickness": 10}},
                     "hovertemplate": "%{text}<br>6M %{y:.1%}<br>Vol %{x:.1%}<extra></extra>"}],
                   {"xaxis": {"title": "20-day volatility (ann.)", "tickformat": ".0%"},
                    "yaxis": {"title": "6-month return", "tickformat": ".0%"}, "title": "Momentum vs risk map",
                    "showlegend": False})
    blocks.append(plot(sc_, 440))
    if tickers.lower() == "nifty50":
        by = res.groupby("Sector")["Score"].mean().sort_values(ascending=False)
        blocks.append(plot(figs.bar_h(by.index, (by - 50) / 100, "Sector score vs neutral (50)", ".0%"), 360))
    return reply(doc(blocks, pdx.source, sectors=sorted(res["Sector"].unique().tolist())), ttl=300)


# ------------------------------------------------------------ portfolio


class Holding(BaseModel):
    ticker: str
    quantity: float = Field(ge=0)
    avg_cost: float | None = None


class RiskReq(BaseModel):
    holdings: list[Holding]
    lookback: str = "2y"
    conf: float = 0.95
    horizon: int = 1
    rf: float = 0.065
    shock: float = -0.15
    participation: float = 0.2


def _holdings_df(hs: list[Holding]) -> pd.DataFrame:
    rows = {}
    for h in hs:
        if h.quantity > 0:
            t = normalize_symbol(h.ticker)
            r = rows.setdefault(t, {"quantity": 0.0, "cost": 0.0, "n": 0})
            r["quantity"] += h.quantity
            if h.avg_cost:
                r["cost"] += h.avg_cost * h.quantity
                r["n"] += h.quantity
    if not rows:
        raise ApiError(400, "Add at least one holding.")
    df = pd.DataFrame({t: {"quantity": r["quantity"], "avg_cost": r["cost"] / r["n"] if r["n"] else np.nan}
                       for t, r in rows.items()}).T
    return df.iloc[:40]


@router.post("/portfolio/risk")
def portfolio_risk(req: RiskReq):
    h = _holdings_df(req.holdings)
    tickers = list(h.index)
    pdx = prices(tickers + [BENCHMARK], "10y", ttl=900)
    avail = [t for t in tickers if t in pdx.close]
    if not avail:
        raise ApiError(404, "None of the holdings returned prices.")
    closes = pdx.close[avail]
    last_px, prev_px = closes.ffill().iloc[-1], closes.ffill().iloc[-2]
    hh = h.loc[avail]
    mv = hh["quantity"] * last_px
    value = float(mv.sum())
    w = mv / value
    cost = (hh["quantity"] * hh["avg_cost"]).where(hh["avg_cost"].notna())
    pnl = mv - cost
    day_pnl = float((hh["quantity"] * (last_px - prev_px)).sum())
    n_look = {"1y": 252, "2y": 504, "3y": 756, "5y": 1260}.get(req.lookback, 504)
    rets = risk.returns(closes).iloc[-n_look:]
    bench = risk.returns(pdx.close[BENCHMARK]) if BENCHMARK in pdx.close else None
    bench_l = bench.iloc[-n_look:] if bench is not None else None
    pr = risk.portfolio_returns(rets, w)
    stats = risk.performance_stats(pr, bench_l, req.rf)
    ok_cols = rets.dropna(axis=1, thresh=int(0.8 * len(rets))).dropna()
    if ok_cols.shape[1] == 0 or len(ok_cols) < 60:
        raise ApiError(400, "Not enough overlapping price history for these holdings.")
    cov = risk.ledoit_wolf_cov(ok_cols)
    w_cov = w.reindex(cov.index).fillna(0)
    w_cov = w_cov / w_cov.sum()
    conf = req.conf if req.conf in (0.95, 0.975, 0.99) else 0.95
    horizon = req.horizon if req.horizon in (1, 5, 10, 21) else 1
    rep = risk.var_report(rets[cov.index], w_cov, value, conf, horizon, n_sims=6000)
    vh = rep.table.loc["Historical simulation"]
    blocks = source_notice(pdx.source, pdx.missing)
    blocks.append(kpis([
        ("Market value", inr(value), f"{len(avail)} positions"),
        ("Unrealised P&L", inr(pnl.sum()) if pnl.notna().any() else "—",
         pct(pnl.sum() / cost.sum(), 2, True) if cost.notna().any() and cost.sum() else "", tone(pnl.sum())),
        ("Today's P&L", inr(day_pnl), pct(day_pnl / (value - day_pnl), 2, True), tone(day_pnl)),
        (f"VaR {conf:.0%} · {horizon}d", inr(vh["VaR ₹"]), pct(vh["VaR %"]) + " (historical)", "down"),
        (f"CVaR {conf:.0%} · {horizon}d", inr(vh["CVaR ₹"]), "expected shortfall", "down"),
        ("Beta vs NIFTY", num(stats.get("Beta")), f"R² {num(stats.get('R²'))}"),
        ("Volatility", pct(stats.get("Volatility"), 1), "annualised"),
        ("Sharpe", num(stats.get("Sharpe")), f"rf {req.rf:.1%}")]))

    # ------------------------------------------------ exposure
    ex = pd.DataFrame({"t": avail, "Stock": [display_name(t) for t in avail], "Qty": hh["quantity"].values,
                       "Avg cost": hh["avg_cost"].values, "LTP": last_px.values, "Value": mv.values, "Weight": w.values,
                       "Day %": (last_px / prev_px - 1).values, "P&L": pnl.values, "P&L %": (pnl / cost).values,
                       "Sector": [sector_of(t) for t in avail]}).sort_values("Value", ascending=False)
    sec = ex.groupby("Sector")["Value"].sum().sort_values(ascending=False)
    hhi_n = risk.effective_n(w)
    warn = []
    if w.max() > 0.2:
        warn.append(callout("warn", f"Concentration: {display_name(w.idxmax())} is {w.max():.0%} of the portfolio. SEBI's "
                                    "mutual-fund single-stock cap is 10%, a useful reference point."))
    sw = ex.groupby("Sector")["Weight"].sum()
    if sw.max() > 0.35:
        warn.append(callout("warn", f"Sector concentration: {sw.idxmax()} is {sw.max():.0%} of the portfolio."))
    t_exp = [table(ex, [col("Stock", "Stock"), col("Qty", "Qty", "n0"), col("Avg cost", "Avg cost", "inr2"),
                        col("LTP", "LTP", "inr2"), col("Value", "Value", "inr0"), col("Weight", "Weight", "bar01"),
                        col("Day %", "Day %", "spct2"), col("P&L", "P&L", "inr0"), col("P&L %", "P&L %", "spct1"),
                        col("Sector", "Sector")], select={"key": "t", "action": "analyze"}),
             cols([plot(figs.donut(sec.index, sec.values, "Sector allocation", inr(value)), 340)],
                  [kpis([("Largest position", pct(w.max(), 1), display_name(w.idxmax())),
                         ("Top 5 weight", pct(w.nlargest(5).sum(), 1)),
                         ("Effective # of bets", f"{hhi_n:.1f}", "1 / Σw² (Herfindahl)"),
                         ("Diversification ratio", num(risk.diversification_ratio(cov, w_cov)), "Σwσ / σp")])] + warn)]

    # ------------------------------------------------ VaR
    tb = rep.table.reset_index()
    final = rep.paths[:, -1] * value
    mc = rep.table.loc["Monte Carlo (Student-t, LW cov)"]
    hist = figs.histogram(final, 60, f"Monte Carlo {horizon}-day P&L distribution (₹)", figs.BLUE,
                          [figs.vline(-mc["VaR ₹"], figs.DOWN, "dash", "VaR"),
                           figs.vline(-mc["CVaR ₹"], figs.DOWN, "solid", "CVaR")])
    t_var = [table(tb, [col("Method", "Method"), col("VaR %", "VaR %", "pct2"), col("CVaR %", "CVaR %", "pct2"),
                        col("VaR ₹", "VaR ₹", "inr0"), col("CVaR ₹", "CVaR ₹", "inr0")]),
             caption(f"Loss not expected to be exceeded on {conf:.0%} of {horizon}-day periods (VaR), and the average loss "
                     "when it is (CVaR). Historical uses overlapping real windows; Cornish-Fisher corrects the normal "
                     "model for skew and fat tails; Monte Carlo draws Student-t (5 d.o.f.) shocks with Ledoit-Wolf "
                     "covariance.")]
    bts = risk.var_backtest(pr, conf)
    right = []
    if bts:
        ser = bts["series"]
        f = figs.lines(ser.index, {"Daily return": dict(y=ser["r"], color=figs.MUTED, width=0.8),
                                   "−VaR (rolling 250d)": dict(y=-ser["var"], color=figs.GOLD, width=1.4)},
                       "1-day VaR backtest", yfmt=".1%")
        br = ser[ser["breach"]]
        f["data"].append({"type": "scatter", "mode": "markers", "name": "Breach", "x": [d.strftime("%Y-%m-%d") for d in br.index],
                          "y": figs._v(br["r"]), "marker": {"color": figs.DOWN, "size": 6}})
        right = [plot(f, 320), caption(f"{bts['breaches']} breaches vs {bts['expected']:.0f} expected over "
                                       f"{bts['observations']} days (rate {bts['breach_rate']:.2%}). Kupiec test p = "
                                       f"{bts['kupiec_p']:.2f} → model **{'passes' if bts['model_ok'] else 'fails'}** at 5% "
                                       "significance.")]
    t_var.append(cols([plot(hist, 320)], right))
    t_var.append(kpis([("Skew", num(stats.get("Skew")), "negative = crash-prone"),
                       ("Excess kurtosis", num(stats.get("Excess kurtosis")), "> 0 = fat tails"),
                       ("Worst day", pct(stats.get("Worst day")), "", "down"), ("Best day", pct(stats.get("Best day")), "", "up")]))

    # ------------------------------------------------ decomposition
    rc_ = risk.risk_contributions(cov, w_cov)
    rc_.index = [display_name(t) for t in rc_.index]
    rc_ = rc_.sort_values("% of risk", ascending=False)
    gp = figs.grouped_h(rc_.index, {"Capital weight": (rc_["Weight"], figs.BLUE), "Risk contribution": (rc_["% of risk"], figs.GOLD)},
                        "Capital vs risk allocation")
    corr = rets[cov.index].corr()
    labels = [display_name(t) for t in corr.index]
    avg_corr = float(corr.values[np.triu_indices_from(corr.values, 1)].mean()) if len(corr) > 1 else float("nan")
    hm = figs.heatmap(corr.values, labels, labels, f"Correlation (avg {avg_corr:.2f})", -1, 1,
                      [[0, figs.BLUE], [0.5, "var(--panel2)"], [1, figs.DOWN]], fmt=".2f")
    t_dec = [plot(gp, max(320, 28 * len(rc_) + 80)),
             caption("A position whose gold bar is longer than its blue bar adds more risk than its capital share. Euler "
                     "decomposition: contributions sum exactly to portfolio volatility."), plot(hm, 420)]
    if bench_l is not None:
        fe = risk.factor_exposures(rets, bench_l)
        fe.index = [display_name(t) for t in fe.index]
        t_dec.append(text("**Market factor exposure**"))
        t_dec.append(table(fe.sort_values("Beta", ascending=False).reset_index().rename(columns={"index": "Stock"}),
                           [col("Stock", "Stock"), col("Beta", "Beta", "n2"), col("Correlation", "Corr", "n2"),
                            col("Total vol", "Total vol", "pct0"), col("Idiosyncratic vol", "Idio. vol", "pct0"),
                            col("Systematic % of var", "Systematic", "pct0")]))

    # ------------------------------------------------ stress
    nifty_px = pdx.close[BENCHMARK] if BENCHMARK in pdx.close else None
    st_ = risk.historical_stress(closes, w, STRESS_SCENARIOS, nifty_px)
    st_["P&L ₹"] = st_["Portfolio %"] / 100 * value
    t_st = [text("**Historical replay**: today's weights held through past crises."),
            table(st_.reset_index(), [col("Scenario", "Scenario"), col("Portfolio %", "Portfolio", "n2"),
                                      col("NIFTY %", "NIFTY", "n2"),
                                      col("Coverage %", "Coverage", "n0", help="Share of today's portfolio that existed then"),
                                      col("P&L ₹", "P&L ₹", "inr0")])]
    valid = st_.dropna(subset=["Portfolio %"])
    if not valid.empty:
        t_st.append(plot(figs.grouped_h(valid.index, {"Portfolio": (valid["Portfolio %"] / 100, figs.GOLD),
                                                      "NIFTY 50": (valid["NIFTY %"] / 100, figs.MUTED)}, "Scenario returns"), 340))
    if bench_l is not None:
        betas = risk.factor_exposures(rets, bench_l)["Beta"]
        shock = float(np.clip(req.shock, -0.6, 0.3))
        pm = risk.factor_shock(betas, w, shock)
        shocks = pd.Series({f"NIFTY {x:+.0%}": risk.factor_shock(betas, w, x) for x in (-0.3, -0.2, -0.1, -0.05, 0.05, 0.1)})
        t_st += [text("**Hypothetical shocks** via each holding's beta to NIFTY."),
                 kpis([("Portfolio move", pct(pm, 1, True), f"{inr(pm * value)} if NIFTY {shock:+.0%}", tone(pm)),
                       ("Portfolio beta", num((w * betas.reindex(w.index).fillna(1)).sum()))]),
                 plot(figs.bar_h(shocks.index, shocks.values, "Beta-implied P&L under market moves", ".1%"), 280),
                 caption("Beta shocks assume correlations hold. In real crises correlations rise, so historical replay is "
                         "usually the more conservative view.")]

    # ------------------------------------------------ performance
    t_perf = []
    if bench_l is not None:
        gr = pd.DataFrame({"Portfolio": (1 + pr).cumprod(), "NIFTY 50": (1 + bench_l.reindex(pr.index).fillna(0)).cumprod()})
        t_perf.append(plot(figs.lines(gr.index, {"Portfolio (current weights)": dict(y=gr["Portfolio"], color=figs.GOLD, width=2),
                                                 "NIFTY 50": dict(y=gr["NIFTY 50"], color=figs.MUTED, width=1.5)},
                                      "Growth of ₹1 (backcast)"), 320))
        dd = risk.drawdown_series(pr)
        t_perf.append(plot(figs.lines(dd.index, {"Drawdown": dict(y=dd, color=figs.DOWN, width=1, fill="tozeroy",
                                                                   fillcolor="rgba(239,68,68,0.18)")}, "Drawdown", yfmt=".0%"), 220))
        rb = risk.rolling_beta(pr, bench_l)
        if not rb.empty:
            f = figs.lines(rb.index, {"Rolling beta": dict(y=rb, color=figs.BLUE, width=1.5)}, "Rolling 3-month beta")
            figs.with_marks(f, [figs.hline(1, figs.MUTED)])
            t_perf.append(plot(f, 220))
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max drawdown", "Calmar", "Beta", "Alpha (ann.)",
            "Tracking error", "Information ratio", "Up capture", "Down capture"]
    pk = {"CAGR", "Volatility", "Max drawdown", "Alpha (ann.)", "Tracking error", "Up capture", "Down capture"}
    t_perf.append(kpis([(k, pct(stats[k], 1) if k in pk else num(stats[k])) for k in keys if k in stats]))
    t_perf.append(caption("Backcast: today's weights applied over the lookback window, rebalanced daily. It shows how this "
                          "mix behaves, not your actual historical returns."))

    # ------------------------------------------------ liquidity
    part = float(np.clip(req.participation, 0.05, 0.5))
    liq = risk.liquidity_profile(w, value, pdx.close[avail], pdx.volume[avail], part)
    liq.index = [display_name(t) for t in liq.index]
    worst = liq["Days to exit"].max()
    t_liq = [table(liq.sort_values("Days to exit", ascending=False).reset_index().rename(columns={"index": "Stock"}),
                   [col("Stock", "Stock"), col("Position ₹", "Position", "inr0"), col("20d ADV ₹", "20d ADV", "inr0"),
                    col("% of ADV", "% of ADV", "n3"), col("Days to exit", "Days to exit", "n2")]),
             caption(f"Whole portfolio can be exited in about {max(worst, 0):.2f} trading days at {part:.0%} of average daily "
                     "value traded, without moving the market much.")]
    blocks.append(tabs([("Exposure", t_exp), ("VaR & tails", t_var), ("Risk decomposition", t_dec), ("Stress tests", t_st),
                        ("Performance", t_perf), ("Liquidity", t_liq)]))
    return reply(doc(blocks, pdx.source))


# ------------------------------------------------------------- optimizer


class OptReq(BaseModel):
    tickers: list[str]
    holdings: list[Holding] = []
    objective: str = "max_sharpe"
    mu_method: str = "capm"
    max_w: float = Field(0.25, ge=0.03, le=1)
    min_w: float = Field(0.0, ge=0, le=0.05)
    sector_cap: float | None = None
    lookback: str = "3y"
    capital: float = Field(1_000_000, ge=1000)
    rf: float = 0.065


@router.post("/optimizer")
def optimizer(req: OptReq):
    tickers = list(dict.fromkeys(normalize_symbol(t) for t in req.tickers))[:60]
    if len(tickers) < 2:
        raise ApiError(400, "Pick at least two stocks.")
    if req.objective not in opt.OBJECTIVES:
        raise ApiError(400, "Unknown objective.")
    pdx = prices(tickers + [BENCHMARK], "5y", ttl=900)
    n = {"1y": 252, "2y": 504, "3y": 756, "5y": 1260}.get(req.lookback, 756)
    avail = [t for t in tickers if t in pdx.close]
    rets = risk.returns(pdx.close[avail]).iloc[-n:]
    rets = rets.dropna(axis=1, thresh=int(0.9 * len(rets))).dropna()
    if rets.shape[1] < 2:
        raise ApiError(400, "Not enough overlapping history for these stocks.")
    bench = risk.returns(pdx.close[BENCHMARK]).iloc[-n:] if BENCHMARK in pdx.close else None
    cov = risk.ledoit_wolf_cov(rets)
    mu = opt.expected_returns(rets, bench, req.mu_method if req.mu_method in ("capm", "shrunk", "historical") else "capm",
                              rf=req.rf)
    sectors = {t: sector_of(t) for t in rets.columns}
    kw = dict(mu=mu, cov=cov, rf=req.rf, min_w=req.min_w, max_w=req.max_w, sectors=sectors, sector_cap=req.sector_cap)
    res = opt.optimize(rets, req.objective, **kw)
    eq = opt.optimize(rets, "equal", **kw)
    pr, pr_eq = risk.portfolio_returns(rets, res.weights), risk.portfolio_returns(rets, eq.weights)
    so = risk.performance_stats(pr, bench, req.rf)
    blocks = source_notice(pdx.source, pdx.missing)
    blocks.append(kpis([("Expected return", pct(res.exp_return, 1), f"vs {pct(eq.exp_return, 1)} equal-wt"),
                        ("Ex-ante volatility", pct(res.volatility, 1), f"vs {pct(eq.volatility, 1)} equal-wt"),
                        ("Ex-ante Sharpe", num(res.sharpe), f"vs {num(eq.sharpe)} equal-wt"),
                        ("Holdings", f"{(res.weights > 0).sum()}", f"eff. N {risk.effective_n(res.weights):.1f}"),
                        ("Beta (backcast)", num(so.get("Beta"))), ("Max DD (backcast)", pct(so.get("Max drawdown"), 1), "", "down")]))
    wv = res.weights[res.weights > 0].sort_values(ascending=False)
    rc_ = risk.risk_contributions(cov, res.weights).loc[wv.index]
    lab = [display_name(t) for t in wv.index]
    t_w = [plot(figs.grouped_h(lab, {"Weight": (wv.values, figs.GOLD), "Risk share": (rc_["% of risk"].values, figs.BLUE)},
                               opt.OBJECTIVES[req.objective]), max(300, 30 * len(wv) + 80))]
    sec_w = res.weights.groupby(pd.Series(sectors)).sum().sort_values(ascending=False)
    sec_w = sec_w[sec_w > 0]
    t_w.append(plot(figs.donut(sec_w.index, sec_w.values, "Sector mix"), 320))

    fr = opt.efficient_frontier(rets, mu, cov, n_points=22, max_w=req.max_w, rf=req.rf)
    data = []
    if not fr.empty:
        data.append({"type": "scatter", "mode": "lines", "name": "Efficient frontier", "x": figs._v(fr["Volatility"]),
                     "y": figs._v(fr["Return"]), "line": {"color": figs.GOLD, "width": 2.5}})
    vols = np.sqrt(np.diag(cov.values) * 252)
    data.append({"type": "scatter", "mode": "markers+text", "name": "Stocks", "x": figs._v(vols),
                 "y": figs._v(mu.reindex(rets.columns).values), "text": [display_name(t) for t in rets.columns],
                 "textposition": "top center", "textfont": {"size": 9, "color": figs.MUTED}, "marker": {"color": figs.MUTED, "size": 7}})
    for key, colr, sym in [("max_sharpe", figs.UP, "star"), ("min_vol", figs.BLUE, "diamond"),
                           ("risk_parity", "#E86AF0", "circle"), ("equal", "var(--text)", "x")]:
        r = opt.optimize(rets, key, **kw)
        data.append({"type": "scatter", "mode": "markers", "name": opt.OBJECTIVES[key], "x": [r.volatility], "y": [r.exp_return],
                     "marker": {"color": colr, "size": 14, "symbol": sym, "line": {"width": 1, "color": "var(--panel)"}}})
    if not fr.empty:
        best = fr.loc[fr["Sharpe"].idxmax()]
        xs = np.linspace(0, fr["Volatility"].max() * 1.1, 20)
        data.append({"type": "scatter", "mode": "lines", "name": "Capital market line", "x": figs._v(xs),
                     "y": figs._v(req.rf + best["Sharpe"] * xs), "line": {"color": figs.MUTED, "dash": "dot", "width": 1}})
    t_fr = [plot(figs.fig(data, {"xaxis": {"title": "Volatility (ann.)", "tickformat": ".0%"},
                                 "yaxis": {"title": "Expected return (ann.)", "tickformat": ".0%"}}), 460),
            caption("Frontier respects the per-stock cap (sector cap applies to the chosen portfolio only). Covariance uses "
                    "Ledoit-Wolf shrinkage, which is far more stable than the raw sample estimate.")]

    last_px = pdx.close[rets.columns].ffill().iloc[-1]
    alloc, cash = opt.discrete_allocation(res.weights, last_px, req.capital)
    al = alloc.reset_index().rename(columns={"index": "t"})
    al.insert(0, "Stock", al["t"].map(display_name))
    t_tr = [table(al, [col("Stock", "Stock"), col("Shares", "Shares", "int"), col("Price ₹", "Price", "inr2"),
                       col("Value ₹", "Value", "inr0"), col("Actual weight", "Actual", "pct1"),
                       col("Target weight", "Target", "pct1")]),
            caption(f"Whole shares only. Leftover cash: {inr(cash)}.")]
    hold = {normalize_symbol(h.ticker): h.quantity for h in req.holdings if h.quantity > 0}
    if hold:
        cur = pd.Series(hold)
        tv = float((cur.reindex(last_px.index).fillna(0) * last_px).sum())
        if tv > 0:
            tgt, _ = opt.discrete_allocation(res.weights, last_px, tv)
            trades = opt.rebalance_trades(cur.reindex(rets.columns).fillna(0), tgt)
            if not trades.empty:
                tr = trades.reset_index().rename(columns={"index": "t"})
                tr.insert(0, "Stock", tr["t"].map(display_name))
                t_tr.append(text("**Rebalance from current holdings** (same total value)"))
                t_tr.append(table(tr, [col("Stock", "Stock"), col("Side", "Side", "badge"), col("Current", "Current", "int"),
                                       col("Target", "Target", "int"), col("Trade", "Trade", "int"),
                                       col("Price ₹", "Price", "inr2"), col("Notional ₹", "Notional", "inr0"),
                                       col("Est. cost ₹", "Est. cost", "inr0")]))
                t_tr.append(caption(f"Turnover {inr(trades['Notional ₹'].sum())}, estimated costs "
                                    f"{inr(trades['Est. cost ₹'].sum())} (≈15 bps per side). Selling may also trigger capital "
                                    "gains tax (STCG 20% / LTCG 12.5% above ₹1.25 L)."))
    cum = pd.DataFrame({"Optimised": (1 + pr).cumprod(), "Equal weight": (1 + pr_eq).cumprod()})
    if bench is not None:
        cum["NIFTY 50"] = (1 + bench.reindex(pr.index).fillna(0)).cumprod()
    t_bc = [plot(figs.lines(cum.index, {k: dict(y=cum[k], width=2 if k == "Optimised" else 1.4,
                                                color=[figs.GOLD, figs.BLUE, figs.MUTED][i]) for i, k in enumerate(cum.columns)},
                            "In-sample backcast (growth of ₹1)"), 340),
            callout("warn", "This backcast uses the same data the optimiser saw, so it flatters the result. Judge portfolios "
                            "by their risk profile (volatility, drawdown, concentration), not by this curve.")]
    blocks.append(tabs([("Weights", t_w), ("Efficient frontier", t_fr), ("Trade list", t_tr), ("Backcast", t_bc)]))
    return reply(doc(blocks, pdx.source))


# ------------------------------------------------------------------- lab


@router.get("/lab/backtest")
def lab_backtest(ticker: str = "RELIANCE.NS", period: str = "5y", cost: float = Query(bt.DEFAULT_COST_BPS, ge=0, le=100),
                 strategies: str = ",".join(bt.STRATEGIES)):
    t = normalize_symbol(ticker)
    period = period if period in ("3y", "5y", "10y") else "5y"
    pdx = prices([t, BENCHMARK], period, ttl=1800)
    if t not in pdx.close:
        raise ApiError(404, f"No data for {t}.")
    df, bench = pdx.ohlcv(t), pdx.close.get(BENCHMARK)
    chosen = [s for s in strategies.split(",") if s in bt.STRATEGIES] or list(bt.STRATEGIES)
    results = {k: bt.run(df, k, bench, cost) for k in chosen}
    series = {bt.STRATEGIES[k].split(":")[0]: dict(y=r["equity"], width=2.2 if k == "composite" else 1.4,
                                                   color=figs.SERIES[i % 8]) for i, (k, r) in enumerate(results.items())}
    x = next(iter(results.values()))["equity"].index
    rows = {bt.STRATEGIES[k].split(":")[0]: {"CAGR": r["stats"].get("CAGR"), "Volatility": r["stats"].get("Volatility"),
                                             "Sharpe": r["stats"].get("Sharpe"), "Max DD": r["stats"].get("Max drawdown"),
                                             "Calmar": r["stats"].get("Calmar"), "Trades": r["stats"].get("Trades"),
                                             "Win rate": r["stats"].get("Win rate"), "Time in mkt": r["stats"].get("Time in market"),
                                             "Costs": r["stats"].get("Costs paid")} for k, r in results.items()}
    blocks = source_notice(pdx.source)
    blocks.append(plot(figs.lines(x, series, f"{display_name(t)}: growth of ₹1, net of costs", ylog=True), 380))
    blocks.append(table(pd.DataFrame(rows).T.reset_index().rename(columns={"index": "Strategy"}),
                        [col("Strategy", "Strategy"), col("CAGR", "CAGR", "spct1"), col("Volatility", "Vol", "pct0"),
                         col("Sharpe", "Sharpe", "n2"), col("Max DD", "Max DD", "pct0"), col("Calmar", "Calmar", "n2"),
                         col("Trades", "Trades", "int"), col("Win rate", "Win rate", "pct0"),
                         col("Time in mkt", "In market", "pct0"), col("Costs", "Costs", "pct1")]))
    blocks.append(caption("Signals use the close of day t and trade on day t+1 (no look-ahead). Long-only, delivery. "
                          "Costs are charged on every entry and exit."))
    return reply(doc(blocks, pdx.source), ttl=900)


@router.get("/lab/ml")
def lab_ml(tickers: str, hz: int = Query(5, ge=1, le=20), relative: int = 0, folds: int = Query(5, ge=3, le=6)):
    from jinni import ml as jml  # lazy: pulls in scikit-learn, only this endpoint needs it

    tk = tick_list(tickers, 10)
    if len(tk) < 1:
        raise ApiError(400, "Pick at least one stock.")
    pdx = prices(tk + [BENCHMARK], "10y", ttl=1800)
    dfs = {t: pdx.ohlcv(t) for t in tk if t in pdx.close}
    try:
        res = jml.walk_forward(dfs, pdx.close.get(BENCHMARK), horizon=hz, n_splits=folds, relative=bool(relative))
    except ValueError as e:
        raise ApiError(400, str(e)) from e
    sm = res.summary
    blocks = source_notice(pdx.source)
    if pdx.source == "demo":
        blocks.append(callout("info", "Synthetic prices are random, so ~0 edge is the correct result here."))
    blocks.append(kpis([("Out-of-sample accuracy", pct(sm["accuracy"], 1), f"{sm['n_oos']:,} predictions"),
                        ("Base rate", pct(sm["base_rate"], 1), "always guess majority"),
                        ("Edge", pct(sm["edge"], 1, True), "accuracy − base", tone(sm["edge"])),
                        ("AUC", num(sm["auc"]), "0.5 = no skill"), ("Brier score", num(sm["brier"], 3), "lower is better")]))
    blocks.append(table(res.folds.reset_index(), [col("Fold", "Fold", "int"), col("Test from", "From", "date"),
                                                  col("Test to", "To", "date"), col("Accuracy", "Accuracy", "pct1"),
                                                  col("Base rate", "Base", "pct1"), col("Edge vs base", "Edge", "spct1"),
                                                  col("AUC", "AUC", "n3"), col("Precision (up)", "Precision", "pct0")]))
    bk = sm["buckets"]
    imp = res.importance.head(10).iloc[::-1]
    left = [plot(figs.bar_v([str(i) for i in bk.index], bk["Avg_fwd_ret"].values,
                            f"Avg {hz}-day return by predicted-probability quintile", ".1%",
                            texts=[f"{v:.2%}" for v in bk["Avg_fwd_ret"]]), 320),
            caption("A useful model shows returns rising from Q1 to Q5.")]
    right = [plot(figs.bar_h(imp.index, imp.values, "Top features (permutation importance, AUC drop)", ".3f",
                             [figs.GOLD] * len(imp)), 320)]
    blocks.append(cols(left, right))
    lp = res.latest.copy()
    lp.index = [display_name(t) for t in lp.index]
    lp["Signal"] = np.where(lp["prob_up"] >= 0.55, "Bullish", np.where(lp["prob_up"] <= 0.45, "Bearish", "Neutral"))
    blocks.append(section("Live probabilities (final model, trained on all data)"))
    blocks.append(table(lp.sort_values("prob_up", ascending=False).reset_index().rename(columns={"ticker": "Stock", "index": "Stock"}),
                        [col("Stock", "Stock"), col("prob_up", f"P(up in {hz}d)", "bar01"), col("as_of", "As of", "date"),
                         col("Signal", "Signal", "badge")]))
    if sm["edge"] < 0.01:
        blocks.append(callout("info", "The model shows little or no edge over the base rate on this setup. Treat its "
                                      "probabilities as noise. This is normal for short-horizon direction prediction."))
    return reply(doc(blocks, pdx.source), ttl=900)


# --------------------------------------------------------------- journal


class Pred(BaseModel):
    id: int | str
    ticker: str
    entry_date: str
    entry: float
    target_price: float
    stop_price: float | None = None
    target_pct: float
    horizon: int
    p_target: float = 0.0
    grade: str = ""
    qty: int = 0
    source: str = "pick"


class JournalReq(BaseModel):
    predictions: list[Pred]


@router.post("/journal/score")
def journal_score(req: JournalReq):
    if not req.predictions:
        return reply(doc([callout("info", "Nothing tracked yet. In Return Finder, use **Track top 5** or **Track this plan**.")]))
    preds = req.predictions[:200]
    tk = list(dict.fromkeys(normalize_symbol(p.ticker) for p in preds))
    pdx = prices(tk, "1y", ttl=900)
    rows = []
    for p in preds:
        t = normalize_symbol(p.ticker)
        if t not in pdx.close:
            ev = {"status": "No data", "exit_date": None, "ret": np.nan, "days": 0}
        else:
            ev = fc.evaluate_tracked(pdx.ohlcv(t), p.entry_date, p.entry, p.target_price, p.stop_price, p.horizon)
        rows.append({"id": p.id, "Stock": display_name(t), "Status": ev["status"], "Return": ev["ret"],
                     "Predicted": p.p_target, "Goal": f"+{p.target_pct:.0%}", "Days": p.horizon, "Grade": p.grade,
                     "From": p.entry_date, "Days held": ev["days"], "Entry ₹": p.entry, "Target ₹": p.target_price,
                     "Stop ₹": p.stop_price, "Qty": p.qty, "Type": "Plan" if p.source == "plan" else "Pick"})
    j = pd.DataFrame(rows)
    closed = j[j["Status"].isin(["Target hit", "Stop hit", "Expired"])]
    hit = closed["Status"] == "Target hit"
    pnl = float((j["Return"] * j["Entry ₹"] * j["Qty"]).where(j["Type"] == "Plan").sum())
    blocks = source_notice(pdx.source)
    blocks.append(kpis([("Tracked", f"{len(j)}", f"{(j['Status'] == 'Open').sum()} still open"),
                        ("Resolved", f"{len(closed)}", "target, stop or expired"),
                        ("Target hit rate", pct(hit.mean(), 0) if len(closed) else "—", "actual", "up"),
                        ("Predicted (avg)", pct(closed["Predicted"].mean(), 0) if len(closed) else "—", "what JINNI said"),
                        ("Avg return", pct(closed["Return"].mean(), 2, True) if len(closed) else "—", "per resolved trade",
                         tone(closed["Return"].mean()) if len(closed) else ""),
                        ("Plan P&L", inr(pnl), "tracked plans, open + closed", tone(pnl))]))
    if 0 < len(closed) < 30:
        blocks.append(caption(f"Only {len(closed)} resolved predictions so far. Accuracy needs about 30+ before it means much."))
    blocks.append(table(j, [col("Stock", "Stock"), col("Status", "Status", "badge"), col("Return", "Return", "spct2"),
                            col("Predicted", "P(target)", "bar01"), col("Goal", "Goal"), col("Days", "Days", "int"),
                            col("Grade", "Grade", "grade"), col("From", "From", "date"), col("Entry ₹", "Entry", "inr2"),
                            col("Target ₹", "Target", "inr2"), col("Stop ₹", "Stop", "inr2"), col("Qty", "Qty", "int"),
                            col("Type", "Type")], select={"key": "id", "action": "journal-row"}))
    if len(closed) >= 5:
        b = closed.assign(bucket=pd.cut(closed["Predicted"], [0, 0.3, 0.45, 0.6, 0.75, 1.0]))
        cal = b.groupby("bucket", observed=True).agg(Predicted=("Predicted", "mean"),
                                                     Actual=("Status", lambda s: (s == "Target hit").mean()),
                                                     Count=("Status", "size"))
        f = figs.bar_v([str(i) for i in cal.index], cal["Actual"].values, "When JINNI said X%, how often did it happen?", ".0%",
                       [figs.BLUE] * len(cal), [f"n={c}" for c in cal["Count"]])
        f["data"].append({"type": "scatter", "mode": "lines+markers", "name": "Predicted", "x": [str(i) for i in cal.index],
                          "y": figs._v(cal["Predicted"]), "line": {"color": figs.GOLD}})
        f["layout"]["yaxis"]["range"] = [0, 1]
        by_g = closed.groupby("Grade").agg(Trades=("Status", "size"), Hit=("Status", lambda s: (s == "Target hit").mean()),
                                           Avg_return=("Return", "mean")).reset_index()
        blocks.append(section("Calibration"))
        blocks.append(plot(f, 300))
        blocks.append(table(by_g, [col("Grade", "Grade", "grade"), col("Trades", "Trades", "int"), col("Hit", "Hit rate", "pct0"),
                                   col("Avg_return", "Avg return", "spct2")]))
    return reply(doc(blocks, pdx.source))
