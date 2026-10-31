"""Stock selection: validated factor ranking, risk-aware basket, whole-share plan for a budget."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from jinni import data as jdata
from jinni import live, market, risk, selection as sel, signals
from jinni import optimizer as opt
from jinni.universe import BENCHMARK, NIFTY50, TRADING_DAYS, display_name, sector_of

from . import figs
from .core import (ApiError, DataUnavailable, caption, callout, col, cols, doc, kpis, memo, pct, plot, reply, section,
                   source_notice, table, text, tone)

router = APIRouter()
PROFILES = {"conservative": "Conservative (minimum volatility)", "balanced": "Balanced (risk parity)",
            "aggressive": "Aggressive (tilted to the highest scores)"}


def _universe(n: int) -> list[str]:
    if jdata.demo_forced():
        return market.demo_universe(n)["ticker"].tolist()[:n]
    top = live.top_by_cap(n)
    if top is None:
        raise DataUnavailable("The live list of NSE stocks (by market cap) could not be loaded right now.")
    return top["t"].tolist()


def _panel(tickers: list[str]):
    def build():
        pdx = market.bulk_prices(tickers, "5y", chunk=60)
        bench = market.bulk_prices([BENCHMARK], "5y", chunk=5).close[BENCHMARK].dropna()
        return pdx, bench
    return memo(("sel-panel", hash(tuple(tickers))), 3600, build)


def _fund(t: str):
    info = memo(("info", t), 3600, lambda: jdata.fetch_info(t))
    return signals.fundamental_score(info)[0]


@router.get("/select")
def select(budget: float = Query(100000, ge=1000), k: int = Query(8, ge=3, le=20), n: int = Query(120, ge=40, le=250),
           h: int = Query(21, ge=5, le=63), profile: str = "balanced", min_val: float = Query(1.0, ge=0),
           min_px: float = Query(20, ge=1)):
    if profile not in PROFILES:
        raise ApiError(422, f"profile must be one of {', '.join(PROFILES)}")
    tickers = _universe(n)
    pdx, bench = _panel(tickers)
    close = pdx.close.loc[:, pdx.close.notna().sum() > 400]
    if close.shape[1] < 40:
        raise DataUnavailable("Too few stocks returned price history to run the selection model.")
    v = memo(("sel-val", hash(tuple(close.columns)), h, str(close.index[-1])), 3600, lambda: sel.validate(close, bench, h))
    st = v.stats
    blocks = source_notice(pdx.source, pdx.missing)

    beat = st["ret_sel"] > st["ret_uni"] and st["excess_t"] >= 1.5
    msg = (f"In the walk-forward test ({st['periods']} periods, {st['from']:%b %Y} to {st['to']:%b %Y}) the selection returned "
           f"{st['ret_sel']:+.1%}/yr after costs vs {st['ret_uni']:+.1%}/yr for the whole universe (excess t-stat {st['excess_t']:.1f}). ")
    blocks.append(callout("ok" if beat else "warn", msg + (
        "That is a statistically meaningful edge, but the universe is today's listed stocks, so survivorship bias flatters it." if beat else
        "That is **not** a reliable edge over simply holding the universe, so treat these picks as a tilt, not a signal.")))
    blocks.append(kpis([
        ("Universe", f"{close.shape[1]} stocks", f"largest {n} NSE by market cap"),
        ("Net return / yr", pct(st["ret_sel"], 1, True), f"universe {pct(st['ret_uni'], 1, True)}", tone(st["ret_sel"] - st["ret_uni"])),
        ("Sharpe", f"{st['sharpe_sel']:.2f}", f"universe {st['sharpe_uni']:.2f}"),
        ("Beat universe", pct(st["hit"], 0), f"of {st['periods']} periods"),
        ("Rank IC", f"{st['ic']:+.3f}", f"t = {st['ic_t']:.1f}")]))

    scores = sel.live_scores(v, close)
    last = close.iloc[-1]
    value_cr = (pdx.close * pdx.volume).iloc[-20:].median() / 1e7
    stale = close.iloc[-5:].isna().all()
    ok = scores.index[(last.reindex(scores.index) >= min_px) & (value_cr.reindex(scores.index) >= min_val)
                      & ~stale.reindex(scores.index).fillna(True)]
    pool = scores.loc[ok].sort_values("Score", ascending=False)
    if len(pool) < k:
        raise ApiError(422, "Fewer stocks than requested pass the price and liquidity filters. Lower the minimum traded value.")
    cand = pool.head(min(len(pool), k * 3))
    with ThreadPoolExecutor(max_workers=8) as ex:
        fs = dict(zip(cand.index, ex.map(_fund, cand.index)))
    cand = cand.assign(Quality=pd.Series(fs))
    cand = cand[~(cand["Quality"] < 30)]  # weak balance sheet / profitability: drop when we have data
    picks = cand.head(k)
    if len(picks) < 3:
        raise ApiError(422, "Not enough candidates survive the quality screen.")
    tk = list(picks.index)

    rets = risk.returns(close[tk].iloc[-504:]).dropna(how="all").fillna(0)
    sectors = {t: (sector_of(t) if t in NIFTY50 and sector_of(t) != "Other" else f"?{t}") for t in tk}
    maxw = 0.3 if profile == "aggressive" else 0.25
    if profile == "aggressive":
        w = sel.tilt_weights(picks["Score"], maxw)
    else:
        w = opt.optimize(rets, "min_vol" if profile == "conservative" else "risk_parity", max_w=maxw,
                         sectors=sectors, sector_cap=0.4).weights
    alloc, cash = opt.discrete_allocation(w, last[tk], budget)
    invested = float(alloc["Value ₹"].sum())
    wt = alloc["Actual weight"] * budget / max(invested, 1)  # weights of what is actually held
    pr = risk.portfolio_returns(rets, wt.reindex(tk).fillna(0))
    cov = risk.ledoit_wolf_cov(rets)
    rc = risk.risk_contributions(cov, wt.reindex(tk).fillna(0))
    vr = risk.var_report(rets, wt.reindex(tk).fillna(0), invested, 0.95, 1, n_sims=4000)
    b = risk.returns(bench.iloc[-504:]).reindex(pr.index).fillna(0)
    beta = float(np.cov(pr, b)[0, 1] / b.var()) if b.var() > 0 else np.nan
    vol_ann = float(pr.std() * np.sqrt(TRADING_DAYS))

    blocks.append(section(f"Your ₹{budget:,.0f} basket · {PROFILES[profile]}"))
    rows = []
    for t in tk:
        if t not in alloc.index:
            continue
        a = alloc.loc[t]
        if int(a["Shares"]) < 1:
            continue  # the optimiser gave it no weight (or the budget cannot buy a share)
        atr = float((pdx.high[t] - pdx.low[t]).iloc[-14:].mean())
        rows.append({"t": t, "Stock": display_name(t), "Rating": sel.rating(picks.loc[t, "Percentile"]), "Score": picks.loc[t, "Score"],
                     "Shares": int(a["Shares"]), "Price": float(a["Price ₹"]), "Value": float(a["Value ₹"]),
                     "Weight": float(a["Actual weight"]), "Risk share": float(rc.loc[t].iloc[0]) if t in rc.index else np.nan,
                     "Stop": float(a["Price ₹"] - 2 * atr), "Quality": picks.loc[t, "Quality"],
                     "Why": sel.drivers_text(picks.loc[t])})
    blocks.append(table(pd.DataFrame(rows), [
        col("Stock", "Stock"), col("Rating", "Rating", "badge"), col("Shares", "Shares", "int"), col("Price", "Price ₹", "inr2"),
        col("Value", "Invest ₹", "inr0"), col("Weight", "Weight", "pct1"), col("Risk share", "Risk share", "pct0", help="Share of the basket's risk this stock contributes"),
        col("Stop", "Stop ₹ (2×ATR)", "inr2"), col("Score", "Score", "n2"), col("Quality", "Quality 0-100", "n0"),
        col("Why", "Strongest signals")], select={"key": "t", "action": "analyze"}))
    blocks.append(kpis([
        ("Invested", f"₹{invested:,.0f}", f"₹{cash:,.0f} cash left"),
        ("Basket volatility", pct(vol_ann, 1), "annualised, last 2y"),
        ("Beta to NIFTY", f"{beta:.2f}", ""),
        ("1-day VaR (95%)", f"₹{vr.table['VaR ₹'].iloc[0]:,.0f}", f"{pct(vr.table['VaR %'].iloc[0], 2)} of the basket"),
        ("Effective positions", f"{risk.effective_n(wt):.1f}", f"of {len(wt)} held")]))
    mix = pd.Series({sectors[t]: float(wt[t]) for t in wt.index if t in sectors}).rename(lambda s: "Other / unclassified" if s.startswith("?") else s)
    mix = mix.groupby(level=0).sum()
    blocks.append(cols([plot(figs.donut(list(mix.index), list(mix.values), "Sector mix"), 300)],
                       [plot(figs.lines(v.curve.index, {"Selection (net of costs)": dict(y=v.curve["Selection"], color=figs.GOLD, width=2.2),
                                                        "Whole universe": dict(y=v.curve["Universe"], color=figs.MUTED, width=1.6)},
                                        "Walk-forward growth of ₹1"), 300)]))

    worst = pool.tail(6).iloc[::-1]
    blocks.append(section("Weakest signals right now (avoid or reduce)"))
    blocks.append(table(pd.DataFrame({"t": worst.index, "Stock": [display_name(t) for t in worst.index], "Score": worst["Score"].values,
                                      "Price": last[worst.index].values, "Why": [sel.drivers_text(r) for _, r in worst.iterrows()]}),
                        [col("Stock", "Stock"), col("Score", "Score", "n2"), col("Price", "Price ₹", "inr2"), col("Why", "Strongest signal")],
                        select={"key": "t", "action": "analyze"}))

    blocks.append(section("Which signals have actually worked (and how they are weighted)"))
    ft = v.factor_table
    blocks.append(table(ft, [col("Factor", "Factor"), col("Mean IC", "Mean IC", "n3"), col("t-stat", "t-stat", "n1"),
                             col("Worked %", "Worked %", "pct0"), col("Weight now", "Weight now", "pct0"),
                             col("What it measures", "What it measures")]))
    if v.weights_now.sum() == 0:
        blocks.append(callout("warn", "No factor has shown a positive information coefficient over the last two years, so scores use "
                                      "equal weights and carry little conviction."))
    blocks.append(caption(
        "IC = rank correlation between a signal and the stock's next-period return; weights come only from the previous 24 "
        "months of IC at each rebalance, so the backtest never peeks. Costs: 15 bps per side on turnover. Survivorship bias: "
        "stocks that were delisted or collapsed are not in today's list. Quality (ROE, debt, margins, P/E) only screens the "
        "final names because free data has no point-in-time fundamentals history. This is analysis, not advice."))
    return reply(doc(blocks, pdx.source, picks=tk, budget=budget), ttl=900)
