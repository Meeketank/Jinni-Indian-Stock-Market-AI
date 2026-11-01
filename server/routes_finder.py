"""Return Finder: whole-market pre-scan (chunked), deep analysis, budget plans and per-stock detail."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from jinni import data as jdata
from jinni import events as jevents
from jinni import forecast as fc
from jinni import market, risk, setups, signals
from jinni import riskcheck as rc
from jinni.universe import BENCHMARK, NIFTY50, display_name, normalize_symbol, sector_of

from . import figs
from .core import (ApiError, DataUnavailable, caption, callout, col, cols, doc, kpis, memo, news_list, pct, plot, prices, reply,
                   section, source_notice, table, tabs, text, tone)
from .routes_stock import ctx as stock_ctx, news_of

router = APIRouter()
DEADLINE_S = 42  # stay well inside the function's 60 s limit


# ------------------------------------------------------- market pre-scan


@router.get("/market/universe")
def universe(sme: int = 0, min_val: float = Query(1.0, ge=0), min_px: float = Query(20, ge=0)):
    """Every liquid listed stock on NSE and BSE (cross-listings once). Falls back to the NSE list alone if the
    exchanges' end-of-day files are unreachable."""
    from jinni import bhav

    def build():
        if not jdata.demo_forced():
            df, info = bhav.latest()
            if len(df):
                keep = df[(df["turnover"] >= min_val * 1e7) & (df["close"] >= min_px)].sort_values("turnover", ascending=False)
                src = f"{info['source']} ({info['date']:%d %b %Y}): {len(df):,} listed, {len(keep):,} trade ≥ ₹{min_val:g} cr and ≥ ₹{min_px:g}"
                return keep["ticker"].tolist(), keep["name"].tolist(), keep["exch"].tolist(), len(df), src
        try:
            u = market.demo_universe() if jdata.demo_forced() else market.nse_universe(include_sme=bool(sme))
        except RuntimeError as e:
            raise DataUnavailable(str(e)) from e
        return u["ticker"].tolist(), u["name"].tolist(), ["NSE"] * len(u), len(u), u.attrs.get("source", "")

    tk, nm, ex, listed, src = memo(("universe", int(sme), float(min_val), float(min_px)), 3600, build)
    return reply({"source": src, "count": len(tk), "listed": listed, "rows": [[a, b, c] for a, b, c in zip(tk, nm, ex)]}, ttl=1800)


@router.get("/market/scan")
def scan_chunk(tickers: str = Query(..., max_length=3000), target: float = Query(0.05, ge=0.005, le=1),
               hz: int = Query(21, ge=2, le=250), budget: float = Query(0, ge=0), min_px: float = 20,
               min_val: float = 1.0):
    """Stage 1 for one chunk of tickers: affordability, liquidity, data quality, quick odds."""
    tk = [t for t in dict.fromkeys(x.strip() for x in tickers.split(",")) if t][:130]
    if not tk:
        raise ApiError(400, "No tickers.")
    pdx = prices(tk, "1y", ttl=3600)
    pf = market.prefilter(pdx, budget or None, target, hz, min_px, min_val)
    rows = []
    for t, r in pf.iterrows():
        rows.append({"t": t, "price": r["Price"], "vol": r["Daily vol"], "val": r["Traded value ₹Cr"],
                     "q": r["Quick P(touch)"], "why": r["Excluded because"]})
    return reply({"rows": rows, "missing": pdx.missing, "source": pdx.source}, ttl=1800)


# ------------------------------------------------------------- deep finder


def _stop_arg(stop: str):
    s = (stop or "auto").lower()
    if s == "auto":
        return "auto", None
    if s == "none":
        return "none", None
    try:
        v = float(s)
        if 0.003 <= v <= 0.5:
            return "fixed", v
    except ValueError:
        pass
    raise ApiError(400, "stop must be 'auto', 'none' or a fraction like 0.03")


def run_finder(tickers: list[str], hz: int, target: float, mode: str, fixed: float | None, use_ml: bool):
    pdx = prices(tickers + [BENCHMARK], "5y", ttl=900)
    bench = pdx.close.get(BENCHMARK)
    dfs = {t: pdx.ohlcv(t) for t in tickers if t in pdx.close and pdx.close[t].notna().sum() > 300}
    stops = {t: (fc.auto_stop_pct(d, hz, target) if mode == "auto" else fixed if mode == "fixed" else None)
             for t, d in dfs.items()}
    ml, ml_err = None, None
    t0 = time.time()
    if use_ml and len(dfs) >= 3:
        try:
            ml = fc.ml_hit_model(dfs, bench, hz, target, stops if mode == "auto" else fixed)
        except ValueError as e:
            ml_err = str(e)
    out, cut = {}, False
    for t, d in dfs.items():
        if time.time() > t0 + DEADLINE_S - 8 and out:
            cut = True
            break
        try:
            out[t] = fc.forecast_stock(t, d, bench, hz, target, stops[t], ml, n_paths=3000)
        except ValueError:
            continue
    act = {t: setups.active_setups(d) for t, d in dfs.items() if t in out}
    base = float(np.nanmean([f.hist["all"]["p_target"] for f in out.values()])) if out else float("nan")
    return out, act, ml, ml_err, base, pdx, dfs, cut


GRADE_COLS = [col("Stock", "Stock"), col("P(touch target)", "Chance to reach target", "bar01",
                                          help="Probability the price touches the target at any point in the window (no stop). "
                                               "Driven mostly by how volatile the stock is."),
              col("P(target first)", "Target before stop", "pct0", help="Reaches the target before hitting the stop"),
              col("P(stop first)", "Stop hit first", "pct0"),
              col("Exp. return", "Avg return / trade", "spct2",
                  help="Average result per trade after stop-outs and ~0.3% costs. Much smaller than the target on purpose."),
              col("Grade", "Grade", "grade"),
              col("Tech", "Technical", "n0", help="JINNI technical score 0-100 (trend, momentum, timing, volume, risk)"),
              col("Quality", "Fundamentals", "n0", help="0-100 from ROE, debt, margins, growth, P/E; blank if unavailable"),
              col("News", "News tone", "tone", help="Recent headline tone for this stock, -1 to +1"),
              col("Flags", "Watch-outs"),
              col("Price", "Price", "inr2"), col("Target ₹", "Target ₹", "inr2"), col("Stop ₹", "Stop ₹", "inr2"),
              col("Stop %", "Stop %", "pct1"), col("Typical days", "Days", "n0", help="Median days to target when hit"),
              col("Verdict", "Verdict"), col("Setups", "Setups")]


def _enrich(tbl: pd.DataFrame, dfs: dict, bench, top: int = 15, budget_s: float = 9.0) -> pd.DataFrame:
    """Technical score for every analysed stock; fundamentals and news tone for the best ``top`` (network calls)."""
    tech = {}
    for t in tbl["ticker"]:
        try:
            tech[t] = float(signals.factor_scores(dfs[t], bench)["Composite"].iloc[-1])
        except Exception:  # noqa: BLE001
            tech[t] = np.nan
    tbl = tbl.assign(Tech=tbl["ticker"].map(tech))
    best = tbl.sort_values("P(touch target)", ascending=False).head(top)["ticker"].tolist()

    def one(t):
        info = memo(("info", t), 3600, lambda: jdata.fetch_info(t))
        q = signals.fundamental_score(info)[0]
        n = memo(("news", t), 600, lambda: jevents.headlines(t, limit=12))
        tn = sm_score(n)
        return t, q, tn

    # Fundamentals and news are network calls: give them a fixed time budget and leave blanks rather than overrun.
    from concurrent.futures import wait

    ex = ThreadPoolExecutor(max_workers=10)
    futs = [ex.submit(one, t) for t in best]
    done, _ = wait(futs, timeout=budget_s)
    ex.shutdown(wait=False, cancel_futures=True)
    got = []
    for f in done:
        try:
            got.append(f.result())
        except Exception:  # noqa: BLE001
            pass
    qd = {t: q for t, q, _ in got}
    nd = {t: n for t, _, n in got}
    tbl = tbl.assign(Quality=tbl["ticker"].map(qd), News=tbl["ticker"].map(nd))
    rsi_ = {t: float(__import__("jinni.indicators", fromlist=["rsi"]).rsi(dfs[t]["Close"]).iloc[-1]) for t in tbl["ticker"]}
    flags = []
    for r in tbl.itertuples():
        f = []
        if r.Quality is not None and np.isfinite(r.Quality) and r.Quality < 35:
            f.append("weak fundamentals")
        if r.News is not None and np.isfinite(r.News) and r.News < -0.25:
            f.append("negative news")
        if rsi_.get(r.ticker, 50) > 75:
            f.append("overbought")
        if r._asdict().get("Tech") is not None and np.isfinite(r.Tech) and r.Tech < 30:
            f.append("weak trend")
        flags.append(", ".join(f))
    return tbl.assign(Flags=flags)


def sm_score(news: pd.DataFrame) -> float:
    from jinni import sentiment as sm

    s = sm.score_news(news)
    return s["score"]


@router.get("/finder")
def finder(target: float = Query(0.05, ge=0.005, le=1), hz: int = Query(21, ge=2, le=250), stop: str = "auto",
           tickers: str = Query(..., max_length=2500), ml: int = 1, budget: float = Query(0, ge=0),
           max_names: int = Query(3, ge=1, le=5)):
    from jinni import bhav

    memo(("bhav",), 900, bhav.latest)  # fills company names for BSE scrip codes
    tk = [normalize_symbol(x) for x in tickers.replace(" ", ",").split(",") if x.strip()]
    tk = list(dict.fromkeys(tk))[:70]
    if not tk:
        raise ApiError(400, "No stocks to search.")
    mode, fixed = _stop_arg(stop)
    res, act, mlr, ml_err, base, pdx, dfs, cut = run_finder(tk, hz, target, mode, fixed, bool(ml))
    if not res:
        return reply(doc([callout("error", "No stock had enough history (about 15 months needed).")]), ttl=60)
    rows = []
    for t, f in res.items():
        rows.append({"ticker": t, "Stock": display_name(t), "Grade": f.grade, "P(target first)": f.p_target,
                     "P(stop first)": f.p_stop, "P(touch target)": f.p_touch, "Exp. return": f.ev, "Price": f.price,
                     "Target ₹": f.target_price, "Stop ₹": f.stop_price, "Stop %": f.stop_pct,
                     "Typical days": f.days_to_target, "Verdict": f.verdict, "Setups": ", ".join(act.get(t, [])),
                     "Sector": sector_of(t), "as_of": f.as_of, "engines": f.engines, "p_target": f.p_target,
                     "notes": f.notes})
    tbl = pd.DataFrame(rows).sort_values("P(touch target)", ascending=False)
    tbl = _enrich(tbl, dfs, base_bench(pdx))
    gc = tbl["Grade"].value_counts()
    hz_label = next((k for k, v in fc.HORIZONS.items() if v == hz), f"{hz} days")
    stop_txt = "auto stop" if mode == "auto" else f"−{fixed:.1%} stop" if fixed else "no stop"
    blocks = source_notice(pdx.source)
    if cut:
        blocks.append(callout("warn", f"Stopped early to stay within the time limit: {len(res)} of {len(dfs)} stocks "
                                      "analysed. Try fewer stocks."))
    blocks.append(text(f"#### Goal: **+{target:.0%} within {hz_label}** · {stop_txt}"))
    best = tbl.loc[tbl["P(touch target)"].idxmax()]
    likely = int((tbl["P(touch target)"] >= 0.5).sum())
    blocks.append(kpis([
        ("Stocks analysed", f"{len(tbl)}", f"{len(tk) - len(tbl)} skipped" if len(tk) > len(tbl) else ""),
        (f"Likely to reach +{target:.0%}", f"{likely}", "≥ 50% chance within the window", "up" if likely else None),
        ("Best chance", pct(best["P(touch target)"], 0), best["Stock"]),
        ("Typical stock", pct(base, 0), "historical hit rate for this goal"),
        ("Grade A / B", f"{gc.get('A', 0)} / {gc.get('B', 0)}", "positive average return per trade after stops", "up"),
        ("ML engine", f"AUC {mlr.auc:.2f}" if mlr else "off",
         (f"weight {mlr.weight:.0%}" if mlr.weight > 0 else "no skill → 0 weight") if mlr else (ml_err or ""))]))
    blocks.append(callout("info", "**How to read this.** “Chance to reach target” is how often the price touches the goal inside the "
                                  "window; it rises with volatility, so the top of the list is the most active stocks. "
                                  "“Target before stop” and “Avg return / trade” are lower because stops get hit and costs apply: "
                                  "an average of well under 1% per trade is normal even when the chance of touching +5% is above 50%. "
                                  "In a walk-forward test on real NSE stocks, the probability engine's rankings were useful "
                                  "(AUC about 0.6 to 0.7) while technical scores and momentum added nothing, so they are shown as "
                                  "watch-outs, not used to rank. See Lab → System accuracy."))
    blocks.append(table(tbl, GRADE_COLS, select={"key": "ticker", "action": "detail"},
                        height=min(560, 44 + 38 * len(tbl))))

    plans = {}
    if budget and budget > 0:
        blocks += budget_plan(tbl, res, dfs, budget, max_names, target, hz, hz_label, plans)
    payload = tbl[["ticker", "Grade", "Price", "Target ₹", "Stop ₹", "Stop %", "as_of", "p_target", "P(stop first)",
                   "Verdict", "engines", "Exp. return"]].to_dict("records")
    return reply(doc(blocks, pdx.source, rows=payload, plans=plans, hz=hz, target=target, stop=stop), ttl=300)


def budget_plan(tbl, res, dfs, budget, max_names, target, hz, hz_label, plans_out) -> list:
    blocks = [section(f"Your ₹{budget:,.0f} plan: +{target:.0%} within {hz_label}")]
    ranked = tbl[tbl["Price"] <= budget].sort_values("Exp. return", ascending=False)
    if ranked.empty:
        blocks.append(callout("warn", "None of the analysed stocks costs less than your budget per share."))
        return blocks
    good = ranked[ranked["Grade"].isin(["A", "B"])]
    if good.empty:
        blocks.append(callout("warn", f"Honest answer: no stock shows a positive expected return for +{target:.0%} within "
                                      f"{hz_label} after costs. The plan below uses the best of what's available, but the "
                                      "odds are against you. A smaller target or a longer window usually helps more than a "
                                      "different stock."))
    pool = good if not good.empty else ranked
    options = {}
    for nm, k in (("Diversified basket", max_names), ("Single best pick", 1)):
        al = fc.allocate_budget(pool, budget, k)
        if al.empty:
            continue
        try:
            sim = fc.simulate_basket(dfs, dict(zip(al["ticker"], al["Shares"])), hz, target,
                                     {t: res[t].stop_pct for t in al["ticker"]}, n_paths=3000)
        except ValueError:
            continue
        options[nm] = (al, sim)
    if not options:
        return blocks
    cmp_rows = [{"Plan": nm, "Stocks": len(al), "Invested ₹": s["value"], "Cash left ₹": al.attrs.get("cash", 0),
                 "P(reach)": s["p_reach"], "P(loss)": s["p_loss"], "Expected ₹": s["exp"] * s["value"],
                 "Bad": s["p5"] * s["value"], "Good": s["p95"] * s["value"]} for nm, (al, s) in options.items()]
    blocks.append(table(pd.DataFrame(cmp_rows), [
        col("Plan", "Plan"), col("Stocks", "Stocks", "int"), col("Invested ₹", "Invested", "inr0"),
        col("Cash left ₹", "Cash left", "inr0"), col("P(reach)", f"P(≥ +{target:.0%})", "pct0"),
        col("P(loss)", "P(loss)", "pct0"), col("Expected ₹", "Expected ₹", "inr0"),
        col("Bad", "Bad case (5%)", "inr0"), col("Good", "Good case (95%)", "inr0")]))
    for nm, (al, sim) in options.items():
        t_ = al.assign(Stock=al["ticker"].map(display_name), Grade=al["ticker"].map(lambda t: res[t].grade),
                       **{"P(target)": al["ticker"].map(lambda t: res[t].p_target),
                          "Target ₹": al["ticker"].map(lambda t: res[t].target_price),
                          "Stop ₹": al["ticker"].map(lambda t: res[t].stop_price)})
        plans_out[nm] = [{"ticker": r.ticker, "shares": int(r.Shares), "price": float(r.Price)} for r in al.itertuples()]
        hist = figs.histogram(sim["returns"] * sim["value"], 50, f"{nm}: simulated profit/loss in ₹ after {hz} trading days",
                              figs.BLUE, [figs.vline(target * sim["value"], figs.UP, "dash", "Your target"),
                                          figs.vline(0, figs.MUTED, "dot")])
        blocks.append(text(f"**{nm}**"))
        blocks.append(table(t_, [col("Stock", "Stock"), col("Grade", "Grade", "grade"), col("Shares", "Shares", "int"),
                                 col("Price", "Price", "inr2"), col("Invest ₹", "Invest ₹", "inr0"),
                                 col("P(target)", "P(target)", "bar01"), col("Target ₹", "Target ₹", "inr2"),
                                 col("Stop ₹", "Stop ₹", "inr2")]))
        blocks.append(plot(hist, 260))
    blocks.append(caption("Stocks in a basket are simulated on the same historical days, so they rise and fall together "
                          "as they really do. Costs (~0.3% round trip) included. Small budgets: brokerage minimums can be "
                          "a bigger share of a ₹1,000 trade than of a ₹1 lakh one."))
    return blocks


# ---------------------------------------------------------------- detail


class Holding(BaseModel):
    ticker: str
    quantity: float
    avg_cost: float | None = None


class DetailReq(BaseModel):
    ticker: str
    target: float = Field(0.05, ge=0.005, le=1)
    hz: int = Field(21, ge=2, le=250)
    stop: str = "auto"
    capital: float = Field(1_000_000, ge=1000)
    rules: dict[str, Any] = {}
    holdings: list[Holding] = []
    open_trades: int = 0
    row: dict[str, Any] | None = None


@router.post("/finder/detail")
def finder_detail(req: DetailReq):
    t = normalize_symbol(req.ticker)
    c = stock_ctx(t)
    mode, fixed = _stop_arg(req.stop)
    stop = fc.auto_stop_pct(c.df, req.hz, req.target) if mode == "auto" else fixed
    f = fc.forecast_stock(t, c.df, c.bench, req.hz, req.target, stop, None, n_paths=4000)
    row = req.row or {}
    p_t = row.get("p_target", f.p_target)
    p_s = row.get("P(stop first)", f.p_stop)
    grade = row.get("Grade", f.grade)
    verdict = row.get("Verdict", f.verdict)
    ev = row.get("Exp. return", f.ev)
    engines = dict(row.get("engines") or f.engines)
    blocks = source_notice(c.source)
    blocks.append({"type": "stockhead", "name": display_name(t), "ticker": t, "sector": c.sector, "asof": f.as_of,
                   "rating": f"GRADE {grade}", "cls": {"A": "b-sbuy", "B": "b-buy", "C": "b-hold"}.get(grade, "b-ssell"),
                   "sub": verdict})
    items = [("Target first", pct(p_t, 1), f"₹{f.target_price:,.2f} (+{f.target_pct:.1%})", "up"),
             ("Stop first", pct(p_s, 1), f"₹{f.stop_price:,.2f} (−{f.stop_pct:.1%})" if f.stop_price else "no stop", "down"),
             ("Touches target", pct(f.p_touch, 1), "at any point"),
             ("Expected return", pct(ev, 2, True), "per trade, after costs", tone(ev))]
    if f.stop_pct:
        be = f.breakeven
        win = p_t / (p_t + p_s) if (p_t + p_s) > 0 else float("nan")
        items.append(("Win share needed", pct(be, 0), f"got {pct(win, 0)} of decided cases", "up" if win > be else "down"))
    items.append(("Typical time", f"{f.days_to_target:.0f} days" if np.isfinite(f.days_to_target) else "—",
                  "median when target hit"))
    blocks.append(kpis(items))
    for n in (row.get("notes") or f.notes):
        blocks.append(caption("• " + n))
    for w in jevents.event_warnings(t, req.hz):
        blocks.append(callout("warn", w))

    # --- price cone
    fan = f.sim.fan
    fut = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(c.df.index[-1], periods=len(fan) + 1)[1:]]
    hist = c.df["Close"].iloc[-max(30, req.hz * 2):]
    data = [{"type": "scatter", "mode": "lines", "x": [d.strftime("%Y-%m-%d") for d in hist.index],
             "y": figs._v(hist, 2), "name": "Price", "line": {"color": figs.MUTED, "width": 1.5}},
            {"type": "scatter", "x": fut, "y": figs._v(fan["p95"], 2), "line": {"width": 0}, "showlegend": False,
             "hoverinfo": "skip"},
            {"type": "scatter", "x": fut, "y": figs._v(fan["p5"], 2), "fill": "tonexty", "line": {"width": 0},
             "fillcolor": "rgba(76,141,255,0.12)", "name": "90% range"},
            {"type": "scatter", "x": fut, "y": figs._v(fan["p75"], 2), "line": {"width": 0}, "showlegend": False,
             "hoverinfo": "skip"},
            {"type": "scatter", "x": fut, "y": figs._v(fan["p25"], 2), "fill": "tonexty", "line": {"width": 0},
             "fillcolor": "rgba(76,141,255,0.25)", "name": "50% range"},
            {"type": "scatter", "x": fut, "y": figs._v(fan["p50"], 2), "name": "Median path",
             "line": {"color": figs.BLUE, "width": 2}}]
    cone = figs.fig(data, {"title": f"Where the price could be over the next {req.hz} trading days",
                           "hovermode": "x unified"})
    marks = [figs.hline(f.target_price, figs.UP, "dash", f"Target ₹{f.target_price:,.0f}")]
    if f.stop_price:
        marks.append(figs.hline(f.stop_price, figs.DOWN, "dash", f"Stop ₹{f.stop_price:,.0f}"))
    figs.with_marks(cone, marks)
    tp = f.sim.terminal_pct
    t_cone = [plot(cone, 380),
              caption(f"{f.sim.n_paths:,} simulated paths (seed {f.sim.seed}, repeatable). Return at day {req.hz}: median "
                      f"{tp[50]:+.1%}, 90% of paths between {tp[5]:+.1%} and {tp[95]:+.1%}. A simulation from the "
                      "stock's own past, not a promise.")]
    # --- engines
    eng = pd.Series(engines)
    eb = figs.bar_h(list(eng.index), list(eng.values), "P(target first) by engine", ".0%",
                    [figs.GOLD, figs.BLUE, "#E86AF0"][:len(eng)], xrange=[0, 1])
    figs.with_marks(eb, [figs.vline(p_t, figs.UP, "dot", f"Blend {p_t:.0%}")])
    t_eng = [plot(eb, 240), caption("Time Machine and History replay always count equally. The ML vote counts only as much "
                                   "as its measured out-of-sample skill (zero at AUC 0.52, full at 0.60).")]
    # --- history
    h = f.hist
    cmp_ = pd.DataFrame({"All past days": h["all"], "Days like today": h["similar"], "Used (shrunk)": h["shrunk"]}).T
    cmp_ = cmp_.reset_index().rename(columns={"index": "Sample"})
    sh = fc.setup_history(c.df, setups.setup_frame(c.df), req.hz, req.target, f.stop_pct).reset_index()
    active = setups.active_setups(c.df)
    t_hist = [table(cmp_, [col("Sample", "Sample"), col("n", "Days", "int"), col("p_target", "Target first", "pct0"),
                           col("p_stop", "Stop first", "pct0"), col("p_neither", "Neither", "pct0"),
                           col("avg_ret", "Avg return", "spct2")]),
              caption("“Days like today” = same side of the 200-DMA and a JINNI score within ±10. Small samples are shrunk "
                      "toward all days. Windows overlap, so the independent sample is smaller than the day count."),
              text("**Every past time each setup fired**" + (f" (active now: {', '.join(active)})" if active
                                                            else " (no setup active today)")),
              table(sh, [col("Setup", "Setup"), col("Times fired", "Times", "int"), col("Target first", "Target", "pct0"),
                         col("Stop first", "Stop", "pct0"), col("Neither", "Neither", "pct0"),
                         col("Avg return", "Avg return", "spct2")])]
    # --- odds grid
    mat = memo(("odds", t, str(c.df.index[-1])), 600, lambda: fc.odds_matrix(c.df))
    t_odds = [plot(figs.heatmap(mat.values, mat.columns, mat.index, f"{display_name(t)}: chance of touching each gain (no stop)",
                                0, 1, texts=[[f"{v:.0%}" for v in r] for r in mat.values]), 280),
              caption("Touch probability: the high reaches the level at least once in the window. Average of the Time "
                      "Machine and history.")]
    # --- trade plan
    t_plan, track = trade_plan_blocks(req, c, f, stop, p_t, grade, verdict, t)
    # --- news
    n, s = news_of(c)
    t_news = []
    if s["n"]:
        t_news.append(kpis([("News tone", s["label"], f"score {s['score']:+.2f}", tone(s["score"]))]))
        t_news.append(news_list([{"ticker": display_name(t), "time": r["Time"], "headline": r["Headline"], "tags": r["Tags"],
                                  "link": r["Link"], "score": float(s["scores"].get(k, 0))} for k, r in n.iterrows()]))
    else:
        t_news.append(caption("No headlines available right now."))
    blocks.append(tabs([("Price cone", t_cone), ("Engines", t_eng), ("History", t_hist), ("Odds table", t_odds),
                        ("Trade plan", t_plan), ("News", t_news)]))
    return reply(doc(blocks, c.source, track=track))


def trade_plan_blocks(req: DetailReq, c, f, stop, p_t, grade, verdict, t):
    rules = {**rc.DEFAULT_RULES, **{k: v for k, v in req.rules.items() if k in rc.DEFAULT_RULES}}
    if not f.stop_price:
        return [callout("info", "Pick a stop-loss (Auto or Fixed) to size a trade.")], None
    capital = req.capital
    sz = rc.size_position(f.price, f.stop_price, capital, capital, rules)
    lim = pd.DataFrame({"Max shares": sz.limits}).reset_index().rename(columns={"index": "Limit"})
    lim["Decides"] = lim["Limit"] == sz.deciding_limit
    out = [table(lim, [col("Limit", "Limit"), col("Max shares", "Max shares", "int"), col("Decides", "Decides", "bool")])]
    if sz.refused:
        out.append(callout("error", sz.refused))
        return out, None
    out.append(text(f"**{sz.qty:,} shares** (₹{sz.qty * f.price:,.0f}), decided by **{sz.deciding_limit.lower()}**. Loss at "
                    f"stop ₹{sz.qty * (f.price - f.stop_price):,.0f}; gain at target "
                    f"₹{sz.qty * (f.target_price - f.price):,.0f}."))
    hold = [h for h in req.holdings if h.quantity > 0]
    sec_vals: dict = {}
    hv = pd.Series(dtype=float)
    rets = None
    if hold:
        hp = prices([h.ticker for h in hold] + [t], "2y", ttl=900)
        last_px = hp.close.ffill().iloc[-1]
        hv = pd.Series({h.ticker: h.quantity * float(last_px.get(h.ticker, np.nan)) for h in hold}).dropna()
        for k, v in hv.items():
            sec_vals[sector_of(k)] = sec_vals.get(sector_of(k), 0) + v
        rets = risk.returns(hp.close).iloc[-504:]
    g = rc.gate(f.price, f.stop_price, f.target_price, sz.qty, sector_of(t), p_t, capital, capital, sec_vals,
                req.open_trades, rules, jevents.event_warnings(t, req.hz))
    out.append({"type": "gate", "status": g.status, "checks": [list(x) for x in g.checks]})
    if sz.qty > 0 and rets is not None and not hv.empty:
        imp = pd.DataFrame(rc.portfolio_impact(rets, hv, t, sz.qty * f.price)).T.reset_index().rename(columns={"index": "Book"})
        out.append(text("**Portfolio impact** (your holdings + this trade)"))
        out.append(table(imp, [col("Book", "Book"), col("Value", "Value", "inr0"), col("VaR %", "1-day VaR 95%", "pct2"),
                               col("VaR ₹", "VaR ₹", "inr0"), col("Volatility", "Volatility", "pct1")]))
    out.append(caption("Paper tracking only: JINNI never places orders. Edit the rules in Settings."))
    track = {"ticker": t, "entry": f.price, "target_pct": f.target_pct, "stop_pct": f.stop_pct,
             "target_price": f.target_price, "stop_price": f.stop_price, "horizon": req.hz, "p_target": p_t,
             "p_stop": f.p_stop, "grade": grade, "verdict": verdict, "qty": sz.qty, "entry_date": f.as_of,
             "source": "plan", "allowed": g.status != "BLOCK"}
    return out, track


def base_bench(pdx):
    return pdx.close[BENCHMARK].dropna() if BENCHMARK in pdx.close else None
