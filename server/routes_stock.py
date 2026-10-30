"""Single-stock endpoints (one per Analyzer tab so each stays fast and cacheable)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from jinni import data as jdata
from jinni import events as jevents
from jinni import forecast as fc
from jinni import indicators as ind
from jinni import patterns as jpat
from jinni import risk
from jinni import seasonality as jseason
from jinni import sentiment as sm
from jinni import setups as jsetups
from jinni import signals, ta
from jinni.universe import BENCHMARK, NIFTY50, display_name, normalize_symbol

from . import figs
from .core import (ApiError, span_bars, span_label, caption, callout, col, cols, doc, inr, kpis, memo, need_series, news_list, num, pct, plot,
                   prices, rating_class, reply, section, source_notice, table, tabs, text, tone)

router = APIRouter()
CTX_TTL = 600


class Ctx:
    def __init__(self, t: str):
        self.t = normalize_symbol(t)
        self.pdx = prices([self.t, BENCHMARK], "10y", ttl=CTX_TTL)
        self.df = need_series(self.pdx, self.t)
        self.bench = self.pdx.close[BENCHMARK] if BENCHMARK in self.pdx.close else None
        self.info = memo(("info", self.t), 3600, lambda: jdata.fetch_info(self.t))
        self.name = self.info.get("longName") or self.info.get("shortName") or display_name(self.t)
        self.sector = self.info.get("sector") or NIFTY50.get(self.t, "")

    @property
    def source(self):
        return self.pdx.source


def ctx(t: str) -> Ctx:
    return memo(("ctx", normalize_symbol(t)), 120, lambda: Ctx(t))


def window_scan(c: Ctx, bars: int):
    """Patterns re-detected inside the visible window, so 3 months and 5 years show different patterns."""
    full, _ = scan_of(c)
    return memo(("wscan", c.t, str(c.df.index[-1]), bars), CTX_TTL,
                lambda: jpat.scan_window(c.df, bars, full["candle_frame"]))


def scan_of(c: Ctx):
    return memo(("scan", c.t, str(c.df.index[-1])), CTX_TTL, lambda: (jpat.scan(c.df), ta.summary(c.df)))


def news_of(c: Ctx):
    n = memo(("news", c.t), 600, lambda: jevents.headlines(c.t))
    return n, sm.score_news(n)


# --------------------------------------------------------------- overview


@router.get("/stock/{t}/overview")
def overview(t: str):
    c = ctx(t)
    df = c.df
    scores = memo(("scores", c.t, str(df.index[-1])), CTX_TTL, lambda: signals.factor_scores(df, c.bench))
    last = scores.iloc[-1]
    fund_score, fund_parts = signals.fundamental_score(c.info)
    composite = float(last["Composite"])
    if fund_score is not None:
        composite = 0.8 * composite + 0.2 * fund_score
    rating = signals.rating_from_score(composite)
    price = float(df["Close"].iloc[-1])
    chg = price / float(df["Close"].iloc[-2]) - 1
    hi52, lo52 = float(df["Close"].iloc[-252:].max()), float(df["Close"].iloc[-252:].min())
    atr = float(ind.atr(df).iloc[-1])
    vol20 = float(ind.realized_vol(df["Close"]).iloc[-1])
    rsi = float(ind.rsi(df["Close"]).iloc[-1])

    blocks = source_notice(c.source)
    blocks.append({"type": "stockhead", "name": c.name, "ticker": c.t, "sector": c.sector,
                   "asof": df.index[-1].strftime("%d %b %Y"), "rating": rating, "cls": rating_class(rating)})
    blocks.append(kpis([
        ("Last price", f"₹{price:,.2f}", pct(chg, 2, True), tone(chg)),
        ("JINNI score", f"{composite:.0f}/100", rating),
        ("52-week range", f"₹{lo52:,.0f}–{hi52:,.0f}", f"{pct(price / hi52 - 1, 1)} from high"),
        ("RSI (14)", f"{rsi:.1f}", "Wilder"),
        ("Volatility (20d)", pct(vol20, 1), "annualised"),
        ("ATR (14)", f"₹{atr:,.2f}", pct(atr / price, 2) + " of price")]))

    scan, (ta_tbl, ta_v) = scan_of(c)
    news, news_s = news_of(c)
    active = scan["chart"] + scan["harmonic"] + scan["wave"] + scan["divergence"] + scan["trend"]
    recent_c = [p for p in scan["candles"] if p.end >= df.index[-3]]
    bull_n = sum(p.bias > 0 for p in active + recent_c)
    bear_n = sum(p.bias < 0 for p in active + recent_c)
    hz = 21
    f1m = fc.forecast_stock(c.t, df, c.bench, hz, 0.05, fc.auto_stop_pct(df, hz, 0.05), None, n_paths=3000)
    cal = jseason.calendar_effects(df["Close"])
    this_month = pd.Timestamp.today().month_name()
    m_row = cal["Month"].loc[this_month] if this_month in cal["Month"].index else None
    parts = {"Technicals (87 indicators)": (ta_v["Overall"]["score"] + 1) * 50, "Factor score": float(last["Composite"]),
             "Patterns": 50 + 50 * (bull_n - bear_n) / max(bull_n + bear_n, 1),
             "Forecast (+5% in 1 month)": float(np.clip(50 + f1m.ev * 1000, 0, 100))}
    if fund_score is not None:
        parts["Fundamentals"] = fund_score
    if np.isfinite(news_s["score"]):
        parts["News sentiment"] = 50 + 50 * news_s["score"]
    overall = float(np.mean(list(parts.values())))
    lab = signals.rating_from_score(overall)
    blocks.append({"type": "score", "label": "JINNI 360° score", "value": overall, "rating": lab, "cls": rating_class(lab)})
    f = figs.bar_h(list(parts), list(parts.values()), "What each lens says (50 = neutral)", ".0f",
                   [figs.UP if v >= 60 else figs.DOWN if v < 42 else figs.GOLD for v in parts.values()],
                   xrange=[0, 105])
    f["layout"]["shapes"] = [{"type": "line", "xref": "x", "yref": "paper", "x0": 50, "x1": 50, "y0": 0, "y1": 1,
                              "line": {"color": figs.MUTED, "dash": "dot"}}]
    blocks.append(plot(f, 280))

    r1y = df["Close"].iloc[-1] / df["Close"].iloc[-min(253, len(df))] - 1
    ev_w = jevents.event_warnings(c.t, 21)
    past = [kpis([("1-year return", pct(r1y, 1, True), "", tone(r1y)),
                  (f"{this_month} history", pct(m_row["Avg %"] / 100 * 21, 1, True) if m_row is not None else "—",
                   f"up {m_row['Up %']:.0%} of days" if m_row is not None else ""),
                  ("Max drawdown", pct(risk.max_drawdown(df["Close"].pct_change())["max_drawdown"], 0), "", "down")])]
    now = [kpis([("Technical summary", ta_v["Overall"]["label"],
                  f"{ta_v['Overall']['buy']} buy · {ta_v['Overall']['sell']} sell"),
                 ("Patterns", f"{bull_n} bullish · {bear_n} bearish", "active chart + recent candles"),
                 ("News tone", news_s["label"], f"{news_s['n']} headlines")])]
    fut = [kpis([("+5% within 1 month", pct(f1m.p_target, 0), f"before a −{f1m.stop_pct:.1%} stop"),
                 ("Median in 1 month", pct(f1m.sim.terminal_pct[50], 1, True),
                  f"90%: {f1m.sim.terminal_pct[5]:+.0%} to {f1m.sim.terminal_pct[95]:+.0%}"),
                 ("Events ahead", f"{len(ev_w)}" if ev_w else "None known", ev_w[0][:40] if ev_w else "")])]
    blocks.append(cols([text("**Past**")] + past, [text("**Present**")] + now, [text("**Future**")] + fut))
    blocks.append(caption("The 360° score averages independent lenses so no single signal dominates. Patterns and "
                          "indicators describe the present; only the Forecast lens is a tested probability."))

    cats = list(signals.WEIGHTS) + (["Fundamentals"] if fund_score is not None else [])
    vals = [float(last[k]) for k in signals.WEIGHTS] + ([fund_score] if fund_score is not None else [])
    radar = figs.fig([{"type": "scatterpolar", "r": vals + vals[:1], "theta": cats + cats[:1], "fill": "toself",
                       "line": {"color": figs.GOLD}, "fillcolor": "rgba(245,184,65,0.18)"}],
                     {"polar": {"bgcolor": "rgba(0,0,0,0)", "radialaxis": {"range": [0, 100], "gridcolor": "var(--grid)"},
                                "angularaxis": {"gridcolor": "var(--grid)"}}, "showlegend": False,
                      "title": "Factor profile (0–100)"})
    sc = scores.iloc[-504:]
    hist = figs.lines(sc.index, {"Composite": dict(y=sc["Composite"], color=figs.GOLD, width=2.2)}, "Score over time",
                      yrange=(0, 100))
    hist["layout"]["shapes"] = [figs.hline(60, figs.UP)[0], figs.hline(42, figs.DOWN)[0]]
    blocks.append(cols([plot(radar, 340)], [plot(hist, 340)]))
    blocks.append(caption("Technical weights: " + ", ".join(f"{k} {int(w * 100)}%" for k, w in signals.WEIGHTS.items())
                          + (". Final score = 80% technical + 20% fundamentals." if fund_score is not None
                             else ". No fundamentals available, so the score is technical only.")))
    ev = signals.score_history_stats(scores["Composite"], df["Close"], 20)
    if not ev.empty:
        blocks.append(section("Did this rating work before on this stock?"))
        e = ev.reset_index()
        blocks.append(table(e, [col("Rating", "Rating", "badge"), col("Days", "Days", "int"),
                                col("Avg 20d return %", "Avg 20d %", "n2"), col("Hit rate % (ret>0)", "Up %", "n0"),
                                col("Worst 20d %", "Worst 20d %", "n1")]))
        blocks.append(caption("In-sample check of a rules-based score (no fitting). Windows overlap, so day counts "
                              "overstate independent observations."))
    return reply(doc(blocks, c.source, price=price, rating=rating, name=c.name, overall=overall), ttl=120)


# ------------------------------------------------------------------ chart


@router.get("/stock/{t}/chart")
def chart(t: str, overlays: str = "EMA 20,SMA 50,SMA 200", panels: str = "RSI 14,MACD",
          layers: str = "Candlestick patterns,Chart patterns,Trendlines,Support/resistance zones",
          n: int = Query(6, ge=1, le=600), unit: str = "month", heikin: int = 0):
    c = ctx(t)
    ov = [x for x in (s.strip() for s in overlays.split(",")) if x in ta.CATALOG and ta.CATALOG[x].kind == "overlay"][:14]
    pn = [x for x in (s.strip() for s in panels.split(",")) if x in ta.CATALOG and ta.CATALOG[x].kind == "panel"][:4]
    ly = [x for x in (s.strip() for s in layers.split(",")) if x in figs.LAYERS]
    bars = span_bars(c.df, n, unit, minimum=5)
    scan = window_scan(c, bars)
    f = figs.price_chart(c.df, ov, pn, ly, scan, bool(heikin), bars)
    blocks = source_notice(c.source)
    blocks.append(text(f"**{span_label(n, unit).capitalize()}** · {bars} trading days · "
                       f"{c.df.index[-bars].strftime('%d %b %Y')} to {c.df.index[-1].strftime('%d %b %Y')}. "
                       "Patterns are detected inside this window, so changing the range changes what is found."))
    blocks.append(plot(f, 520 + 150 * len(pn)))
    blocks.append(caption("▲/▼ markers = bullish/bearish candlestick patterns; shaded bands = support (green) / "
                          "resistance (red) zones; ✓ = breakout confirmed by a close. Tap a marker for its name."))
    blocks.append(section("Technical events"))
    blocks.append(text("\n".join(f"- {e}" for e in signals.detect_events(c.df))))
    blocks.append(section("Pivot levels (all methods)"))
    pv = ta.pivots_all(c.df).reset_index().rename(columns={"index": "Level"})
    blocks.append(table(pv, [col("Level", "Level")] + [col(k, k, "inr2") for k in
                                                       ["Classic", "Fibonacci", "Camarilla", "Woodie", "DeMark"]]))
    vp = ta.volume_profile(c.df)
    blocks.append(kpis([("Volume POC (1y)", f"₹{vp['POC']:,.2f}", "most-traded price"),
                        ("Value area high", f"₹{vp['VAH']:,.2f}", "70% of volume"),
                        ("Value area low", f"₹{vp['VAL']:,.2f}")]))
    return reply(doc(blocks, c.source), ttl=300)


# --------------------------------------------------------------- patterns


@router.get("/stock/{t}/patterns")
def patterns(t: str, h: int = Query(10, ge=5, le=20), n: int = Query(6, ge=1, le=600), unit: str = "month"):
    c = ctx(t)
    bars = span_bars(c.df, n, unit, minimum=5)
    scan = window_scan(c, bars)
    blocks = source_notice(c.source)
    all_p = scan["chart"] + scan["harmonic"] + scan["wave"] + scan["divergence"] + scan["trend"] + scan["gap"]
    cndl = [p for p in scan["candles"]]
    blocks.append(text(f"**{len(all_p)} chart patterns and {len(cndl)} recent candle patterns** in the last "
                       f"{span_label(n, unit)} ({bars} trading days). Everything listed below is drawn on this chart."))
    f = figs.price_chart(c.df, ["EMA 20", "SMA 50"], [], [x for x in figs.LAYERS if x != "Fibonacci"], scan, False, bars)
    blocks.append(plot(f, 560))
    blocks.append(caption("Green = bullish, red = bearish, blue = trendlines, dotted lines = pattern targets. "
                          "Change the range above to look for shorter or longer patterns."))
    rows = [{"Pattern": p.name, "Type": p.kind.title(), "Bias": p.bias_text, "Status": p.status,
             "Since": p.start, "Target": p.target, "Invalidation": p.invalidation, "Note": p.note} for p in all_p]
    rows += [{"Pattern": p.name, "Type": "Candle", "Bias": p.bias_text, "Status": "printed", "Since": p.end,
              "Target": None, "Invalidation": None, "Note": ""} for p in cndl]
    if rows:
        blocks.append(table(pd.DataFrame(rows), [col("Pattern", "Pattern"), col("Type", "Type"), col("Bias", "Bias", "badge"),
                                                 col("Status", "Status"), col("Since", "Since", "date"),
                                                 col("Target", "Target ₹", "inr2"), col("Invalidation", "Invalid. ₹", "inr2"),
                                                 col("Note", "What it means")]))
    else:
        blocks.append(callout("info", "No chart patterns in this window. Try a longer range."))
    blocks.append(section("Did these patterns work on this stock before?"))
    cs = jpat.candle_stats(c.df, scan_of(c)[0]["candle_frame"], h)
    cs = cs[cs["Times"] > 0].sort_values("Times", ascending=False).reset_index()
    blocks.append(text(f"**Candlestick patterns**: average {h}-day move after each, on {display_name(c.t)}'s own history "
                       f"(normal {h}-day average: {cs.attrs.get('base_mean', 0):+.2%})."))
    blocks.append(table(cs, [col("Pattern", "Pattern"), col("Bias", "Bias", "badge"), col("Times", "Times", "int"),
                             col(f"Avg {h}d", f"Avg {h}d", "spct2"), col("Up %", "Up %", "pct0"),
                             col("Worked %", "Worked %", "pct0", help="Moved in the pattern's direction"),
                             col("Edge vs avg", "Edge", "spct2"), col("Last seen", "Last seen", "date")]))
    ph = memo(("phist", c.t, max(h, 10), str(c.df.index[-1])), 3600,
              lambda: jpat.pattern_history(c.df.iloc[-1260:], max(h, 10)))
    blocks.append(text("**Chart & harmonic patterns**, replayed through history using only information available at the time:"))
    ph = ph.reset_index()
    cols_ = [col("Pattern", "Pattern"), col("Times", "Times", "int")] + [
        col(k, k, "spct2" if k.startswith("Avg") else "pct0") for k in ph.columns if k not in ("Pattern", "Times")]
    blocks.append(table(ph, cols_))
    blocks.append(caption("Small counts are noise. A pattern that 'worked' 50% of the time on one stock has no edge there."))
    return reply(doc(blocks, c.source), ttl=600)


# ------------------------------------------------------------- indicators


@router.get("/stock/{t}/indicators")
def indicators(t: str):
    c = ctx(t)
    _, (tbl, v) = scan_of(c)
    blocks = source_notice(c.source)
    gs = []
    for cat in ["Overall", "Moving averages", "Momentum"]:
        sc = v[cat]["score"]
        g = figs.gauge(sc * 100, (-100, 100), cat, v[cat]["label"],
                       figs.UP if sc > .1 else figs.DOWN if sc < -.1 else figs.GOLD,
                       [dict(range=[-100, -50], color="rgba(239,68,68,.25)"), dict(range=[50, 100], color="rgba(34,197,94,.25)")],
                       [-100, -50, 0, 50, 100])
        gs.append([plot(g, 270)])
    blocks.append(cols(*gs))
    blocks.append(kpis([(cat, v[cat]["label"], f"{v[cat]['buy']} buy · {v[cat]['neutral']} neutral · {v[cat]['sell']} sell")
                        for cat in ta.CATEGORIES]))
    t_ = tbl.drop(columns=["Votes"])
    blocks.append(table(t_, [col("Indicator", "Indicator"), col("Category", "Category"), col("Value", "Value", "n2"),
                             col("Signal", "Signal", "badge"), col("About", "About")], height=560))
    blocks.append(caption(f"{len(ta.CATALOG)} indicators. Measurement-only indicators (ATR, volatility, width…) show "
                          "values but don't vote. Indicator signals are descriptive, and many are correlated: 20 moving "
                          "averages agreeing is one opinion, not twenty."))
    return reply(doc(blocks, c.source), ttl=600)


# ------------------------------------------------------------------- odds


@router.get("/stock/{t}/odds")
def odds(t: str, target: float = Query(0.05, ge=0.005, le=1), hz: int = Query(21, ge=2, le=250)):
    c = ctx(t)
    mat = memo(("odds", c.t, str(c.df.index[-1])), CTX_TTL, lambda: fc.odds_matrix(c.df))
    blocks = source_notice(c.source)
    blocks.append(text("Chance of the price **touching** each gain within each window (no stop), averaged from 3,000 "
                       "simulated paths and the stock's own history."))
    blocks.append(plot(figs.heatmap(mat.values, mat.columns, mat.index, "", 0, 1,
                                    texts=[[f"{v:.0%}" for v in r] for r in mat.values]), 280))
    stop = fc.auto_stop_pct(c.df, hz, target)
    fq = fc.forecast_stock(c.t, c.df, c.bench, hz, target, stop, None, n_paths=4000)
    blocks.append(kpis([("Target first", pct(fq.p_target, 0), f"₹{fq.target_price:,.2f}", "up"),
                        ("Stop first", pct(fq.p_stop, 0), f"₹{fq.stop_price:,.2f} (−{stop:.1%} auto)", "down"),
                        ("Expected return", pct(fq.ev, 2, True), "after costs", tone(fq.ev)),
                        ("Grade", fq.grade, fq.verdict)]))
    act = jsetups.active_setups(c.df)
    blocks.append(caption(("Active setups: " + ", ".join(act) + ". ") if act else "No chart setup active today. "))
    return reply(doc(blocks, c.source), ttl=600)


# ------------------------------------------------------------ seasonality


@router.get("/stock/{t}/seasonality")
def seasonality(t: str):
    c = ctx(t)
    cal = jseason.calendar_effects(c.df["Close"])
    blocks = source_notice(c.source)
    blocks.append(text(f"How {display_name(c.t)} has behaved by calendar (last {len(c.df) // 252} years). Rows marked "
                       "significant survive a correction for testing many rows at once."))
    left, right = [], []
    for side, key in ((left, "Month"), (right, "Weekday")):
        tt = cal[key].dropna(subset=["Avg %"])
        side.append(plot(figs.bar_v(tt.index, tt["Avg %"].values, f"Average daily return by {key.lower()} (%)", ".2f",
                                    texts=[f"{u:.0%} up" for u in tt["Up %"]]), 300))
    blocks.append(cols(left, right))
    sp = cal["Special days"].reset_index().rename(columns={"index": "Day type"})
    blocks.append(table(sp, [col("Day type", "Day type"), col("Days", "Days", "int"), col("Avg %", "Avg %", "n3"),
                             col("Up %", "Up %", "pct0"), col("p-value", "p-value", "n3"),
                             col("Signif. (after testing all rows)", "Significant?", "bool")]))
    up = jseason.upcoming(c.df["Close"])
    if not up.empty:
        blocks.append(section("Coming up"))
        blocks.append(table(up.reset_index(), [col("Date", "Date", "date"), col("Event", "Event"),
                                               col("Past avg %", "Past avg %", "n2"), col("Past up %", "Past up %", "pct0"),
                                               col("Past events", "Events", "int")]))
    return reply(doc(blocks, c.source), ttl=900)


# -------------------------------------------------------------- trade plan


@router.get("/stock/{t}/plan")
def plan(t: str, side: str = "LONG", capital: float = Query(1_000_000, ge=1000), risk_pct: float = Query(0.01, gt=0, le=0.1)):
    c = ctx(t)
    side = "SHORT" if side.upper() == "SHORT" else "LONG"
    p = signals.trade_plan(c.df, side, capital, risk_pct)
    rows = pd.DataFrame(p.as_rows())
    blocks = source_notice(c.source)
    blocks.append(table(rows, [col("Level", "Level"), col("Price", "Price", "inr2"), col("Move %", "Move %", "n2"),
                               col("R multiple", "R", "n1")]))
    blocks.append(kpis([("Position size", f"{p.qty:,} shares", inr(p.qty * p.entry)),
                        ("Capital at risk", inr(p.capital_at_risk), pct(p.capital_at_risk / capital, 2) + " of capital"),
                        ("Risk / share", f"₹{p.risk_per_share:,.2f}", pct(p.risk_per_share / p.entry) + " stop distance")]))
    if side == "SHORT":
        blocks.append(caption("Shorting delivery equity is not allowed in India; use stock futures/options (F&O names "
                              "only) or treat this as an exit plan."))
    blocks.append(caption("Stop = tighter of 2×ATR or just beyond the nearest swing level. Targets at 1.5R / 2.5R / 4R, "
                          "with T1 pulled to the nearest resistance (support for shorts) when that comes first."))
    return reply(doc(blocks, c.source, plan={"entry": p.entry, "stop": p.stop, "targets": p.targets, "qty": p.qty}), ttl=300)


# ------------------------------------------------------------ fundamentals


@router.get("/stock/{t}/fundamentals")
def fundamentals(t: str):
    c = ctx(t)
    m = c.info
    if not m:
        return reply(doc([callout("info", "Fundamentals are unavailable for this symbol right now.")], c.source), ttl=300)

    def g(k, f=lambda v: v):
        v = m.get(k)
        try:
            return f(v) if v is not None else "—"
        except Exception:  # noqa: BLE001
            return "—"
    dy = m.get("dividendYield")
    items = [("Market cap", g("marketCap", inr)), ("P/E (TTM)", g("trailingPE", lambda v: f"{v:.1f}")),
             ("P/B", g("priceToBook", lambda v: f"{v:.2f}")), ("ROE", g("returnOnEquity", lambda v: pct(v, 1))),
             ("Debt / Equity", g("debtToEquity", lambda v: f"{v / 100:.2f}")),
             ("Dividend yield", f"{dy:.2f}%" if isinstance(dy, (int, float)) else "—"),
             ("Profit margin", g("profitMargins", lambda v: pct(v, 1))),
             ("Earnings growth", g("earningsGrowth", lambda v: pct(v, 1, True)))]
    blocks = source_notice(c.source) + [kpis(items)]
    _, parts = signals.fundamental_score(m)
    if parts:
        s = pd.Series(parts)
        blocks.append(plot(figs.bar_h(s.index, s.values / 100 - 0.5, "Fundamental sub-scores (centred at 50)", ".0%"), 260))
        blocks.append(caption("Bars show each sub-score relative to a neutral 50."))
    return reply(doc(blocks, c.source), ttl=1800)


# -------------------------------------------------------------------- news


@router.get("/stock/{t}/news")
def stock_news(t: str):
    c = ctx(t)
    n, s = news_of(c)
    blocks = [callout("warn", w) for w in jevents.event_warnings(c.t, 21)]
    if s["n"]:
        blocks.append(kpis([("News tone", s["label"], f"score {s['score']:+.2f} (−1 to +1)", tone(s["score"])),
                            ("Positive headlines", f"{s['pos']}", "", "up"),
                            ("Negative headlines", f"{s['neg']}", "", "down")]))
        items = [{"ticker": display_name(c.t), "time": r["Time"], "headline": r["Headline"], "tags": r["Tags"],
                  "link": r["Link"], "score": float(s["scores"].get(k, 0))} for k, r in n.iterrows()]
        blocks.append(news_list(items))
    else:
        blocks.append(caption("No headlines available right now."))
    blocks.append(caption("Tone is scored with a finance word list (negation-aware). It measures how headlines read, "
                          "not whether they are right."))
    return reply(doc(blocks, c.source), ttl=300)
