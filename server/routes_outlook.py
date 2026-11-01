"""Outlook (ML direction estimate for the market or a stock) and live multi-source sentiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from jinni import data as jdata
from jinni import newsfeeds as nf
from jinni import outlook as ol
from jinni.universe import BENCHMARK, display_name

from . import figs
from .core import (ApiError, caption, callout, col, cols, doc, kpis, memo, news_list, pct, plot, prices, reply, section,
                   source_notice, table, text, tone)
from .routes_stock import ctx

router = APIRouter()
THEMES_MARKET = None  # all themes feed the market tone


def _news():
    return memo(("news-all",), 600, lambda: nf.fetch_all())


def _company_news(c):
    base = c.name.replace(" Limited", "").replace(" Ltd.", "").replace(" Ltd", "").strip()
    q = f"\"{base}\" OR {c.t.split('.')[0]} share OR stock"
    return memo(("news-co", c.t), 600, lambda: nf.fetch_all({"Company": q}, [c.t], general=False))


def _cues():
    return memo(("cues",), 1800, lambda: prices(ol.TICKERS, "10y", ttl=1800))


def _market_fit(h: int):
    def build():
        pdx = _cues()
        tg = pdx.ohlcv(BENCHMARK)
        return ol.fit_outlook(tg, pdx.close.drop(columns=[BENCHMARK], errors="ignore"), h, True)
    return memo(("outlook-mkt", h), 1800, build)


def _lean(p: float, edge: str = "weak") -> tuple[str, str]:
    if edge == "none":
        return "No reliable lean: the model has no proven edge", "gold"
    strength = "Leans" if edge in ("modest", "useful") else "Slight lean"
    return ((f"{strength} bullish", "up") if p >= 0.55 else (f"{strength} bearish", "down") if p <= 0.45
            else ("No clear lean", "gold"))


def _tone_label(t: float) -> str:
    return ("Very positive" if t > 0.35 else "Positive" if t > 0.1 else "Very negative" if t < -0.35
            else "Negative" if t < -0.1 else "Neutral") if np.isfinite(t) else "No data"


def _disclaimer(o: ol.Outlook, what: str, hz: str) -> dict:
    edge = {"none": "no measurable edge over a coin flip", "weak": "a weak edge", "modest": "a modest edge",
            "useful": "a useful edge"}[o.edge]
    return callout("warn", f"**This is a machine-learning estimate, not a prediction you can rely on.** Over {o.n_oos:,} "
                           f"out-of-sample days the model for {what} had an AUC of {o.auc:.3f} (0.500 = coin flip), "
                           f"meaning {edge} on {hz}. Short-term moves are mostly noise; news and events can break any pattern. "
                           "Use it as one input, never as the decision.")


def _outlook_blocks(o: ol.Outlook, tl: dict, what: str, hz_label: str, tone_note: str) -> list[dict]:
    p_final = tl["p"]
    label, cls = _lean(p_final, o.edge)
    blocks = [_disclaimer(o, what, hz_label)]
    em = f"±{o.exp_move:.1%}" if np.isfinite(o.exp_move) else "—"
    blocks.append(kpis([
        (f"Chance {what} is higher {hz_label}", pct(p_final, 0), label, cls),
        ("Trained model only", pct(o.p_cal, 0), f"raw score {o.p_raw:.0%}"),
        ("News & event tilt", f"{tl['shift'] * 100:+.1f} pts", tone_note, tone(tl["shift"])),
        ("Typical move (1σ)", em, "from current volatility"),
        ("Validated AUC", f"{o.auc:.3f}", f"accuracy {o.accuracy:.1%} vs {o.base_rate:.1%} always-up"),
    ]))
    d = o.drivers
    f = figs.bar_h(d["Driver"], (d["Push"] * 100).tolist(), "What is pushing the call today (percentage points)", ".1f",
                   [figs.UP if v >= 0 else figs.DOWN for v in d["Push"]],
                   xrange=[float(d["Push"].min() * 100 * 1.5 - 1), float(d["Push"].max() * 100 * 1.5 + 1)])
    blocks.append(plot(f, 120 + 34 * len(d)))
    blocks.append(table(d.assign(Push=d["Push"] * 100), [
        col("Driver", "Input"), col("Today", "Today", "n3"), col("Typical", "Normally", "n3"),
        col("Push", "Push (pts)", "n1signed")]))
    blocks.append(section("Does a higher score really mean a better day?"))
    blocks.append(text("Out-of-sample calls split into five equal groups by the model's score. If it works, the share of up "
                       "days should rise from Q1 to Q5."))
    blocks.append(table(o.buckets, [col("Bucket", "Group"), col("Calls", "Calls", "int"), col("Avg_p", "Avg score", "pct0"),
                                    col("Up", "Actually up", "pct0"), col("Avg_move", "Avg move", "spct2")]))
    blocks.append(section("Walk-forward validation (train on the past, test on the next block)"))
    blocks.append(table(o.folds, [col("Test from", "From", "date"), col("Test to", "To", "date"), col("Days", "Days", "int"),
                                  col("Accuracy", "Accuracy", "pct1"), col("Base rate", "Always-up", "pct1"),
                                  col("AUC", "AUC", "n3")]))
    blocks.append(caption(f"{o.train_rows:,} training days up to {o.as_of:%d %b %Y}. Inputs are only things known at that day's "
                          "close (including the US session that ends after India shuts). Labels use a purge gap so "
                          "overlapping periods never leak. The news tilt is capped at ±3 points and is not part of the "
                          "validation."))
    return blocks


@router.get("/outlook/market")
def outlook_market(h: int = Query(1, ge=1, le=5)):
    o = _market_fit(h)
    news, status = _news()
    t = nf.overall_tone(news)
    tl = ol.tilt(o.p_cal, t["tone"], t["n"])
    note = (f"{_tone_label(t['tone']).lower()} tone across {t['n']} headlines" if t["n"] else "no headlines reachable")
    hz = "the next session" if h == 1 else f"within {h} sessions"
    blocks = source_notice("live") + _outlook_blocks(o, tl, "NIFTY", hz, note)
    return reply(doc(blocks, "live", p=tl["p"], p_model=o.p_cal, auc=o.auc, h=h), ttl=300)


@router.get("/stock/{t}/outlook")
def outlook_stock(t: str, h: int = Query(1, ge=1, le=10)):
    c = ctx(t)
    if len(c.df) < 900:
        raise ApiError(422, f"{c.t} has only {len(c.df)} trading days of history; the outlook needs about 4 years to train "
                            "and validate honestly.", code="short_history")

    def build():
        pdx = _cues()
        bench = pdx.close[BENCHMARK] if BENCHMARK in pdx.close else None
        return ol.fit_outlook(c.df, pdx.close.drop(columns=[BENCHMARK], errors="ignore"), h, False, bench)

    o = memo(("outlook-stk", c.t, h, str(c.df.index[-1])), 1800, build)
    co, _ = _company_news(c)
    mk, _ = _news()
    tc, tm = nf.overall_tone(co), nf.overall_tone(mk)
    parts = [(tc["tone"], 0.7, tc["n"]), (tm["tone"], 0.3, tm["n"])]
    ok = [(x, w, n) for x, w, n in parts if np.isfinite(x)]
    tone_v = sum(x * w for x, w, _ in ok) / sum(w for _, w, _ in ok) if ok else np.nan
    n_tot = tc["n"] + tm["n"]
    tl = ol.tilt(o.p_cal, tone_v, n_tot)
    note = (f"company {_tone_label(tc['tone']).lower()} ({tc['n']}), market {_tone_label(tm['tone']).lower()} ({tm['n']})")
    hz = "the next session" if h == 1 else f"within {h} sessions"
    blocks = source_notice(c.source) + _outlook_blocks(o, tl, display_name(c.t), hz, note)
    if not co.empty:
        blocks.append(section(f"Latest headlines about {display_name(c.t)}"))
        blocks.append(news_list(_items(co.head(25))))
    return reply(doc(blocks, c.source, p=tl["p"], p_model=o.p_cal, auc=o.auc, h=h), ttl=300)


def _items(df: pd.DataFrame) -> list[dict]:
    return [{"ticker": r["Source"], "time": r["Time"], "headline": r["Headline"], "tags": ", ".join(r["Topics"][:2]),
             "link": r["Link"], "score": float(r["Tone"])} for _, r in df.iterrows()]


@router.get("/sentiment/live")
def sentiment_live():
    """Every free source we can reach, scored by theme, with an event radar and a back-test of news tone."""
    news, status = _news()
    blocks = []
    ok = {k: v for k, v in status.items() if v != "unreachable"}
    if news.empty:
        blocks.append(callout("warn", "No news feed could be reached right now, so there is no sentiment to show. "
                                      "JINNI does not fill the gap with made-up numbers."))
        return reply(doc(blocks, "live"))
    t = nf.overall_tone(news)
    top = nf.topic_table(news)
    radar = nf.event_radar(news)
    blocks.append(kpis([
        ("Overall news tone", _tone_label(t["tone"]), f"{t['tone']:+.2f} · {t['n']} headlines", tone(t["tone"])),
        ("Positive / negative", f"{t['share_pos']:.0%} / {t['share_neg']:.0%}", "share of headlines"),
        ("Sources answering", f"{len(ok)} of {len(status)}", "Google News, publishers, Yahoo"),
        ("Events in the news", str(len(radar)), (radar.iloc[0]["Event"] if len(radar) else "none detected"))]))
    if len(top):
        f = figs.bar_h(top["Theme"], top["Tone"], "Tone by theme (recency-weighted, −1 to +1)", ".2f",
                       [figs.UP if v > 0.1 else figs.DOWN if v < -0.1 else figs.GOLD for v in top["Tone"]], xrange=[-1, 1], center=0)
        blocks.append(plot(f, 100 + 38 * len(top)))
        blocks.append(table(top, [col("Theme", "Theme"), col("Headlines", "Headlines", "int"), col("Tone", "Tone", "tone"),
                                  col("Positive", "Pos", "int"), col("Negative", "Neg", "int")], sort=("Headlines", True)))
    blocks.append(section("Event radar: what the news says is coming"))
    if len(radar):
        blocks.append(table(radar, [col("Event", "Event"), col("Imminent", "Sounds imminent", "int"), col("Mentions", "Mentions", "int"),
                                    col("Tone", "Tone", "tone"), col("Latest", "Latest headline")]))
        blocks.append(caption("Counts headlines that mention each event and flags ones that use words like 'tomorrow', 'next "
                              "week' or 'ahead of'. Tone shows whether coverage reads market-friendly or not; it does not "
                              "say who wins an election or what a decision will be."))
    else:
        blocks.append(callout("info", "No scheduled market-moving event stands out in today's headlines."))
    gd = memo(("gdelt",), 3600, lambda: nf.gdelt_tone())
    if gd is not None:
        pdx = prices([BENCHMARK], "2y", ttl=1800)
        bt = ol.tone_vs_next_day(gd, pdx.close[BENCHMARK].dropna())
        blocks.append(section("Has news tone predicted NIFTY before?"))
        if bt:
            verdict = ("a statistically meaningful link" if bt["p"] < 0.05 else
                       "no reliable link (what you would expect from noise)")
            blocks.append(text(f"GDELT's daily tone for Indian-market news, {bt['from']:%d %b} to {bt['to']:%d %b %Y} ({bt['n']} days): "
                               f"change in tone vs the next session's NIFTY return has rank correlation **{bt['rho']:+.2f}** "
                               f"(p = {bt['p']:.2f}), {verdict}. Free news archives are short, so this is why news only "
                               "nudges the outlook (max ±3 points) instead of driving it."))
        f = figs.lines(gd.index, {"GDELT tone (India markets news)": dict(y=gd, color=figs.BLUE, width=1.6)}, "Daily news tone, last 90 days")
        blocks.append(plot(f, 260))
    blocks.append(section("Headlines"))
    blocks.append(news_list(_items(news.head(60))))
    blocks.append(section("Source status"))
    st = pd.DataFrame([{"Source": k, "Headlines": v if v != "unreachable" else None,
                        "Status": "ok" if v != "unreachable" else "unreachable"} for k, v in status.items()])
    blocks.append(table(st, [col("Source", "Source"), col("Headlines", "Headlines", "int"), col("Status", "Status", "badge")]))
    return reply(doc(blocks, "live", tone=t["tone"]), ttl=300)


# ------------------------------------------------------------------ accuracy


@router.get("/accuracy")
def accuracy(target: float = Query(0.05, ge=0.01, le=0.3), h: int = Query(21, ge=3, le=63), n: int = Query(20, ge=8, le=40)):
    """Walk-forward accuracy of 'chance of reaching +target within h days' on live data for the largest NIFTY stocks."""
    from jinni import accuracy as ac
    from jinni.universe import NIFTY50

    from .core import nifty50

    tk = nifty50()[:n]

    def build():
        pdx = prices(tk + [BENCHMARK], "5y", ttl=3600)
        dfs = {t: pdx.ohlcv(t) for t in tk if t in pdx.close and pdx.close[t].notna().sum() > 500}
        bench = pdx.close[BENCHMARK].dropna()
        return ac.collect(dfs, bench, target, h, step=10, n_paths=400), pdx.source

    s, source = memo(("accuracy", tuple(tk), target, h), 6 * 3600, build)
    if s.empty:
        raise ApiError(422, "Not enough price history to measure accuracy.")
    sm_ = ac.summary(s)
    cal = ac.calibration(s["p_sim"], s["hit"])
    sig = ac.signal_table(s)
    blocks = source_notice(source)
    blocks.append(callout("info", f"**Walk-forward test.** For {sm_['stocks']} stocks, every 10th day from {sm_['from']:%b %Y} to "
                                  f"{sm_['to']:%b %Y} ({sm_['n']:,} predictions) the model saw only the past, stated the chance of "
                                  f"touching +{target:.0%} within {h} trading days, and we then checked what happened."))
    skill = sm_["brier_base"] - sm_["brier_sim"]
    blocks.append(kpis([
        ("Real hit rate", pct(sm_["base"], 0), f"of all predictions reached +{target:.0%}"),
        ("Model's average", pct(sm_["mean_pred"], 0), "average stated chance"),
        ("Ranking skill (AUC)", f"{sm_['auc_sim']:.2f}", "0.50 = no skill, 1.00 = perfect", "up" if sm_["auc_sim"] >= 0.58 else None),
        ("Calibration error (Brier)", f"{sm_['brier_sim']:.3f}", f"vs {sm_['brier_base']:.3f} for always saying {sm_['base']:.0%}",
         "up" if skill > 0 else "down")]))
    blocks.append(kpis([
        ("Avg result per trade", pct(sm_["ret_all"], 2, True), f"+{target:.0%} target, auto stop, after 0.3% costs", tone(sm_["ret_all"])),
        ("Best-ranked fifth", pct(sm_["ret_top"], 2, True), f"{pct(sm_['hit_top'], 0)} reached the target", tone(sm_["ret_top"]))]))
    f = figs.fig([{"type": "scatter", "mode": "markers+lines", "x": cal["Predicted"].tolist(), "y": cal["Actual"].tolist(),
                   "name": "Model", "marker": {"size": 10, "color": figs.GOLD}, "line": {"color": figs.GOLD}},
                  {"type": "scatter", "mode": "lines", "x": [0, 1], "y": [0, 1], "name": "Perfect", "line": {"color": figs.MUTED, "dash": "dot"}}],
                 {"title": "When the model says X%, how often did it happen?", "xaxis": {"title": "Stated chance", "tickformat": ".0%"},
                  "yaxis": {"title": "What actually happened", "tickformat": ".0%"}})
    blocks.append(plot(f, 340))
    blocks.append(table(cal, [col("Samples", "Predictions", "int"), col("Predicted", "Stated chance", "pct0"), col("Actual", "Actually reached", "pct0")]))
    blocks.append(section("Which signals can rank stocks that reach the target?"))
    blocks.append(table(sig, [col("Signal", "Signal"), col("AUC", "AUC", "n3"), col("Top 20% hit", "Top 20% hit rate", "pct0"),
                              col("Bottom 20% hit", "Bottom 20% hit rate", "pct0"), col("Base rate", "Base rate", "pct0"),
                              col("Samples", "Samples", "int")]))
    tech = sig[sig["Signal"] == "Technical score"]
    vol = sig[sig["Signal"] == "Volatility"]
    lines = []
    if len(vol):
        lines.append(f"- **Volatility does the heavy lifting**: ranking by volatility alone gives AUC {vol['AUC'].iloc[0]:.2f}. "
                     "Stocks that move more reach any target more often, and they also hit stops more often.")
    if len(tech):
        a = tech["AUC"].iloc[0]
        lines.append(f"- **Technical score**: AUC {a:.2f}. " + ("No usable ranking power on this sample, so JINNI does not use it to rank, only to flag." if a < 0.54
                     else "Some ranking power."))
    lines.append("- **Sentiment and fundamentals** have no free point-in-time history, so they cannot be back-tested here. They are "
                 "shown as watch-outs and, for the outlook, as a capped tilt.")
    blocks.append(text("\n".join(lines)))
    blocks.append(caption("Limits: large, liquid NIFTY stocks only; windows overlap, so the effective sample is smaller than the count; "
                          "survivorship bias (today's index members); 'reached' means the day's high touched the level. Past accuracy "
                          "does not guarantee future accuracy."))
    return reply(doc(blocks, source, auc=sm_["auc_sim"], base=sm_["base"]), ttl=1800)
