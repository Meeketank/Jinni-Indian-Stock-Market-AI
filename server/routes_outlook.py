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
