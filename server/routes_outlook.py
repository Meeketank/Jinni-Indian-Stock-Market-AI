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
        return ac.collect(dfs, bench, target, h, step=10, n_paths=400), pdx.source, ac.range_coverage(dfs, h, step=15, n_paths=400)

    s, source, cov = memo(("accuracy", tuple(tk), target, h), 6 * 3600, build)
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
    if cov.get("n"):
        blocks.append(section("Are the price-forecast ranges honest?"))
        blocks.append(kpis([
            ("Inside the 9-in-10 range", pct(cov["in90"], 0), "should be about 90%", "up" if abs(cov["in90"] - 0.9) < 0.05 else "down"),
            ("Inside the half range", pct(cov["in50"], 0), "should be about 50%", "up" if abs(cov["in50"] - 0.5) < 0.07 else "down"),
            ("Ended above the middle", pct(cov["above_mid"], 0), "should be about 50%"),
            ("Typical range width", pct(cov["avg_width"], 0), f"low to high, {h} days ahead")]))
        blocks.append(caption(f"{cov['n']:,} forecasts on {cov['stocks']} stocks: each range was made with only the past, then compared "
                              f"with the price {h} trading days later. This is the test behind the Forecast tab."))
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


# ------------------------------------------------------------------ scenarios


def _scenario_series(symbol: str):
    """(close series, display label, source). Indices use the longest history Yahoo has so rare events are covered."""
    from jinni.universe import normalize_symbol

    sym = normalize_symbol(symbol)
    if sym.startswith("^"):
        pdx = prices([sym], "max", ttl=3600)
        return pdx.close[sym].dropna(), display_name(sym), pdx.source, sym
    c = ctx(sym)
    return c.df["Close"], display_name(c.t), c.source, c.t


@router.get("/scenarios/events")
def scenarios_events(symbol: str = "^NSEI"):
    """For each kind of scheduled event: the level if it goes well, the typical level, and the level if it goes badly."""
    from jinni import scenarios as sc
    from jinni import seasonality as se

    close, label, source, sym = _scenario_series(symbol)
    last, today = float(close.iloc[-1]), close.index[-1]
    blocks = source_notice(source)
    blocks.append(callout("warn", "**Scenarios, not predictions.** Nobody knows the outcome of an election, a rate decision or a Budget "
                                  "in advance. What can be measured is what each kind of outcome was worth before: the best third of "
                                  "past reactions ('goes well'), the middle ('typical') and the worst third ('goes badly'), turned into "
                                  f"price levels from today's {label} at ₹{last:,.2f}."))
    try:
        news, _ = _news()
        radar = nf.event_radar(news) if len(news) else pd.DataFrame()
    except Exception:  # noqa: BLE001
        radar = pd.DataFrame()
    hot = {"Election": "Election / results", "RBI": "RBI policy", "Fed": "US Fed (FOMC)", "Budget": "Union Budget"}
    tabs_, summary = [], []
    for name, (dates, off) in se.BUILTIN_EVENTS.items():
        br = sc.branches(close, dates, off)
        if not br or not br["rows"]:
            continue
        nxt = [d for d in pd.to_datetime(dates) if d > today]
        in_news = next((int(radar.loc[radar["Event"] == v, "Mentions"].iloc[0]) for k, v in hot.items()
                        if k.lower() in name.lower() and len(radar) and (radar["Event"] == v).any()), 0)
        b = [kpis([("Past occurrences", str(br["n"]), "in this price history"),
                   ("Next scheduled", f"{min(nxt):%d %b %Y}" if nxt else "not scheduled", f"in {(min(nxt) - today).days} days" if nxt else "dates end with the last known one"),
                   ("In today's news", f"{in_news} headlines" if in_news else "quiet", "live event radar")])]
        b.append(plot(figs.fork_chart(close.iloc[-60:], br["rows"], f"{label} after: {name}"), 360))
        t = pd.DataFrame(br["rows"])
        b.append(table(t, [col("When", "When"), col("Unfav ₹", "If it goes badly ₹", "inr2"), col("Base ₹", "Typical ₹", "inr2"),
                           col("Fav ₹", "If it goes well ₹", "inr2"), col("Unfavourable", "Badly %", "spct2"), col("Base", "Typical %", "spct2"),
                           col("Favourable", "Well %", "spct2"), col("Worst", "Worst ever", "spct2"), col("Best", "Best ever", "spct2"),
                           col("Up %", "Rose", "pct0"), col("n", "Events", "int")]))
        bt = sc.branch_backtest(close, dates, off)
        sz = sc.event_day_size(close, dates, off)
        notes = []
        if bt:
            w = bt.get("1 week later") or next(iter(bt.values()))
            notes.append(f"**Back-test:** replaying the past {w['n']} occurrences using only earlier ones, the 1-week result landed between "
                         f"the 'badly' and 'well' levels {w['inside']:.0%} of the time, and inside the full earlier range {w['inside_extremes']:.0%}.")
        else:
            notes.append("**Back-test:** too few past occurrences to replay (needs at least 7). Treat these levels as rough.")
        if sz:
            notes.append(f"Event days moved {sz['event']:.2%} on average vs {sz['normal']:.2%} on ordinary days "
                         f"({'bigger' if sz['ratio'] > 1.15 else 'smaller' if sz['ratio'] < 0.85 else 'about the same'}).")
        b.append(text(" ".join(notes)))
        h = br["history"].reset_index()
        b.append(table(h, [col("Date", "Past event", "date")] + [col(k, k, "spct2") for k in sc.HORIZONS], height=320))
        tabs_.append((min(nxt) if nxt else pd.Timestamp.max, name.split(" →")[0], b))
        wk = next((r for r in br["rows"] if r["When"] == "1 week later"), br["rows"][0])
        summary.append({"Event": name.split(" →")[0], "Next": min(nxt) if nxt else None, "Badly ₹": wk["Unfav ₹"], "Typical ₹": wk["Base ₹"],
                        "Well ₹": wk["Fav ₹"], "Events": br["n"]})
    if not tabs_:
        blocks.append(callout("info", f"{label} does not have enough price history covering past events to build scenarios."))
        return reply(doc(blocks, source), ttl=600)
    tabs_.sort(key=lambda x: x[0])
    blocks.append(section("One week after each kind of event"))
    blocks.append(table(pd.DataFrame(summary), [col("Event", "Event"), col("Next", "Next date", "date"), col("Badly ₹", "If badly ₹", "inr2"),
                                                col("Typical ₹", "Typical ₹", "inr2"), col("Well ₹", "If well ₹", "inr2"), col("Events", "Past events", "int")]))
    blocks.append({"type": "tabs", "tabs": [{"label": l, "blocks": b} for _, l, b in tabs_]})
    blocks.append(caption("Levels use closes: the close before the event to the close on the event day, 5 and 21 sessions later. "
                          "Event dates are built in up to the last published schedule; add your own in the Event lab."))
    return reply(doc(blocks, source), ttl=600)


@router.get("/scenarios/whatif")
def scenarios_whatif(symbol: str, nifty: float = Query(0, ge=-50, le=50), sp500: float = Query(0, ge=-50, le=50),
                     crude: float = Query(0, ge=-60, le=100), usdinr: float = Query(0, ge=-20, le=20),
                     gold: float = Query(0, ge=-40, le=60)):
    """Expected move of a stock if the market, US equities, crude, the rupee or gold jump (inputs in percent)."""
    from jinni import scenarios as sc

    c = ctx(symbol)
    fx = memo(("factors",), 3600, lambda: prices(list(sc.FACTORS), "5y", ttl=3600))
    fit = memo(("factorfit", c.t, str(c.df.index[-1])), 3600, lambda: sc.fit_factors(c.df["Close"], fx.close))
    if not fit:
        raise ApiError(422, "Not enough overlapping history to estimate this stock's sensitivities.")
    price = float(c.df["Close"].iloc[-1])
    blocks = source_notice(c.source)
    custom = {"^NSEI": nifty / 100, "^GSPC": sp500 / 100, "BZ=F": crude / 100, "USDINR=X": usdinr / 100, "GC=F": gold / 100}
    custom = {k: v for k, v in custom.items() if v}
    if custom:
        w = sc.what_if(fit, custom)
        desc = ", ".join(f"{sc.FACTORS[k]} {v:+.0%}" for k, v in custom.items())
        blocks.append(kpis([("Your scenario", pct(w["move"], 1, True), desc, tone(w["move"])),
                            ("Price in that scenario", f"₹{price * (1 + w['move']):,.2f}", f"now ₹{price:,.2f}"),
                            ("Likely band", f"₹{price * (1 + w['low']):,.0f} – ₹{price * (1 + w['high']):,.0f}", "±1 week of stock-specific noise")]))
    rows = []
    for name, sh in sc.PRESETS.items():
        w = sc.what_if(fit, sh)
        rows.append({"Scenario": name, "Move": w["move"], "Price ₹": price * (1 + w["move"]), "Low ₹": price * (1 + w["low"]), "High ₹": price * (1 + w["high"])})
    t = pd.DataFrame(rows)
    f = figs.bar_h(t["Scenario"], t["Move"], f"{display_name(c.t)}: expected move in each scenario", ".1%",
                   xrange=[float(t["Move"].min() * 1.6 - 0.01), float(t["Move"].max() * 1.6 + 0.01)])
    blocks.append(plot(f, 120 + 40 * len(t)))
    blocks.append(table(t, [col("Scenario", "Scenario"), col("Move", "Expected move", "spct1"), col("Price ₹", "Price ₹", "inr2"),
                            col("Low ₹", "Low ₹", "inr2"), col("High ₹", "High ₹", "inr2")]))
    sens = pd.DataFrame([{"Driver": sc.FACTORS[k], "Sensitivity": v, "Meaning": f"a +1% move in {sc.FACTORS[k]} has gone with {v:+.2f}% in this stock (other drivers held fixed)"}
                         for k, v in fit["betas"].items()])
    blocks.append(section("Sensitivities and how far to trust them"))
    blocks.append(table(sens, [col("Driver", "Driver"), col("Sensitivity", "Beta", "n2"), col("Meaning", "Meaning")]))
    trust = "strong" if fit["oos_r2"] > 0.4 else "moderate" if fit["oos_r2"] > 0.15 else "weak"
    blocks.append(text(f"**Back-test:** fitted on the first {fit['weeks'] - fit['oos_weeks']} weeks and checked on the next {fit['oos_weeks']} it had "
                       f"not seen, these drivers explained **{max(fit['oos_r2'], 0):.0%}** of the stock's weekly moves (correlation "
                       f"{fit['oos_corr']:.2f}): a {trust} guide. The rest is company-specific news, which is why each scenario has a band."))
    blocks.append(caption("Weekly returns over the last three years, ordinary least squares. Scenarios assume the historical relationship "
                          "holds and say nothing about how likely each shock is."))
    return reply(doc(blocks, c.source, r2=fit["r2"], oos_r2=fit["oos_r2"]), ttl=600)
