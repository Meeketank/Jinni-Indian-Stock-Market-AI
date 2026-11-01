"""Market-wide endpoints: health, catalog, search, home, mood, news, event studies, calendar."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query

from jinni import data as jdata
from jinni import events as jevents
from jinni import indicators as ind
from jinni import market, ta
from jinni import seasonality as se
from jinni import sentiment as sm
from jinni.universe import (BENCHMARK, INDICES, MACRO, NIFTY50, SECTOR_INDICES, STRESS_SCENARIOS, display_name,
                            normalize_symbol, sector_of)

from . import figs
from .core import (ApiError, nifty50, caption, callout, col, cols, doc, inr, kpis, memo, news_list, num, pct, plot, prices,
                   reply, section, source_notice, table, tabs, tape, text, tone)

router = APIRouter()


# ----------------------------------------------------------------- health


@router.get("/health")
def health(deep: int = 0):
    out = {"ok": True, "demo": jdata.demo_forced()}
    if deep:
        try:
            p = jdata.get_prices(["^NSEI"], "5d", allow_demo=False) if not jdata.demo_forced() else None
            out["live_data"] = bool(p is not None and p.source == "live")
            out["latest"] = str(p.close.index[-1].date()) if p is not None else None
        except Exception as e:  # noqa: BLE001
            out["live_data"] = False
            out["error"] = str(e)[:200]
    return reply(out)


# ---------------------------------------------------------------- catalog


@router.get("/catalog")
def catalog():
    from jinni import backtest as bt
    from jinni import forecast as fc
    from jinni import optimizer as opt
    from jinni import setups as jsetups

    inds = [{"name": k, "cat": v.cat, "kind": v.kind, "desc": v.desc} for k, v in ta.CATALOG.items()]
    nifty = [{"t": t, "name": display_name(t), "sector": s} for t, s in NIFTY50.items()]
    return reply({
        "indicators": inds, "categories": ta.CATEGORIES, "layers": figs.LAYERS, "nifty50": nifty,
        "horizons": fc.HORIZONS, "strategies": bt.STRATEGIES, "objectives": opt.OBJECTIVES,
        "events": list(se.BUILTIN_EVENTS), "indices": {k: v for k, v in INDICES.items()},
        "stress": list(STRESS_SCENARIOS), "setups": list(jsetups.SETUPS), "sectors": sorted(set(NIFTY50.values())),
        "nifty50_source": NIFTY50.source,
    }, ttl=3600 if nifty else 60)  # do not let a failed index lookup stick in the CDN


def _local_hits(q: str, limit: int = 12) -> list[dict]:
    """Match against every NSE-listed name (live list -> today's cache -> bundled snapshot)."""
    def load():
        try:
            return market.demo_universe(1000) if jdata.demo_forced() else market.nse_universe()
        except RuntimeError:
            return pd.DataFrame({"symbol": [], "name": []})  # Yahoo search below still answers

    uni = memo(("search-universe", jdata.demo_forced()), 3600, load)
    words = [w for w in q.replace("&", " & ").split() if w]
    hits = []
    for sym, name in zip(uni["symbol"], uni["name"]):
        s, nm = sym.upper(), str(name).upper()
        if s == q.replace(" ", ""):
            score = 0
        elif s.startswith(q.replace(" ", "")):
            score = 1
        elif all(w in nm or w in s for w in words):
            score = 2 if nm.startswith(words[0]) else 3
        else:
            continue
        hits.append((score, len(s), sym, str(name)))
    hits.sort()
    return [{"t": f"{s}.NS", "name": n.title(), "symbol": s, "src": "NSE"} for _, _, s, n in hits[:limit]]


def _yahoo_hits(q: str, limit: int = 10) -> list[dict]:
    """Yahoo's own search: covers every NSE and BSE listing, including ones newer than our list."""
    if jdata.demo_forced():
        return []
    try:
        res = jdata._yf().Search(q, max_results=limit * 2, news_count=0, lists_count=0, recommended=0, timeout=6,
                                 raise_errors=False).quotes
    except Exception:  # noqa: BLE001
        return []
    out = []
    for r in res or []:
        sym = str(r.get("symbol", ""))
        if r.get("quoteType") not in (None, "EQUITY", "ETF") or not sym.endswith((".NS", ".BO")):
            continue
        out.append({"t": sym, "name": (r.get("longname") or r.get("shortname") or sym).title(),
                    "symbol": sym.rsplit(".", 1)[0], "src": "NSE" if sym.endswith(".NS") else "BSE"})
    return out[:limit]


@router.get("/search")
def search(q: str = Query("", max_length=40)):
    q = " ".join(q.strip().upper().split())
    if not q:
        return reply({"results": []}, ttl=3600)

    def build():
        merged, seen = [], set()
        for r in _local_hits(q) + _yahoo_hits(q):
            key = r["symbol"]  # same company on NSE and BSE: keep the NSE line
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
        return merged[:15]

    return reply({"results": memo(("search", q), 3600, build)}, ttl=600)


# ------------------------------------------------------------------- home


def _board_table(df: pd.DataFrame):
    df = df.assign(Cap=df["Mkt cap"] / 1e7)
    cc = [col("Stock", "Stock"), col("Price", "Price ₹", "inr2"), col("1D", "Today", "spct2"),
          col("Volume", "Volume", "int"), col("Cap", "Mkt cap ₹ cr", "n0"), col("52w high", "52w high ₹", "inr2"),
          col("P/E", "P/E", "n1")]
    return table(df, cc, select={"key": "t", "action": "analyze"})


@router.get("/home/pulse")
def home_pulse():
    """Indices, macro, VIX and the NIFTY chart: ~10 symbols, so it is quick and loads first."""
    idx_t, mac_t = list(INDICES), list(MACRO)
    pdx = memo(("home-pulse",), 90, lambda: prices(idx_t + mac_t, "1y", ttl=90))
    c = pdx.close
    blocks = source_notice(pdx.source)
    tp = []
    for t in idx_t + mac_t:
        if t in c:
            s = c[t].dropna()
            if len(s) > 1:
                tp.append((display_name(t), float(s.iloc[-1]), float(s.iloc[-1] / s.iloc[-2] - 1)))
    if tp:
        blocks.append(tape(tp))
    items = []
    nifty = c[BENCHMARK].dropna() if BENCHMARK in c else None
    if nifty is not None and len(nifty) > 200:
        last, prev = nifty.iloc[-1], nifty.iloc[-2]
        s50, s200 = ind.sma(nifty, 50).iloc[-1], ind.sma(nifty, 200).iloc[-1]
        regime, rcls = (("Bull trend", "up") if last > s50 > s200 else ("Bear trend", "down") if last < s50 < s200
                        else ("Range / transition", "gold"))
        items.append(("NIFTY 50", f"{last:,.0f}", pct(last / prev - 1, 2, True), tone(last - prev)))
        items.append(("Market regime", regime, f"50-DMA {s50:,.0f} · 200-DMA {s200:,.0f}", rcls))
        items.append(("NIFTY 20d vol", pct(ind.realized_vol(nifty, 20).iloc[-1], 1), "annualised realised"))
    if "^INDIAVIX" in c:
        vix = c["^INDIAVIX"].dropna()
        if len(vix) > 1:
            lvl = vix.iloc[-1]
            mood = "Calm" if lvl < 13 else "Normal" if lvl < 18 else "Elevated" if lvl < 24 else "Fear"
            items.append(("India VIX", f"{lvl:.2f}", f"{mood} · {pct(lvl / vix.iloc[-2] - 1, 2, True)}",
                          "down" if lvl >= 18 else "up"))
    if items:
        blocks.append(kpis(items))
    if nifty is not None:
        lo, hi = float(nifty.min()), float(nifty.max())
        f = figs.lines(nifty.index, {
            "NIFTY 50": dict(y=nifty, color=figs.GOLD, width=2, fill="tozeroy", fillcolor="rgba(245,184,65,0.06)"),
            "50-DMA": dict(y=ind.sma(nifty, 50), color=figs.BLUE, width=1.2),
            "200-DMA": dict(y=ind.sma(nifty, 200), color=figs.MUTED, width=1.2, dash="dot")},
            title="NIFTY 50 · 1 year", yrange=(lo * .97, hi * 1.02))
        blocks.append(plot(f, 320))
    blocks.append(caption(f"As of {c.index[-1]:%d %b %Y}. Daily bars from Yahoo Finance (may lag the exchange)."))
    return reply(doc(blocks, pdx.source), ttl=60)


@router.get("/home/sectors")
def home_sectors():
    sec_t = list(SECTOR_INDICES)
    pdx = memo(("home-sectors",), 300, lambda: prices(sec_t, "1y", ttl=300))
    c = pdx.close
    blocks = source_notice(pdx.source)
    rows = []
    for t, nm in SECTOR_INDICES.items():
        if t in c:
            s = c[t].dropna()
            if len(s) > 22:
                rows.append({"Sector": nm, "1D": s.iloc[-1] / s.iloc[-2] - 1, "1W": s.iloc[-1] / s.iloc[-6] - 1,
                             "1M": s.iloc[-1] / s.iloc[-22] - 1, "3M": s.iloc[-1] / s.iloc[max(-64, -len(s))] - 1})
    if rows:
        sdf = pd.DataFrame(rows).set_index("Sector").sort_values("1M", ascending=False)
        z = sdf.values * 100
        lim = max(1.0, float(np.nanmax(np.abs(z))))
        f = figs.heatmap(z, sdf.columns, sdf.index, "Sector rotation", -lim, lim,
                         [[0, figs.DOWN], [0.5, "var(--panel2)"], [1, figs.UP]],
                         [[f"{v:+.1f}%" for v in r] for r in z], fmt=".1f")
        blocks.append(plot(f, 360))
    else:
        blocks.append(callout("warn", "Sector index data is not available right now."))
    return reply(doc(blocks, pdx.source), ttl=180)


@router.get("/home/movers")
def home_movers(min_cr: float = Query(1.0, ge=0, le=100)):
    """Gainers, losers and most-active across ALL listed stocks, NSE and BSE (one row per company, NSE preferred).

    Source: the exchanges' official end-of-day files, so every listing is covered and the date is exact. Stocks that
    traded less than ``min_cr`` crore rupees are left out so a handful of shares cannot top the list."""
    from jinni import bhav

    df, info = memo(("bhav",), 900, bhav.latest)
    blocks = []
    if df.empty:
        # Last resort: Yahoo's live screener (NSE + BSE) when the exchange files cannot be fetched.
        from jinni import live

        boards = memo(("home-movers-live",), 60, lambda: {k: live.board(k, 15) for k in live.BOARDS})
        got = {k: v for k, v in boards.items() if v is not None}
        if not got:
            blocks.append(callout("warn", "Neither the NSE/BSE end-of-day files nor the live screener could be reached right now, "
                                          "so there are no movers to show. JINNI will not guess."))
            return reply(doc(blocks, "live"))
        names = {"gainers": "Top gainers", "losers": "Top losers", "active": "Most active"}
        blocks.append(tabs([(names[k], [_board_table(v)]) for k, v in got.items()]))
        blocks.append(caption("Live screener fallback (NSE and BSE)."))
        return reply(doc(blocks, "live"), ttl=60)
    d = df[(df["turnover"] >= min_cr * 1e7) & (df["close"] >= 5)].copy()
    d["Cap"] = d["turnover"] / 1e7
    d["Stock"] = d["name"] + d["exch"].map(lambda e: "" if e == "NSE" else " (BSE)")
    cc = [col("Stock", "Stock"), col("exch", "Exch"), col("close", "Price ₹", "inr2"), col("chg", "Today", "spct2"),
          col("Cap", "Turnover ₹ cr", "n1"), col("volume", "Volume", "int")]
    sel = {"key": "ticker", "action": "analyze"}
    cols_ = ["ticker", "Stock", "exch", "close", "chg", "Cap", "volume"]
    when = f"{info['date']:%d %b %Y}" if info.get("date") else "latest"
    blocks.append(tabs([
        ("Top gainers", [table(d.nlargest(20, "chg")[cols_], cc, select=sel)]),
        ("Top losers", [table(d.nsmallest(20, "chg")[cols_], cc, select=sel)]),
        ("Most active", [table(d.nlargest(20, "Cap")[cols_], cc, select=sel)])]))
    adv, dec = int((d["chg"] > 0).sum()), int((d["chg"] < 0).sum())
    blocks.append(kpis([("Stocks covered", f"{len(df):,}", f"{(df['exch'] == 'NSE').sum():,} NSE · {(df['exch'] == 'BSE').sum():,} BSE-only"),
                        ("Advancing / declining", f"{adv:,} / {dec:,}", f"stocks trading ≥ ₹{min_cr:g} cr", "up" if adv >= dec else "down"),
                        ("Data", when, info.get("source", ""))]))
    blocks.append(caption(f"All NSE and BSE listings from the exchanges' official end-of-day files, {when} close. Cross-listed "
                          "companies appear once (NSE). BSE-only stocks are marked (BSE). Tap a row to analyse it."))
    return reply(doc(blocks, "live"), ttl=300)


@router.get("/home/nifty50")
def home_nifty50():
    stk_t = nifty50()
    pdx = memo(("home-n50",), 120, lambda: prices(stk_t, "1y", ttl=120))
    c = pdx.close
    blocks = source_notice(pdx.source, pdx.missing)
    sc = c[[t for t in stk_t if t in c]]
    if len(sc) <= 50:
        blocks.append(callout("warn", "NIFTY 50 prices are not available right now."))
        return reply(doc(blocks, pdx.source))
    above50 = (sc.iloc[-1] > sc.rolling(50).mean().iloc[-1]).mean()
    above200 = (sc.iloc[-1] > sc.rolling(200, min_periods=150).mean().iloc[-1]).mean()
    adv, dec = int((sc.iloc[-1] > sc.iloc[-2]).sum()), int((sc.iloc[-1] < sc.iloc[-2]).sum())
    blocks.append(kpis([("Breadth (>50-DMA)", pct(above50, 0), f">200-DMA {pct(above200, 0)}", "up" if above50 >= .5 else "down"),
                        ("Advance / Decline", f"{adv} / {dec}", "NIFTY 50 today", "up" if adv >= dec else "down")]))
    chg = pd.DataFrame({"Stock": [display_name(t) for t in sc.columns], "Price": sc.iloc[-1].values,
                        "1D": (sc.iloc[-1] / sc.iloc[-2] - 1).values, "1W": (sc.iloc[-1] / sc.iloc[-6] - 1).values,
                        "1M": (sc.iloc[-1] / sc.iloc[-22] - 1).values,
                        "From 52w high": (sc.iloc[-1] / sc.max() - 1).values, "t": list(sc.columns)}).dropna(subset=["1D"])
    cc = [col("Stock", "Stock"), col("Price", "Price ₹", "inr2"), col("1D", "1D", "spct2"), col("1W", "1W", "spct1"),
          col("1M", "1M", "spct1"), col("From 52w high", "vs 52w high", "pct1")]
    blocks.append(table(chg, cc, select={"key": "t", "action": "analyze"}, sort=("1D", True), height=520))
    return reply(doc(blocks, pdx.source), ttl=120)


# ------------------------------------------------------------------- mood


@router.get("/mood")
def mood():
    pdx = memo(("mood",), 300, lambda: prices(["^NSEI", "^INDIAVIX", "GC=F"] + nifty50(), "2y", ttl=300))
    nifty = pdx.close["^NSEI"].dropna()
    vix = pdx.close.get("^INDIAVIX")
    gold = pdx.close.get("GC=F")
    stk = pdx.close[[t for t in NIFTY50 if t in pdx.close]]
    m = sm.market_mood(nifty, vix, stk, gold)
    blocks = source_notice(pdx.source)
    sc = m["score"]
    g = figs.gauge(sc, (0, 100), "Market mood", m["label"], figs.GOLD,
                   [dict(range=[0, 20], color="rgba(239,68,68,.35)"), dict(range=[20, 40], color="rgba(239,68,68,.15)"),
                    dict(range=[60, 80], color="rgba(34,197,94,.15)"), dict(range=[80, 100], color="rgba(34,197,94,.35)")],
                   [0, 20, 40, 60, 80, 100])
    comp = pd.Series(m["components"]).dropna()
    b = figs.bar_h(comp.index, comp.values, "Components (percentile vs last year)", ".0f",
                   [figs.UP if v >= 60 else figs.DOWN if v < 40 else figs.GOLD for v in comp.values],
                   xrange=[0, 108])
    blocks.append(cols([plot(g, 280)], [plot(b, 280)]))
    hist = sm.mood_history(nifty, vix, stk).dropna()
    if len(hist) > 20:
        nn = nifty.loc[hist.index[0]:]
        f = figs.lines(hist.index, {"Mood": dict(y=hist, color=figs.GOLD, width=1.6),
                                    "NIFTY 50": dict(y=nn.reindex(hist.index), color=figs.MUTED, width=1, y2=True)},
                       "Mood history vs NIFTY", yrange=(0, 100))
        f["layout"]["shapes"] = [
            {"type": "rect", "xref": "paper", "yref": "y", "x0": 0, "x1": 1, "y0": 0, "y1": 20,
             "fillcolor": "rgba(239,68,68,.08)", "line": {"width": 0}},
            {"type": "rect", "xref": "paper", "yref": "y", "x0": 0, "x1": 1, "y0": 80, "y1": 100,
             "fillcolor": "rgba(34,197,94,.08)", "line": {"width": 0}}]
        blocks.append(plot(f, 300))
        fwd = nifty.shift(-21) / nifty - 1
        z = pd.DataFrame({"mood": hist, "fwd": fwd.reindex(hist.index)}).dropna()
        if len(z) > 60:
            z["Zone"] = pd.cut(z["mood"], [0, 20, 40, 60, 80, 100],
                               labels=["Extreme fear", "Fear", "Neutral", "Greed", "Extreme greed"])
            t = z.groupby("Zone", observed=True)["fwd"].agg(["size", "mean", lambda x: (x > 0).mean()])
            t.columns = ["Days", "NIFTY next month", "Up %"]
            blocks.append(text("**What happened next** (NIFTY's following month, by mood zone):"))
            blocks.append(table(t, [col("Zone", "Zone"), col("Days", "Days", "int"),
                                    col("NIFTY next month", "NIFTY next month", "spct2"), col("Up %", "Up %", "pct0")],
                                index=True, index_label="Zone"))
    blocks.append(caption("Contrarian reading: extreme fear has often preceded better returns than extreme greed, "
                          "but not reliably enough to time the market on its own."))
    return reply(doc(blocks, pdx.source, score=sc, label=m["label"]), ttl=300)


# ------------------------------------------------------------------- news


def news_summary(tickers: list[str]) -> tuple[list[dict], list[dict]]:
    rows, items = [], []
    for t in tickers:
        n = memo(("news", t), 600, lambda t=t: jevents.headlines(t, limit=12))
        s = sm.score_news(n)
        rows.append({"Ticker": display_name(t), "Tone": s["score"], "Mood": s["label"], "Headlines": s["n"],
                     "Positive": s["pos"], "Negative": s["neg"]})
        if not n.empty:
            for k, r in n.iterrows():
                items.append({"ticker": display_name(t), "time": r["Time"], "headline": r["Headline"],
                              "tags": r["Tags"], "link": r["Link"], "score": sm.score_headline(r["Headline"])})
    items.sort(key=lambda x: str(x["time"]), reverse=True)
    return rows, items


@router.get("/news")
def news(tickers: str = Query("^NSEI", max_length=400)):
    tk = [normalize_symbol(x) for x in tickers.replace(" ", ",").split(",") if x.strip()][:12]
    if not tk:
        raise ApiError(400, "Give at least one symbol.")
    rows, items = news_summary(tk)
    blocks = []
    if rows:
        blocks.append(table(pd.DataFrame(rows), [col("Ticker", "Ticker"), col("Tone", "Tone", "tone"),
                                                 col("Mood", "Mood"), col("Headlines", "Headlines", "int"),
                                                 col("Positive", "Pos", "int"), col("Negative", "Neg", "int")],
                            sort=("Tone", True)))
    if items:
        blocks.append(news_list(items[:60]))
    else:
        blocks.append(callout("info", "No headlines available right now (feeds unreachable, or demo mode)."))
    blocks.append(caption("Headline tone is a snapshot of the last few days: the free feed has no long archive, "
                          "so it can't be back-tested. Use the Event Lab with dated news to test reactions."))
    return reply(doc(blocks), ttl=300)


# --------------------------------------------------------- event study / lab


@router.get("/events/study")
def event_study(symbol: str = "^NSEI", years: str = "10y", mode: str = "rule", nth: str = "2nd", day: str = "Friday",
                event: str = "", dates: str = "", offset: int = 0, window: int = Query(5, ge=1, le=15),
                label: str = "Event"):
    if years not in ("3y", "5y", "10y"):
        years = "10y"
    sym = normalize_symbol(symbol)
    pdx = prices([sym], years, ttl=3600)
    if sym not in pdx.close:
        raise ApiError(404, f"No price history for {sym}.")
    close = pdx.close[sym].dropna()
    if mode == "rule":
        wds = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        if day not in wds:
            raise ApiError(400, "Unknown weekday.")
        wi = wds.index(day)
        if nth == "Every":
            ds = list(close.index[close.index.weekday == wi])
            label = f"Every {day}"
        else:
            ds = se.nth_weekday_dates(close.index, wi, {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "Last": -1}.get(nth, 2))
            label = f"{nth} {day} of the month"
    elif mode == "builtin":
        if event not in se.BUILTIN_EVENTS:
            raise ApiError(400, "Unknown built-in event.")
        ds, offset = se.BUILTIN_EVENTS[event]
        label = event
    else:
        ds = pd.to_datetime([x.strip() for x in dates.replace(",", "\n").splitlines() if x.strip()], errors="coerce")
        ds = ds.dropna()
        if len(ds) == 0:
            raise ApiError(400, "No valid dates found. Use YYYY-MM-DD.")
    res = se.event_study(close, ds, int(offset), window, label)
    s = res.summary
    blocks = source_notice(pdx.source)
    if not s["n"]:
        blocks.append(callout("warn", "None of those dates fall inside the price history."))
        return reply(doc(blocks), ttl=600)
    blocks.append(text(f"#### {label} → {display_name(sym)}"))
    blocks.append(callout("info", res.verdict))
    blocks.append(kpis([
        ("Events", f"{s['n']}", f"reaction: {'next' if offset else 'same'} session"),
        ("Up on the day", pct(s["hit"], 0), f"normal days {s['others_hit']:.0%}"),
        ("Average move", f"{s['mean']:+.2f}%", f"normal {s['others_mean']:+.2f}%", tone(s["mean"])),
        ("Typical size", f"{s['avg_abs']:.2f}%", f"normal {s['others_abs']:.2f}%"),
        ("Best / worst", f"{s['best']:+.1f}% / {s['worst']:+.1f}%"),
        ("Luck test (p)", f"{s['p_mean']:.2f}", "< 0.05 = unlikely to be chance")]))
    left, right = [], []
    if not res.path.empty:
        f = figs.lines(res.path.index, {"Average": dict(y=res.path["Avg %"], color=figs.GOLD, width=2),
                                        "Median": dict(y=res.path["Median %"], color=figs.BLUE, dash="dot")},
                       "Average path around the event (% vs day before)", extra_layout={"xaxis": {"title": "Trading days"}})
        sh, _ = figs.vline(0, figs.MUTED, "dot")
        f["layout"]["shapes"] = [sh]
        left.append(plot(f, 300))
    t = res.table
    right.append(plot(figs.bar_v([d.strftime("%Y-%m-%d") for d in t.index], t["Day-0 %"].values,
                                 "Reaction on each occurrence (%)", ".1f"), 300))
    blocks.append(cols(left, right))
    if len(res.by_year) > 1:
        by = res.by_year.reset_index().rename(columns={"index": "Year", res.by_year.index.name or "index": "Year"})
        by.columns = ["Year", "Events", "Avg day-0 %", "Up %"]
        blocks.append(text("**Consistency by year**"))
        blocks.append(table(by, [col("Year", "Year", "text"), col("Events", "Events", "int"),
                                 col("Avg day-0 %", "Avg day-0 %", "n2"), col("Up %", "Up %", "pct0")]))
    tt = t.reset_index().rename(columns={t.index.name or "index": "Date"})
    blocks.append({"type": "details", "title": "Every occurrence",
                   "blocks": [table(tt, [col("Date", "Date", "date")] + [col(c, c, "n2") for c in tt.columns[1:]])]})
    blocks.append(caption("If you test many rules, some will look significant by chance. Prefer effects that hold "
                          "across years and have an economic reason to exist."))
    return reply(doc(blocks, pdx.source), ttl=900)


@router.get("/events/calendar")
def event_calendar(symbol: str = "^NSEI"):
    sym = normalize_symbol(symbol)
    pdx = prices([sym], "10y", ttl=3600)
    if sym not in pdx.close:
        raise ApiError(404, f"No price history for {sym}.")
    cl = pdx.close[sym].dropna()
    cal = se.calendar_effects(cl)
    b = cal["_base"].iloc[0]
    blocks = source_notice(pdx.source)
    blocks.append(text(f"Baseline: average day {b['Avg %']:+.3f}%, up {b['Up %']:.0%} of days."))
    for key in ("Month", "Weekday", "Week of month"):
        t = cal[key].dropna(subset=["Avg %"])
        blocks.append(plot(figs.bar_v(t.index, t["Avg %"].values, f"By {key.lower()}: average daily return % "
                                      "(labels = % of days up)", ".2f", texts=[f"{u:.0%}" for u in t["Up %"]]), 280))
    blocks.append(text("**Special days**"))
    sp = cal["Special days"].reset_index().rename(columns={"index": "Day type"})
    blocks.append(table(sp, [col("Day type", "Day type"), col("Days", "Days", "int"), col("Avg %", "Avg %", "n3"),
                             col("Up %", "Up %", "pct0"), col("p-value", "p-value", "n3"),
                             col("Signif. (after testing all rows)", "Significant?", "bool")]))
    blocks.append(text("**Coming up in the next 30 days**"))
    up = se.upcoming(cl)
    if up.empty:
        blocks.append(caption("No recurring events in the next 30 days."))
    else:
        u = up.reset_index()
        blocks.append(table(u, [col("Date", "Date", "date"), col("Event", "Event"), col("Past avg %", "Past avg %", "n2"),
                                col("Past up %", "Past up %", "pct0"), col("Past events", "Events", "int")]))
    blocks.append(caption("F&O expiry follows NSE's schedule: monthly contracts expired on the last Thursday until "
                          "Aug 2025 and on the last Tuesday since Sep 2025."))
    return reply(doc(blocks, pdx.source), ttl=3600)
