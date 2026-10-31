"""Live news from many free sources, scored for tone and grouped by what moves Indian markets.

Sources (all free, no API key): Google News RSS (India edition, one search per topic), publisher RSS feeds
(Economic Times, Moneycontrol, Mint, Business Standard, BusinessLine), Yahoo Finance RSS, and GDELT's global
news tone timeline. Each source is fetched in parallel with a short timeout; a source that fails is reported
in ``status`` and the rest still answer. Nothing here is invented: no feed reachable means no headlines.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd

from .data import demo_forced
from .sentiment import score_headline

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36 JINNI"
TIMEOUT = 7

# Google News searches, one per theme. {when} keeps results recent.
GOOGLE_QUERIES = {
    "Markets": "Sensex OR Nifty OR \"Indian stock market\"",
    "Economy & RBI": "RBI OR \"India inflation\" OR \"India GDP\" OR \"repo rate\"",
    "Politics & elections": "India election OR \"exit poll\" OR \"assembly election\" OR \"Lok Sabha\" OR \"state election results\"",
    "Policy & budget": "\"Union Budget\" OR GST OR SEBI OR \"India tax\" OR \"government policy\" stocks",
    "Geopolitics": "war OR sanctions OR tariffs OR \"trade deal\" India markets",
    "Commodities": "crude oil OR Brent OR gold OR OPEC",
    "Global markets": "\"Wall Street\" OR \"US Federal Reserve\" OR \"Dow Jones\" OR Nasdaq",
    "FII/DII flows": "FII OR FPI OR DII flows India equities",
    "Rupee": "rupee dollar forex India",
}
PUBLISHER_FEEDS = {
    "Economic Times": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "Mint": "https://www.livemint.com/rss/markets",
    "Business Standard": "https://www.business-standard.com/rss/markets-106.rss",
    "BusinessLine": "https://www.thehindubusinessline.com/markets/feeder/default.rss",
}
YAHOO_FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=IN&lang=en-IN"
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"

# Which headlines belong to which theme (a headline can belong to several).
TOPICS = {
    "Politics & elections": r"\b(election|elections|exit polls?|poll results?|vote counting|counting day|assembly polls?|lok sabha|"
                            r"bypolls?|coalition|bjp|congress|nda|ndtv poll|mandate|manifesto|swearing-in)\b",
    "Central banks & rates": r"\b(rbi|mpc|repo rate|rate cut|rate hike|fed|fomc|powell|ecb|boj|interest rates?|monetary policy|"
                             r"bond yields?|treasury)\b",
    "Budget & policy": r"\b(budget|gst|sebi|tax|fiscal|disinvestment|subsidy|policy|regulat\w+|ministry|cabinet)\b",
    "Geopolitics & trade": r"\b(war|ceasefire|sanctions?|tariffs?|trade deal|conflict|missile|attack|border|china|russia|"
                            r"ukraine|israel|iran|pakistan|trump|geopolitic\w*)\b",
    "Commodities": r"\b(crude|brent|oil|opec|gold|silver|natural gas|commodit\w+)\b",
    "Global markets": r"\b(wall street|dow|nasdaq|s&p 500|nikkei|hang seng|us stocks?|global markets?|asian markets?)\b",
    "Flows & rupee": r"\b(fii|fpi|dii|foreign investors?|outflows?|inflows?|rupee|dollar|forex|usd/?inr)\b",
    "Earnings & corporate": r"\b(q[1-4]|results?|earnings|profit|revenue|ebitda|guidance|dividend|buyback|ipo|merger|acquisition)\b",
    "Indian markets": r"\b(sensex|nifty|bank nifty|dalal street|indian stocks?|indian equit\w+|nse|bse)\b",
}

# Events worth flagging before they happen. `soon` words mean it is imminent.
EVENTS = {
    "Election / results": r"\b(election|elections|exit polls?|poll results?|counting|vote count\w*|assembly polls?)\b",
    "RBI policy": r"\b(rbi (mpc|policy|monetary)|mpc meeting|repo rate|monetary policy committee)\b",
    "US Fed (FOMC)": r"\b(fomc|fed (meeting|decision|policy)|powell|federal reserve)\b",
    "Union Budget": r"\b(union budget|budget (session|2\d{3}|speech)|fiscal deficit target)\b",
    "Inflation / GDP data": r"\b(cpi|inflation data|gdp data|gdp numbers|iip|wpi)\b",
    "Earnings season": r"\b(q[1-4] (results|earnings)|earnings season|results season)\b",
    "Trade / tariffs": r"\b(tariffs?|trade deal|trade talks|trade war|reciprocal)\b",
    "Geopolitical risk": r"\b(war|missile|ceasefire|escalat\w+|conflict|sanctions)\b",
    "OPEC / crude": r"\b(opec|crude (oil )?(price|surge|slump))\b",
}
SOON = re.compile(r"\b(today|tomorrow|tonight|this week|next week|on (monday|tuesday|wednesday|thursday|friday)|"
                  r"ahead of|eyes|awaits?|key (event|data)|d-day|results? day|counting day|(\d+) days? (to|until|left))\b", re.I)


def _get(url: str, timeout: int = TIMEOUT) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:  # noqa: BLE001 - any network failure just means this source is unavailable
        return None


def parse_rss(raw: bytes, source: str, theme: str | None = None) -> list[dict]:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    rows = []
    for it in root.iter("item"):
        title = re.sub(r"\s+", " ", (it.findtext("title") or "")).strip()
        if not title:
            continue
        pub = it.findtext("source") or None  # Google News names the real publisher here
        if pub and title.endswith(f" - {pub}"):
            title = title[: -len(pub) - 3].strip()
        try:
            ts = parsedate_to_datetime(it.findtext("pubDate")).astimezone()
        except Exception:  # noqa: BLE001
            ts = None
        rows.append({"Time": ts, "Headline": title, "Source": pub or source, "Link": it.findtext("link") or "",
                     "Feed": source, "Theme": theme})
    return rows


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower())[:90]


def classify(title: str) -> list[str]:
    t = title.lower()
    return [k for k, rx in TOPICS.items() if re.search(rx, t)]


def _google(q: str, days: int = 3) -> str:
    return ("https://news.google.com/rss/search?q=" + urllib.parse.quote(f"{q} when:{days}d")
            + "&hl=en-IN&gl=IN&ceid=IN:en")


def fetch_all(extra_queries: dict[str, str] | None = None, tickers: list[str] | None = None,
              max_age_days: float = 4, general: bool = True) -> tuple[pd.DataFrame, dict]:
    """(headlines, status). Every row is scored and tagged. ``status`` = {source: count or error text}."""
    if demo_forced():
        return pd.DataFrame(columns=["Time", "Headline", "Source", "Link", "Feed", "Theme", "Tone", "Topics"]), {}
    jobs: list[tuple[str, str, str | None]] = []  # (label, url, theme)
    for theme, q in {**(GOOGLE_QUERIES if general else {}), **(extra_queries or {})}.items():
        jobs.append((f"Google News · {theme}", _google(q), theme))
    for name, url in (PUBLISHER_FEEDS if general else {}).items():
        jobs.append((name, url, None))
    for t in tickers or []:
        jobs.append((f"Yahoo · {t}", YAHOO_FEED.format(t=t), None))

    def run(job):
        label, url, theme = job
        raw = _get(url)
        return label, (parse_rss(raw, label, theme) if raw else None)

    status, rows = {}, []
    with ThreadPoolExecutor(max_workers=12) as ex:
        for label, got in ex.map(run, jobs):
            status[label] = "unreachable" if got is None else len(got)
            rows += got or []
    return _finish(rows, max_age_days), status


def _finish(rows: list[dict], max_age_days: float) -> pd.DataFrame:
    cols = ["Time", "Headline", "Source", "Link", "Feed", "Theme", "Tone", "Topics"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["_k"] = df["Headline"].map(_norm)
    df = df.drop_duplicates("_k").drop(columns="_k")
    if df["Time"].notna().any():
        now = pd.Timestamp.now(tz=df["Time"].dropna().iloc[0].tz)
        age = df["Time"].map(lambda t: (now - t).total_seconds() / 86400 if t is not None else np.nan)
        df = df[(age <= max_age_days) | age.isna()]
    df["Tone"] = df["Headline"].map(score_headline)
    df["Topics"] = df["Headline"].map(classify)
    df["Topics"] = [sorted(set(t) | ({th} if th in TOPICS else set())) for t, th in zip(df["Topics"], df["Theme"])]
    return df.sort_values("Time", ascending=False, na_position="last").reset_index(drop=True)[cols]


def weights(df: pd.DataFrame, half_life_h: float = 24.0) -> pd.Series:
    """Recency weights: a headline loses half its weight every ``half_life_h`` hours."""
    if df.empty or not df["Time"].notna().any():
        return pd.Series(1.0, index=df.index)
    now = pd.Timestamp.now(tz=df["Time"].dropna().iloc[0].tz)
    age_h = df["Time"].map(lambda t: (now - t).total_seconds() / 3600 if t is not None else 48.0).clip(lower=0)
    return 0.5 ** (age_h / half_life_h)


def topic_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per theme: headline count, recency-weighted tone, share positive / negative."""
    rows = []
    w = weights(df)
    for th in TOPICS:
        m = df["Topics"].map(lambda x, th=th: th in x)
        if not m.any():
            continue
        d, ww = df[m], w[m]
        tone = float((d["Tone"] * ww).sum() / ww.sum()) if ww.sum() > 0 else float(d["Tone"].mean())
        rows.append({"Theme": th, "Headlines": int(m.sum()), "Tone": tone, "Positive": int((d["Tone"] > 0.1).sum()),
                     "Negative": int((d["Tone"] < -0.1).sum())})
    return pd.DataFrame(rows).sort_values("Headlines", ascending=False) if rows else pd.DataFrame(
        columns=["Theme", "Headlines", "Tone", "Positive", "Negative"])


def overall_tone(df: pd.DataFrame, themes: tuple[str, ...] | None = None) -> dict:
    d = df if themes is None else df[df["Topics"].map(lambda x: any(t in x for t in themes))]
    if d.empty:
        return {"tone": np.nan, "n": 0}
    w = weights(d)
    return {"tone": float((d["Tone"] * w).sum() / w.sum()), "n": int(len(d)),
            "share_neg": float((d["Tone"] < -0.1).mean()), "share_pos": float((d["Tone"] > 0.1).mean())}


def event_radar(df: pd.DataFrame) -> pd.DataFrame:
    """Events the news is currently talking about, how soon they sound, and the tone around them."""
    rows = []
    w = weights(df)
    for ev, rx in EVENTS.items():
        m = df["Headline"].str.lower().str.contains(rx, regex=True)
        if m.sum() < 2:
            continue
        d, ww = df[m], w[m]
        soon = d["Headline"].map(lambda t: bool(SOON.search(t)))
        tone = float((d["Tone"] * ww).sum() / ww.sum())
        heat = float(ww.sum())  # recency-weighted mention count
        rows.append({"Event": ev, "Mentions": int(m.sum()), "Heat": heat, "Imminent": int(soon.sum()), "Tone": tone,
                     "Latest": d.iloc[0]["Headline"]})
    out = pd.DataFrame(rows)
    return out.sort_values(["Imminent", "Heat"], ascending=False).reset_index(drop=True) if len(out) else out


# ------------------------------------------------------------------ GDELT


def gdelt_tone(query: str = "(sensex OR nifty OR \"indian stock market\" OR \"indian economy\")", days: int = 90
               ) -> pd.Series | None:
    """Daily average tone of global news about ``query`` (GDELT, free). None if unreachable."""
    if demo_forced():
        return None
    url = (f"{GDELT}?query={urllib.parse.quote(query)}&mode=timelinetone&format=json&timespan={int(days)}d")
    raw = _get(url, timeout=12)
    return parse_gdelt_timeline(raw) if raw else None


def parse_gdelt_timeline(raw: bytes) -> pd.Series | None:
    try:
        tl = json.loads(raw)["timeline"][0]["data"]
        s = pd.Series({pd.to_datetime(p["date"], utc=True).tz_localize(None).normalize(): float(p["value"]) for p in tl})
    except Exception:  # noqa: BLE001
        return None
    s = s.groupby(level=0).mean().sort_index()
    return s if len(s) >= 5 else None
