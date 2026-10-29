"""News headlines (Yahoo Finance RSS) and corporate-event dates (earnings, ex-dividend).

Headlines are tagged so the important ones stand out. In demo mode, or when
the feeds are unreachable, everything returns empty, never an error.
"""

from __future__ import annotations

import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

import pandas as pd

from .data import demo_forced

TAGS = {
    "Results": r"\b(q[1-4]|results?|earnings|profit|revenue|ebitda|quarter)\b",
    "Dividend/Bonus": r"\b(dividend|bonus|split|buyback|record date)\b",
    "Deal": r"\b(acqui|merger|stake|deal|order win|order|contract|tie-?up|jv|ipo|qip|block deal)",
    "Rating": r"\b(upgrade|downgrade|target price|rating|brokerage|buy call|sell call|overweight|underweight)\b",
    "Regulatory": r"\b(sebi|rbi|cci|penalty|probe|ban|court|tax notice|gst)\b",
    "Management": r"\b(ceo|cfo|md|chairman|resign|appoint)\b",
    "Macro": r"\b(fii|fpi|inflation|repo|budget|gdp|crude|rupee|fed)\b",
}


def tag(title: str) -> list[str]:
    t = title.lower()
    return [k for k, rx in TAGS.items() if re.search(rx, t)]


def headlines(ticker: str, limit: int = 15, timeout: int = 8) -> pd.DataFrame:
    if demo_forced():
        return pd.DataFrame(columns=["Time", "Headline", "Tags", "Link"])
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=IN&lang=en-IN"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 JINNI"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            root = ET.fromstring(r.read())
    except Exception:
        return pd.DataFrame(columns=["Time", "Headline", "Tags", "Link"])
    rows = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        try:
            ts = parsedate_to_datetime(it.findtext("pubDate")).astimezone()
        except Exception:
            ts = None
        rows.append({"Time": ts, "Headline": title, "Tags": ", ".join(tag(title)), "Link": it.findtext("link")})
    return pd.DataFrame(rows[:limit])


def upcoming_events(ticker: str) -> dict:
    """{'earnings': date|None, 'ex_dividend': date|None} from Yahoo's calendar."""
    out = {"earnings": None, "ex_dividend": None}
    if demo_forced():
        return out
    try:
        import yfinance as yf

        cal = yf.Ticker(ticker).calendar or {}
        e = cal.get("Earnings Date")
        if isinstance(e, (list, tuple)) and e:
            e = e[0]
        out["earnings"] = pd.Timestamp(e).date() if e is not None else None
        d = cal.get("Ex-Dividend Date")
        out["ex_dividend"] = pd.Timestamp(d).date() if d is not None else None
    except Exception:
        pass
    return out


def event_warnings(ticker: str, horizon_days: int, events: dict | None = None) -> list[str]:
    ev = events if events is not None else upcoming_events(ticker)
    today = datetime.now().date()
    end = today + timedelta(days=int(horizon_days * 7 / 5) + 1)
    warns = []
    if ev.get("earnings") and today <= ev["earnings"] <= end:
        warns.append(f"Results due {ev['earnings']:%d %b}: inside the trade window, so a gap past the stop or target is possible.")
    if ev.get("ex_dividend") and today <= ev["ex_dividend"] <= end:
        warns.append(f"Ex-dividend {ev['ex_dividend']:%d %b}: the price drops by the dividend that day.")
    return warns
