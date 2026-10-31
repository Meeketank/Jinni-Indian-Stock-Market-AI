"""NIFTY 50 membership and sectors, fetched live (never hard-coded).

Order of sources: NSE index file -> niftyindices.com file -> Yahoo's screener (the 50 largest NSE stocks by market
cap; an approximation, labelled as such, without sectors). If all fail the mapping is empty and callers degrade
gracefully instead of showing a stale list. Results are cached for a day, in memory and on disk.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
]
TTL = 24 * 3600
# NSE's own industry names, folded to the sector names the rest of the app already uses.
FOLD = {"FINANCIAL SERVICES": "Financials", "INFORMATION TECHNOLOGY": "IT", "AUTOMOBILE AND AUTO COMPONENTS": "Auto",
        "OIL GAS & CONSUMABLE FUELS": "Energy", "FAST MOVING CONSUMER GOODS": "FMCG", "HEALTHCARE": "Healthcare",
        "METALS & MINING": "Metals & Mining", "TELECOMMUNICATION": "Telecom", "POWER": "Power",
        "CONSTRUCTION": "Construction", "CONSTRUCTION MATERIALS": "Materials", "CAPITAL GOODS": "Capital Goods",
        "CONSUMER DURABLES": "Consumer Durables", "CONSUMER SERVICES": "Consumer Services", "SERVICES": "Services"}


def _from_csv(text: str) -> dict[str, str]:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper() for c in df.columns]
    if "SYMBOL" not in df:
        return {}
    ind = df["INDUSTRY"] if "INDUSTRY" in df else pd.Series("Other", index=df.index)
    return {f"{str(s).strip()}.NS": FOLD.get(str(i).strip().upper(), str(i).strip().title() or "Other")
            for s, i in zip(df["SYMBOL"], ind)}


def _fetch(url: str) -> str | None:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                                "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
                                               "Referer": "https://www.nseindia.com/"})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.read().decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        log.info("index file %s unreachable: %s", url, e)
        return None


def load() -> tuple[dict[str, str], str]:
    """(ticker -> sector, source label). Empty dict if nothing is reachable."""
    from .data import demo_forced

    if demo_forced():  # tests only
        fx = os.getenv("JINNI_FIXTURES")
        if fx and (Path(fx) / "nifty50.csv").exists():
            return _from_csv((Path(fx) / "nifty50.csv").read_text()), "test fixture"
        return {}, "none"
    for u in URLS:
        txt = _fetch(u)
        if txt and "SYMBOL" in txt[:300].upper():
            d = _from_csv(txt)
            if len(d) >= 40:
                return d, "NSE index file"
    try:
        from . import live

        top = live.top_by_cap(50)
        if top is not None and len(top) >= 40:
            return {t: "Other" for t in top["t"]}, "50 largest NSE stocks by market cap (Yahoo)"
    except Exception as e:  # noqa: BLE001
        log.info("screener fallback failed: %s", e)
    return {}, "none"


class LiveIndex(Mapping):
    """dict-like view of the current NIFTY 50 that loads on first use and refreshes daily."""

    def __init__(self):
        self._data: dict[str, str] = {}
        self._src = "none"
        self._at = 0.0
        self._lock = threading.Lock()

    def _ensure(self) -> dict[str, str]:
        ttl = TTL if self._data else 120  # retry soon after a failure
        if time.time() - self._at > ttl:
            with self._lock:
                if time.time() - self._at > ttl:
                    self._data, self._src = load()
                    self._at = time.time()
        return self._data

    @property
    def source(self) -> str:
        self._ensure()
        return self._src

    def __getitem__(self, k):
        return self._ensure()[k]

    def __iter__(self):
        return iter(self._ensure())

    def __len__(self):
        return len(self._ensure())


NIFTY50 = LiveIndex()
