"""Whole-market universe: every NSE-listed equity, bulk prices, and a fast pre-filter.

The full list comes live from NSE (EQUITY_L.csv). If NSE is unreachable a
bundled snapshot is used, and a user-supplied CSV can override both. Prices
for ~2,000 stocks are downloaded in chunks and cached on disk for the day, so
repeat scans are instant.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import pickle
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import PriceData, demo_forced, get_prices
from .universe import NIFTY50

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
FALLBACK = HERE / "data_files" / "EQUITY_L_fallback.csv"
from .paths import cache_dir  # noqa: E402

CACHE_DIR = cache_dir()
NSE_URLS = [
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
]
SME_URL = "https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv"


def _clean(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df.columns = [c.strip().upper() for c in df.columns]
    out = pd.DataFrame({
        "symbol": df["SYMBOL"].astype(str).str.strip(),
        "name": df.get("NAME OF COMPANY", df["SYMBOL"]).astype(str).str.strip(),
        "series": df.get("SERIES", pd.Series("EQ", index=df.index)).astype(str).str.strip(),
        "isin": df.get("ISIN NUMBER", pd.Series("", index=df.index)).astype(str).str.strip(),
    })
    out["ticker"] = out["symbol"] + ".NS"
    out.attrs["source"] = source
    return out.drop_duplicates("symbol").reset_index(drop=True)


def _fetch(url: str, timeout: int = 20) -> str | None:
    import urllib.request

    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/126.0 Safari/537.36",
        "Accept": "text/csv,*/*", "Referer": "https://www.nseindia.com/"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "ignore")
    except Exception as e:
        log.info("NSE list fetch failed for %s: %s", url, e)
        return None


def nse_universe(include_sme: bool = False, series: tuple = ("EQ", "BE", "BZ"), override_csv: bytes | None = None
                 ) -> pd.DataFrame:
    """All NSE-listed equities. Live list → today's disk cache → bundled snapshot."""
    if override_csv:
        df = _clean(pd.read_csv(io.BytesIO(override_csv)), "uploaded file")
    else:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = CACHE_DIR / f"equity_list_{date.today():%Y%m%d}.csv"
        df = None
        if cache.exists():
            df = _clean(pd.read_csv(cache), "NSE (cached today)")
        elif not demo_forced():
            for u in NSE_URLS:
                txt = _fetch(u, timeout=8)
                if txt and "SYMBOL" in txt[:200].upper():
                    cache.write_text(txt)
                    df = _clean(pd.read_csv(io.StringIO(txt)), "NSE live")
                    break
        if df is None:
            df = _clean(pd.read_csv(FALLBACK), "bundled snapshot (older; may miss recent listings)")
        if include_sme and not demo_forced():
            txt = _fetch(SME_URL)
            if txt:
                sme = _clean(pd.read_csv(io.StringIO(txt)), "NSE SME")
                df = pd.concat([df, sme], ignore_index=True)
    src = df.attrs.get("source", "")
    allowed = set(series) | ({"SM", "ST"} if include_sme else set())
    df = df[df["series"].isin(allowed)].reset_index(drop=True)
    df.attrs["source"] = src
    return df


def bulk_prices(tickers: list[str], period: str = "1y", chunk: int = 150, progress=None) -> PriceData:
    """Download many tickers in chunks, with a same-day disk cache per chunk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tickers = list(dict.fromkeys(tickers))
    parts, missing, source = [], [], "live"
    chunks = [tickers[i:i + chunk] for i in range(0, len(tickers), chunk)]
    for k, ch in enumerate(chunks):
        key = hashlib.md5(("|".join(ch) + period).encode()).hexdigest()[:12]
        f = CACHE_DIR / f"px_{date.today():%Y%m%d}_{period}_{key}.pkl"
        pdx = None
        if f.exists():
            try:
                pdx = pickle.loads(f.read_bytes())
            except Exception:
                pdx = None
        if pdx is None:
            for attempt in range(3):
                try:
                    pdx = get_prices(ch, period, allow_demo=demo_forced())  # never mix synthetic into live scans
                    break
                except Exception:
                    time.sleep(2 * (attempt + 1))
            if pdx is not None and pdx.source == "live":
                try:
                    f.write_bytes(pickle.dumps(pdx))
                except Exception:
                    pass
        if pdx is None:
            missing += ch  # chunk failed after retries (e.g. rate limit): skipped, reported as missing
        else:
            parts.append(pdx)
            missing += pdx.missing
            source = pdx.source if pdx.source == "demo" else source
        if progress:
            progress((k + 1) / len(chunks), f"Downloaded {min((k + 1) * chunk, len(tickers)):,} / {len(tickers):,}")
    if not parts:
        raise RuntimeError("No price data could be downloaded.")
    cat = lambda attr: pd.concat([getattr(p, attr) for p in parts], axis=1)  # noqa: E731
    out = PriceData(cat("open"), cat("high"), cat("low"), cat("close"), cat("volume"), source, missing)
    out.close = out.close.loc[:, ~out.close.columns.duplicated()]
    for a in ("open", "high", "low", "volume"):
        setattr(out, a, getattr(out, a).loc[:, ~getattr(out, a).columns.duplicated()])
    return out


def bundled_universe() -> pd.DataFrame:
    """Offline snapshot of NSE equities (used for symbol search and as a fallback list)."""
    return _clean(pd.read_csv(FALLBACK), "bundled snapshot")


def demo_universe(n: int = 300) -> pd.DataFrame:
    """In demo mode, a manageable slice of the real symbol list."""
    base = _clean(pd.read_csv(FALLBACK), "bundled snapshot (demo)")
    base = base[base["series"] == "EQ"]
    nifty = pd.DataFrame({"symbol": [t[:-3] for t in NIFTY50], "name": [t[:-3] for t in NIFTY50],
                          "series": "EQ", "isin": "", "ticker": list(NIFTY50)})
    out = pd.concat([nifty, base.sample(min(n, len(base)), random_state=1)]).drop_duplicates("ticker")
    out.attrs["source"] = "bundled snapshot (demo)"
    return out.reset_index(drop=True)


def prefilter(pdata: PriceData, budget: float | None, target: float, horizon: int, min_price: float = 10.0,
              min_value_cr: float = 0.5, max_flat_days: int = 3) -> pd.DataFrame:
    """Stage 1 on every stock: affordability, liquidity, data quality and a
    closed-form estimate of the odds of touching the target in time.

    P(touch) ≈ 2·(1 − Φ(ln(1+X) / (σ√T))) for a driftless random walk with
    daily volatility σ; used only to rank candidates for the full engines."""
    c, v, h, lo = pdata.close, pdata.volume, pdata.high, pdata.low
    last = c.ffill().iloc[-1]
    stale = c.iloc[-5:].isna().all()
    lr = np.log(c).diff()
    sig = lr.iloc[-120:].std()
    value_cr = (c * v).iloc[-20:].median() / 1e7
    flat = ((h - lo).iloc[-20:] <= 1e-9 * c.iloc[-20:]).sum()  # circuit-locked / no-trade days
    hist_days = c.notna().sum()
    p_touch = 2 * (1 - norm.cdf(np.log1p(target) / (sig * np.sqrt(horizon))))
    df = pd.DataFrame({"Price": last, "Daily vol": sig, "Traded value ₹Cr": value_cr, "Locked days (20d)": flat,
                       "History days": hist_days, "Quick P(touch)": p_touch})
    reasons = pd.Series("", index=df.index)
    reasons[stale] += "no recent trades; "
    reasons[df["Price"] < min_price] += f"price < ₹{min_price:.0f}; "
    if budget:
        reasons[df["Price"] > budget] += "costs more than budget; "
    reasons[df["Traded value ₹Cr"] < min_value_cr] += "illiquid; "
    reasons[df["Locked days (20d)"] > max_flat_days] += "circuit-locked; "
    reasons[df["History days"] < 150] += "too new; "
    df["Excluded because"] = reasons.str.rstrip("; ")
    return df.sort_values("Quick P(touch)", ascending=False)
