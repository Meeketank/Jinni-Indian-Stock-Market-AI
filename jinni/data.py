"""Market data access.

Live data comes from Yahoo Finance (NSE ``.NS`` / BSE ``.BO`` tickers). When
Yahoo is unreachable or ``JINNI_DEMO=1`` is set, a deterministic synthetic
market is generated instead so the app, tests and screenshots still work.
Every result carries a ``source`` flag so the UI can label demo data clearly.
"""

from __future__ import annotations

import hashlib
import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .universe import BENCHMARK, NIFTY50, TRADING_DAYS, sector_of

log = logging.getLogger(__name__)

PERIOD_DAYS = {"6mo": 126, "1y": 252, "2y": 504, "3y": 756, "5y": 1260, "10y": 2520}


# Per-request override (the web API sets this from the X-Jinni-Demo header); None = use the env var.
DEMO_OVERRIDE: ContextVar[bool | None] = ContextVar("jinni_demo_override", default=None)


def demo_forced() -> bool:
    o = DEMO_OVERRIDE.get()
    if o is not None:
        return o
    return os.getenv("JINNI_DEMO", "").strip() in {"1", "true", "yes"}


@dataclass
class PriceData:
    """OHLCV panel. ``close`` etc. are wide frames: index=date, columns=ticker."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    source: str = "live"  # "live" or "demo"
    missing: list[str] = field(default_factory=list)

    def ohlcv(self, ticker: str) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "Open": self.open[ticker],
                "High": self.high[ticker],
                "Low": self.low[ticker],
                "Close": self.close[ticker],
                "Volume": self.volume[ticker],
            }
        )
        return df.dropna(subset=["Close"])

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)


# --------------------------------------------------------------------- live


_YF_READY = False


def _yf():
    """Import yfinance once and point its caches at a writable directory (Vercel: /tmp only)."""
    global _YF_READY
    import yfinance as yf

    if not _YF_READY:
        try:
            from .paths import cache_dir

            loc = cache_dir() / "yfinance"
            loc.mkdir(parents=True, exist_ok=True)
            yf.set_tz_cache_location(str(loc))
        except Exception:
            pass
        _YF_READY = True
    return yf


def _download_yahoo(tickers: list[str], period: str, interval: str) -> PriceData | None:
    try:
        yf = _yf()
    except ImportError:
        return None
    try:
        raw = yf.download(
            tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception as exc:  # network / rate limit
        log.warning("Yahoo download failed: %s", exc)
        return None
    if raw is None or raw.empty:
        return None

    def field_frame(name: str) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            if name not in raw.columns.get_level_values(0):
                return pd.DataFrame(index=raw.index)
            f = raw[name]
        else:  # single ticker
            f = raw[[name]].rename(columns={name: tickers[0]})
        return f.reindex(columns=tickers)

    frames = {k: field_frame(k) for k in ["Open", "High", "Low", "Close", "Volume"]}
    idx = frames["Close"].index
    if getattr(idx, "tz", None) is not None:
        for k in frames:
            frames[k].index = frames[k].index.tz_localize(None)
    close = frames["Close"]
    missing = [t for t in tickers if t not in close or close[t].notna().sum() < 5]
    ok = [t for t in tickers if t not in missing]
    if not ok:
        return None
    # Drop rows where every ticker is NaN (holidays across exchanges).
    keep = close[ok].notna().any(axis=1)
    return PriceData(
        open=frames["Open"].loc[keep, ok],
        high=frames["High"].loc[keep, ok],
        low=frames["Low"].loc[keep, ok],
        close=close.loc[keep, ok],
        volume=frames["Volume"].loc[keep, ok].fillna(0),
        source="live",
        missing=missing,
    )


def fetch_info(ticker: str) -> dict:
    """Fundamentals snapshot from Yahoo. Returns {} on failure or in demo mode."""
    if demo_forced():
        return _demo_info(ticker)
    try:
        return _yf().Ticker(ticker).info or {}
    except Exception:
        return {}  # never substitute made-up fundamentals for real ones


_ALIVE: bool | None = None


def _yahoo_alive() -> bool:
    global _ALIVE
    if _ALIVE is None:
        _ALIVE = _download_yahoo([BENCHMARK], "5d", "1d") is not None
    return _ALIVE


# --------------------------------------------------------------------- demo


def _seed(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


_SECTOR_BETA = {
    "Financials": 1.15, "IT": 0.85, "Energy": 1.0, "Auto": 1.05, "FMCG": 0.6,
    "Healthcare": 0.7, "Metals & Mining": 1.35, "Power": 0.9, "Telecom": 0.75,
}


# Approximate price levels so demo portfolios look realistic (not market data).
_DEMO_LAST = {
    "^NSEI": 25500, "^BSESN": 83500, "^NSEBANK": 57000, "RELIANCE.NS": 1450, "HDFCBANK.NS": 980,
    "ICICIBANK.NS": 1380, "INFY.NS": 1500, "TCS.NS": 3050, "BHARTIARTL.NS": 2000, "LT.NS": 3900,
    "ITC.NS": 410, "SUNPHARMA.NS": 1700, "MARUTI.NS": 16000,
}


def _demo_prices(tickers: list[str], n_days: int) -> PriceData:
    """Synthetic but realistic market: one market factor + sector factors + noise,
    fat tails (Student-t), volatility clustering and a 2020-style crash."""
    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=n_days)
    rng = np.random.default_rng(42)
    # market factor with GARCH-ish clustering
    n = len(dates)
    shocks = rng.standard_t(5, n) / np.sqrt(5 / 3)
    vol = np.empty(n)
    v = 0.011
    for i in range(n):
        v = np.sqrt(0.000002 + 0.08 * (v * shocks[i - 1]) ** 2 + 0.9 * v**2) if i else v
        vol[i] = min(v, 0.05)
    mkt = 0.0005 + vol * shocks
    crash = (dates >= "2020-02-20") & (dates <= "2020-03-23")
    mkt[crash] -= 0.012
    sectors = sorted({sector_of(t) for t in tickers})
    sec_f = {s: rng.normal(0, 0.006, n) for s in sectors}

    o, h, lo, c, vlm = {}, {}, {}, {}, {}
    for t in tickers:
        r = np.random.default_rng(_seed(t))
        if t.startswith("^") or t not in NIFTY50:
            beta = 1.0 if t == BENCHMARK else r.uniform(0.6, 1.3)
            idio_sd = 0.0 if t == BENCHMARK else 0.005
            sec = np.zeros(n)
        else:
            s = sector_of(t)
            beta = _SECTOR_BETA.get(s, 1.0) * r.uniform(0.85, 1.15)
            idio_sd = r.uniform(0.008, 0.016)
            sec = sec_f[s]
        alpha = r.normal(0.0002, 0.0003)
        ret = alpha + beta * (mkt - 0.0005) + 0.0004 + sec + r.normal(0, idio_sd, n)
        if t == "^INDIAVIX":
            # VIX tracks the market's conditional volatility (annualised, in %)
            level = vol * np.sqrt(TRADING_DAYS) * 100
            close = pd.Series(np.clip(level + r.normal(0, 0.4, n), 9, 80), index=dates)
        else:
            start = r.uniform(150, 4000) if t not in {BENCHMARK, "^BSESN"} else (8000 if t == BENCHMARK else 27000)
            close = pd.Series(start * np.exp(np.cumsum(ret)), index=dates)
            if t in _DEMO_LAST:
                close *= _DEMO_LAST[t] / close.iloc[-1]
        rng_day = np.abs(r.normal(0, 1, n)) * (vol + idio_sd) * close.values
        op = close.shift(1).fillna(close.iloc[0]).values * (1 + r.normal(0, 0.003, n))
        o[t] = op
        h[t] = np.maximum(op, close.values) + rng_day * 0.5
        lo[t] = np.minimum(op, close.values) - rng_day * 0.5
        c[t] = close.values
        base_v = r.uniform(5e5, 2e7)
        vlm[t] = base_v * np.exp(r.normal(0, 0.35, n)) * (1 + 15 * np.abs(ret))

    mk = lambda d: pd.DataFrame(d, index=dates)  # noqa: E731
    return PriceData(mk(o), mk(h), mk(lo), mk(c), mk(vlm), source="demo")


def _demo_info(ticker: str) -> dict:
    r = np.random.default_rng(_seed(ticker + "info"))
    name = ticker.replace(".NS", "").replace(".BO", "")
    return {
        "longName": f"{name} Ltd (demo)",
        "sector": sector_of(ticker),
        "marketCap": float(r.uniform(5e11, 2e13)),
        "trailingPE": float(r.uniform(10, 70)),
        "priceToBook": float(r.uniform(1, 15)),
        "returnOnEquity": float(r.uniform(0.05, 0.35)),
        "debtToEquity": float(r.uniform(0, 150)),
        "dividendYield": float(r.uniform(0, 3.5)),
        "profitMargins": float(r.uniform(0.03, 0.3)),
        "revenueGrowth": float(r.uniform(-0.05, 0.25)),
        "earningsGrowth": float(r.uniform(-0.1, 0.35)),
    }


# --------------------------------------------------------------------- api


def get_prices(tickers: list[str] | tuple[str, ...], period: str = "2y", interval: str = "1d",
               allow_demo: bool = True) -> PriceData:
    """Fetch OHLCV for many tickers in one request, falling back to demo data
    (unless ``allow_demo`` is False, in which case a failure raises)."""
    tickers = list(dict.fromkeys(tickers))  # dedupe, keep order
    if not tickers:
        raise ValueError("No tickers given")
    if not demo_forced():
        live = _download_yahoo(tickers, period, interval)
        if live is not None:
            return live
        if not allow_demo:
            raise RuntimeError("Live data unavailable")
    n_days = PERIOD_DAYS.get(period, 2 * TRADING_DAYS)
    # Always generate enough history for the stress scenarios in demo mode.
    demo = _demo_prices(sorted(set(tickers) | {BENCHMARK}), max(n_days, 1800))
    keep = demo.close.index[-n_days:]
    cols = [t for t in tickers]
    return PriceData(
        demo.open.loc[keep, cols], demo.high.loc[keep, cols], demo.low.loc[keep, cols],
        demo.close.loc[keep, cols], demo.volume.loc[keep, cols], source="demo",
    )
