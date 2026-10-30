"""Live NSE-wide quotes from Yahoo's screener (one request for the whole exchange, not one per stock).

Used for the home page movers so they are real, current and market-wide instead of a fixed list.
Failure returns ``None`` (never made-up rows) so the caller can say the feed is unavailable.
"""

from __future__ import annotations

import logging

import pandas as pd

from .data import _yf, demo_forced

log = logging.getLogger(__name__)

# (sortField, ascending) per board
BOARDS = {
    "gainers": ("percentchange", False),
    "losers": ("percentchange", True),
    "active": ("dayvolume", False),
}
MIN_CAP = 5e9        # ₹500 cr market cap: keeps penny / illiquid names out of "movers"
MIN_PRICE = 20.0


def _query():
    from yfinance import EquityQuery

    return EquityQuery("and", [
        EquityQuery("eq", ["exchange", "NSI"]),
        EquityQuery("gt", ["intradaymarketcap", MIN_CAP]),
        EquityQuery("gt", ["intradayprice", MIN_PRICE]),
    ])


def _rows(quotes: list[dict]) -> pd.DataFrame:
    out = []
    for q in quotes or []:
        sym = q.get("symbol")
        px = q.get("regularMarketPrice")
        if not sym or px is None:
            continue
        out.append({
            "t": sym, "Stock": q.get("shortName") or q.get("longName") or sym.replace(".NS", ""),
            "Price": float(px), "Change": q.get("regularMarketChange"),
            "1D": (q.get("regularMarketChangePercent") or 0) / 100, "Volume": q.get("regularMarketVolume"),
            "Mkt cap": q.get("marketCap"), "52w high": q.get("fiftyTwoWeekHigh"), "52w low": q.get("fiftyTwoWeekLow"),
            "P/E": q.get("trailingPE"),
        })
    return pd.DataFrame(out)


def board(kind: str, size: int = 15) -> pd.DataFrame | None:
    """Top ``size`` NSE stocks for 'gainers' | 'losers' | 'active', or None if the feed is unreachable."""
    if kind not in BOARDS or demo_forced():
        return None
    field, asc = BOARDS[kind]
    try:
        resp = _yf().screen(_query(), size=size, sortField=field, sortAsc=asc)
        df = _rows(resp.get("quotes", []) if isinstance(resp, dict) else [])
    except Exception as e:  # noqa: BLE001 - network, rate limit, schema drift
        log.warning("NSE screener '%s' failed: %s", kind, e)
        return None
    return df if len(df) else None
