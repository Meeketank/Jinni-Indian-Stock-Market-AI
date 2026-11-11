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


def top_by_cap(n: int = 50) -> pd.DataFrame | None:
    """The n largest NSE stocks by market cap."""
    if demo_forced():
        return None
    try:
        resp = _yf().screen(_query(), size=n, sortField="intradaymarketcap", sortAsc=False)
        df = _rows(resp.get("quotes", []) if isinstance(resp, dict) else [])
    except Exception as e:  # noqa: BLE001
        log.warning("NSE screener (top by cap) failed: %s", e)
        return None
    return df if len(df) else None


def universe(max_pages: int = 14) -> pd.DataFrame | None:
    """Every NSE-listed stock Yahoo knows, paged 250 at a time (largest first). None if unreachable."""
    if demo_forced():
        return None
    from yfinance import EquityQuery

    q = EquityQuery("eq", ["exchange", "NSI"])
    rows: list[dict] = []
    try:
        for page in range(max_pages):
            resp = _yf().screen(q, size=250, offset=page * 250, sortField="intradaymarketcap", sortAsc=False)
            quotes = resp.get("quotes", []) if isinstance(resp, dict) else []
            rows += [{"symbol": x["symbol"].rsplit(".", 1)[0], "name": x.get("longName") or x.get("shortName") or x["symbol"]}
                     for x in quotes if str(x.get("symbol", "")).endswith(".NS")]
            if len(quotes) < 250:
                break
    except Exception as e:  # noqa: BLE001
        log.warning("NSE screener (universe) failed after %d rows: %s", len(rows), e)
    return pd.DataFrame(rows) if rows else None


def _nice(n: str) -> str:
    n = str(n).strip()
    return n.title() if n.isupper() else n


def exchange_quotes(exchange: str = "BSE", max_pages: int = 20, workers: int = 6) -> pd.DataFrame | None:
    """Latest quote for every stock Yahoo lists on an exchange ("NSI" or "BSE"), fetched 250 at a time in parallel.
    Used when the exchange's own end-of-day file cannot be downloaded. None if Yahoo's screener is unreachable."""
    if demo_forced():
        return None
    from concurrent.futures import ThreadPoolExecutor

    from yfinance import EquityQuery

    q = EquityQuery("eq", ["exchange", exchange])
    suffix = ".NS" if exchange == "NSI" else ".BO"

    def page(i: int) -> list[dict]:
        try:
            resp = _yf().screen(q, size=250, offset=i * 250, sortField="intradaymarketcap", sortAsc=False)
            return resp.get("quotes", []) if isinstance(resp, dict) else []
        except Exception as e:  # noqa: BLE001
            log.warning("screener %s page %d failed: %s", exchange, i, e)
            return []

    first = page(0)
    if not first:
        return None
    quotes = list(first)
    if len(first) == 250:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for got in ex.map(page, range(1, max_pages)):
                quotes += got
    rows = []
    for x in quotes:
        sym, px, prev = str(x.get("symbol", "")), x.get("regularMarketPrice"), x.get("regularMarketPreviousClose")
        if not sym.endswith(suffix) or not px or not prev:
            continue
        vol = x.get("regularMarketVolume") or 0
        rows.append({"ticker": sym, "symbol": sym[: -len(suffix)], "name": _nice(x.get("longName") or x.get("shortName") or sym),
                     "exch": "NSE" if exchange == "NSI" else "BSE", "isin": "", "series": "", "open": x.get("regularMarketOpen"),
                     "high": x.get("regularMarketDayHigh"), "low": x.get("regularMarketDayLow"), "close": float(px),
                     "prev": float(prev), "volume": float(vol), "turnover": float(px) * float(vol), "date": pd.Timestamp.today().normalize()})
    return pd.DataFrame(rows).drop_duplicates("ticker") if rows else None
