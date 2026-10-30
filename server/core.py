"""Shared plumbing for the web API: JSON cleaning, caching, data access and UI block builders.

Endpoints return *blocks*: small JSON descriptions of KPIs, tables, charts and
text that the static front end (public/js) renders generically. Keeping the
presentation logic here means the front end stays thin and every number is
formatted by the same Python code that computed it.
"""

from __future__ import annotations

import json
import math
import threading
import time
from datetime import date, datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
from fastapi import HTTPException
from fastapi.responses import Response

from jinni import data as jdata
from jinni.fmt import inr, num, pct, tone  # noqa: F401  (re-exported for routes)

# ------------------------------------------------------------------ errors


class ApiError(HTTPException):
    def __init__(self, status: int, message: str, **extra):
        super().__init__(status_code=status, detail={"error": message, **extra})


class DataUnavailable(ApiError):
    def __init__(self, message: str = "Live market data is temporarily unavailable."):
        super().__init__(503, message, code="data_unavailable")


# -------------------------------------------------------------- JSON output


def clean(o: Any, nd: int = 6) -> Any:
    """Make anything JSON-safe: NaN/inf → null, numpy/pandas scalars → Python, dates → ISO."""
    if o is None or isinstance(o, (bool, str)):
        return o
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if not math.isfinite(f) else round(f, nd)
    if isinstance(o, (pd.Timestamp, datetime)):
        return None if pd.isna(o) else o.strftime("%Y-%m-%d") if (o.hour, o.minute, o.second) == (0, 0, 0) \
            else o.strftime("%Y-%m-%d %H:%M")
    if isinstance(o, date):
        return o.isoformat()
    if o is pd.NaT:
        return None
    if isinstance(o, dict):
        return {str(k): clean(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v, nd) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v, nd) for v in o.tolist()]
    if isinstance(o, pd.Series):
        return [clean(v, nd) for v in o.tolist()]
    if isinstance(o, pd.DataFrame):
        return clean(o.to_dict("records"), nd)
    if isinstance(o, pd.Interval):
        return str(o)
    return str(o)


def reply(payload: dict, ttl: int = 0, swr: int = 0, status: int = 200) -> Response:
    """JSON response; ``ttl`` > 0 lets Vercel's CDN cache identical GET requests (s-maxage)."""
    body = json.dumps(clean(payload), allow_nan=False, separators=(",", ":"))
    headers = {}
    if ttl and status == 200:
        headers["Cache-Control"] = f"public, s-maxage={ttl}, stale-while-revalidate={swr or ttl * 3}"
    else:
        headers["Cache-Control"] = "no-store"
    return Response(content=body, media_type="application/json", status_code=status, headers=headers)


# ----------------------------------------------------------------- caching


class TTLCache:
    """Tiny thread-safe in-process cache. Serverless instances are reused between
    requests (Fluid compute), so this saves repeat downloads when warm."""

    def __init__(self, max_items: int = 128):
        self.max = max_items
        self._d: dict = {}
        self._lock = threading.Lock()

    def get(self, key, ttl: float):
        with self._lock:
            hit = self._d.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        return None

    def put(self, key, value):
        with self._lock:
            if len(self._d) >= self.max:
                oldest = min(self._d, key=lambda k: self._d[k][0])
                self._d.pop(oldest, None)
            self._d[key] = (time.time(), value)
        return value


_PRICE_CACHE = TTLCache(96)
_MEMO = TTLCache(256)


def memo(key: tuple, ttl: float, fn: Callable):
    """Cache fn() under (demo flag, key) for ttl seconds."""
    k = (jdata.demo_forced(),) + key
    v = _MEMO.get(k, ttl)
    if v is None:
        v = _MEMO.put(k, fn())
    return v


def prices(tickers, period: str = "2y", ttl: float = 600) -> jdata.PriceData:
    """Cached OHLCV download. Never silently substitutes synthetic data for live data:
    a failed live request raises DataUnavailable unless demo mode is on."""
    tk = tuple(dict.fromkeys(tickers))
    key = (jdata.demo_forced(), tk, period)
    hit = _PRICE_CACHE.get(key, ttl)
    if hit is not None:
        return hit
    try:
        pdx = jdata.get_prices(list(tk), period, allow_demo=jdata.demo_forced())
    except RuntimeError as e:
        raise DataUnavailable() from e
    return _PRICE_CACHE.put(key, pdx)


def need_series(pdx: jdata.PriceData, ticker: str, min_len: int = 60) -> pd.DataFrame:
    if ticker not in pdx.close or pdx.close[ticker].dropna().shape[0] < min_len:
        raise ApiError(404, f"No price history for {ticker}. Search by company name in the box above, or use .NS (NSE) / .BO (BSE) symbols.")
    return pdx.ohlcv(ticker)


# -------------------------------------------------------------- time spans

UNITS = {"day": "days", "week": "weeks", "month": "months", "quarter": "quarters", "year": "years", "fy": "financial years"}
_MONTHS = {"day": 0, "week": 0, "month": 1, "quarter": 3, "year": 12}


def span_start(last: pd.Timestamp, n: int, unit: str) -> pd.Timestamp:
    """First calendar date of a window of ``n`` ``unit`` ending at ``last``. 'fy' = Indian financial years
    (1 April to 31 March), counting the current one: n=1 is the financial year to date."""
    if unit not in UNITS:
        raise ApiError(422, f"Unknown unit '{unit}'. Use one of: {', '.join(UNITS)}.")
    n = int(max(1, min(n, 600)))
    if unit == "day":
        return last - pd.Timedelta(days=n)
    if unit == "week":
        return last - pd.Timedelta(weeks=n)
    if unit == "fy":
        fy_start = pd.Timestamp(year=last.year if last.month >= 4 else last.year - 1, month=4, day=1)
        return fy_start - pd.DateOffset(years=n - 1)
    return last - pd.DateOffset(months=_MONTHS[unit] * n)


def span_bars(df: pd.DataFrame, n: int, unit: str, minimum: int = 20) -> int:
    """Number of trading bars of ``df`` inside the span (at least ``minimum``, at most all of it)."""
    start = span_start(df.index[-1], n, unit)
    return int(min(len(df), max(minimum, (df.index >= start).sum())))


def span_label(n: int, unit: str) -> str:
    return f"{n} {UNITS[unit][:-1] if n == 1 else UNITS[unit]}"


# ------------------------------------------------------------------ blocks


def kpis(items: list[tuple]) -> dict:
    """(label, value, sub=None, tone=None)"""
    out = []
    for it in items:
        d = {"label": it[0], "value": str(it[1])}
        if len(it) > 2 and it[2]:
            d["sub"] = str(it[2])
        if len(it) > 3 and it[3]:
            d["tone"] = it[3]
        out.append(d)
    return {"type": "kpis", "items": out}


def section(title: str) -> dict:
    return {"type": "section", "title": title}


def text(md: str, muted: bool = False) -> dict:
    return {"type": "text", "text": md, "muted": muted}


def caption(md: str) -> dict:
    return text(md, muted=True)


def callout(kind: str, md: str) -> dict:
    """kind: info | warn | error | ok"""
    return {"type": "callout", "kind": kind, "text": md}


def plot(fig: dict, height: int = 360) -> dict:
    return {"type": "plot", "fig": fig, "height": height}


def tabs(items: list[tuple[str, list]]) -> dict:
    return {"type": "tabs", "tabs": [{"label": l, "blocks": b} for l, b in items]}


def cols(*groups: list, ratio: list[int] | None = None) -> dict:
    return {"type": "cols", "cols": [list(g) for g in groups], "ratio": ratio}


def tape(items: list[tuple]) -> dict:
    return {"type": "tape", "items": [[n, p, c] for n, p, c in items]}


def news_list(rows: list[dict]) -> dict:
    return {"type": "news", "items": rows}


def badge_html(label: str, cls: str) -> dict:
    return {"type": "badge", "label": label, "cls": cls}


def source_notice(source: str, missing: list | None = None) -> list[dict]:
    out = []
    if source == "demo":
        out.append({"type": "demo"})
    if missing:
        out.append(caption("Skipped (no data): " + ", ".join(m.replace(".NS", "") for m in missing[:20])
                           + (f" and {len(missing) - 20} more" if len(missing) > 20 else "")))
    return out


def col(key: str, label: str, fmt: str = "text", help: str | None = None, **kw) -> dict:
    d = {"key": key, "label": label, "fmt": fmt}
    if help:
        d["help"] = help
    d.update(kw)
    return d


def table(df: pd.DataFrame, columns: list[dict] | None = None, index: bool = False, index_label: str = "",
          select: dict | None = None, height: int | None = None, sort: tuple[str, bool] | None = None,
          empty: str = "Nothing to show.") -> dict:
    """Table block. ``columns`` fixes order/format; formats the front end understands:
    text int n0 n1 n2 n3 pct0 pct1 pct2 spct1 spct2 inr0 inr2 date bar01 bar100 badge link"""
    d = df.reset_index() if index else df.copy()
    if index and index_label:
        d = d.rename(columns={d.columns[0]: index_label})
    if columns is None:
        columns = [col(str(c), str(c), "text") for c in d.columns]
    keys = [c["key"] for c in columns]
    if select and select["key"] not in keys:
        keys = keys + [select["key"]]  # keep the click-through key even though it is not displayed
    d = d[[k for k in keys if k in d.columns]]
    columns = [c for c in columns if c["key"] in d.columns]
    return {"type": "table", "columns": columns, "rows": clean(d.to_dict("records")), "select": select,
            "height": height, "sort": list(sort) if sort else None, "empty": empty}


def rating_class(rating: str) -> str:
    return {"STRONG BUY": "b-sbuy", "BUY": "b-buy", "HOLD": "b-hold", "SELL": "b-sell",
            "STRONG SELL": "b-ssell", "NEUTRAL": "b-hold"}.get(rating, "b-hold")


def doc(blocks: list, source: str | None = None, **data) -> dict:
    out = {"blocks": blocks}
    if source:
        out["source"] = source
    if data:
        out["data"] = data
    return out
