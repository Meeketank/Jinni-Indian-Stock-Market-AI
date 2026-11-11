"""Every listed stock on NSE and BSE from the exchanges' own end-of-day bhavcopy files.

One file per exchange covers the whole market (about 2,000 NSE and 4,700 BSE listings), so "all stocks" movers, the
liquid-stock universe for the Finder and BSE-only names all come from two downloads instead of thousands of requests.
Cross-listed companies are matched by ISIN and shown once, on NSE. BSE-only stocks use Yahoo's ``<scrip code>.BO``.

Values are the latest official end-of-day prints (the file for the most recent trading day), labelled with their date.
If neither exchange can be reached the caller is told so; nothing is invented.
"""

from __future__ import annotations

import io
import logging
import os
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .data import demo_forced
from .paths import cache_dir

log = logging.getLogger(__name__)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")
BSE_URL = "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{d:%Y%m%d}_F_0000.CSV"
NSE_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv"
BSE_EQUITY_SERIES = {"A", "B", "T", "X", "XT", "Z", "M", "MT", "S", "P"}
NSE_EQUITY_SERIES = {"EQ", "BE", "BZ", "SM", "ST", "BL"}
COLS = ["ticker", "symbol", "name", "exch", "isin", "series", "open", "high", "low", "close", "prev", "volume", "turnover", "date"]


def _get(url: str, referer: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": referer, "Accept": "text/csv,*/*",
                                                   "Accept-Language": "en-IN,en;q=0.9", "Origin": referer.rstrip("/")})
        with urllib.request.urlopen(req, timeout=15) as r:
            txt = r.read().decode("utf-8", "ignore")
        return txt if len(txt) > 5000 and "<html" not in txt[:300].lower() else None
    except Exception as e:  # noqa: BLE001
        log.info("bhavcopy %s unreachable: %s", url, e)
        return None


def nice_name(n: str) -> str:
    """Title-case only names that arrive in ALL CAPS, so "ICICI Bank Limited" is not turned into "Icici Bank Limited"."""
    n = str(n).strip()
    return n.title() if n.isupper() else n


def parse_bse(text: str) -> pd.DataFrame:
    d = pd.read_csv(io.StringIO(text), dtype=str)
    d.columns = [c.strip() for c in d.columns]
    d = d[(d["FinInstrmTp"] == "STK") & d["SctySrs"].isin(BSE_EQUITY_SERIES) & d["ISIN"].str.startswith("INE", na=False)]
    num = lambda c: pd.to_numeric(d[c], errors="coerce")  # noqa: E731
    out = pd.DataFrame({
        "ticker": d["FinInstrmId"].str.strip() + ".BO", "symbol": d["TckrSymb"].str.strip(), "name": d["FinInstrmNm"].map(nice_name),
        "exch": "BSE", "isin": d["ISIN"].str.strip(), "series": d["SctySrs"], "open": num("OpnPric"), "high": num("HghPric"),
        "low": num("LwPric"), "close": num("ClsPric"), "prev": num("PrvsClsgPric"), "volume": num("TtlTradgVol"),
        "turnover": num("TtlTrfVal"), "date": pd.to_datetime(d["TradDt"])})
    return out.dropna(subset=["close", "prev"]).query("close > 0 and prev > 0").reset_index(drop=True)[COLS]


def parse_nse(text: str, isin_by_symbol: dict[str, str] | None = None) -> pd.DataFrame:
    d = pd.read_csv(io.StringIO(text), dtype=str, skipinitialspace=True)
    d.columns = [c.strip().upper() for c in d.columns]
    d = d[d["SERIES"].str.strip().isin(NSE_EQUITY_SERIES)]
    num = lambda c: pd.to_numeric(d[c].str.strip(), errors="coerce")  # noqa: E731
    sym = d["SYMBOL"].str.strip()
    out = pd.DataFrame({
        "ticker": sym + ".NS", "symbol": sym, "name": sym, "exch": "NSE", "isin": sym.map(isin_by_symbol or {}).fillna(""),
        "series": d["SERIES"].str.strip(), "open": num("OPEN_PRICE"), "high": num("HIGH_PRICE"), "low": num("LOW_PRICE"),
        "close": num("CLOSE_PRICE"), "prev": num("PREV_CLOSE"), "volume": num("TTL_TRD_QNTY"),
        "turnover": num("TURNOVER_LACS") * 1e5, "date": pd.to_datetime(d["DATE1"].str.strip(), format="%d-%b-%Y", errors="coerce")})
    return out.dropna(subset=["close", "prev"]).query("close > 0 and prev > 0").reset_index(drop=True)[COLS]


def merge(nse: pd.DataFrame | None, bse: pd.DataFrame | None) -> pd.DataFrame:
    """One row per company: NSE where listed there, otherwise the BSE listing. Names come from BSE when NSE has none."""
    frames = []
    if nse is not None and len(nse):
        n = nse.copy()
        if bse is not None and len(bse):
            # Borrow the company name from the BSE row of the same company: by ISIN when both sides have one,
            # otherwise by trading symbol. An empty ISIN must never match (it would copy one name onto everything).
            with_isin = bse[bse["isin"].astype(str).str.len() > 0].drop_duplicates("isin").set_index("isin")["name"]
            by_sym = bse.drop_duplicates("symbol").set_index("symbol")["name"]
            has = n["isin"].astype(str).str.len() > 0
            from_isin = n["isin"].where(has).map(with_isin)
            n["name"] = from_isin.fillna(n["symbol"].map(by_sym)).fillna(n["name"])
        frames.append(n)
    if bse is not None and len(bse):
        if nse is not None and len(nse):
            has_isin = bse["isin"].astype(str).str.len() > 0
            dup = (has_isin & bse["isin"].isin(set(nse["isin"]) - {""})) | (~has_isin & bse["symbol"].isin(set(nse["symbol"])))
            b = bse[~dup]
        else:
            b = bse
        frames.append(b)
    if not frames:
        return pd.DataFrame(columns=COLS)
    out = pd.concat(frames, ignore_index=True)
    out["chg"] = out["close"] / out["prev"] - 1
    return out.drop_duplicates("ticker").reset_index(drop=True)


def _isin_map() -> dict[str, str]:
    try:
        from . import market

        u = market.nse_universe()
        _NSE_NAMES.update(dict(zip(u["symbol"], u["name"].map(nice_name))))
        return dict(zip(u["symbol"], u["isin"]))
    except Exception:  # noqa: BLE001
        return {}


_NSE_NAMES: dict[str, str] = {}


def _fixture() -> pd.DataFrame:
    fx = os.getenv("JINNI_FIXTURES")
    if not fx or not (Path(fx) / "bse_bhavcopy_sample.csv").exists():
        return pd.DataFrame(columns=COLS)
    return merge(None, parse_bse((Path(fx) / "bse_bhavcopy_sample.csv").read_text().replace("\r", "")))


def live_now() -> tuple[pd.DataFrame, dict] | None:
    """Current quotes for every NSE and BSE stock from Yahoo's screener (delayed a few minutes), or None."""
    if demo_forced():
        return None
    from . import live

    nse = live.exchange_quotes("NSI")
    if nse is None or len(nse) < 500:
        return None
    out = merge(nse, live.exchange_quotes("BSE"))
    return out, {"source": "Yahoo live quotes (NSE + BSE)", "date": date.today(), "live": True}


def latest(max_back: int = 7, prefer_live: bool = False) -> tuple[pd.DataFrame, dict]:
    """``prefer_live``: on a trading day whose official file is not out yet, use live quotes for both exchanges so
    every row is from the same moment (movers). The Finder keeps the official file: it needs full-day turnover."""
    got = None
    if prefer_live and date.today().weekday() < 5 and not (cache_dir() / f"bhav_{date.today():%Y%m%d}.pkl").exists():
        got = live_now()
    out, info = got if got else _latest(max_back)
    if len(out):
        from .universe import NAME_HINTS

        NAME_HINTS.update(dict(zip(out["ticker"], out["name"])))
    return out, info


def _latest(max_back: int = 7) -> tuple[pd.DataFrame, dict]:
    """(all listed stocks with the latest EOD prices, info). Empty frame + info['error'] if nothing is reachable."""
    if demo_forced():
        return _fixture(), {"source": "test fixture", "date": None}
    cache = cache_dir() / "bhav_latest.pkl"
    day = date.today()
    for back in range(max_back):
        d = day - timedelta(days=back)
        if d.weekday() >= 5:
            continue
        f = cache_dir() / f"bhav_{d:%Y%m%d}.pkl"
        if f.exists():
            try:
                got, src0 = pd.read_pickle(f)
                return got, {"source": src0 + " (cached)", "date": d}
            except Exception:  # noqa: BLE001
                pass
        bt = _get(BSE_URL.format(d=d), "https://www.bseindia.com/")
        nt = _get(NSE_URL.format(d=d), "https://www.nseindia.com/")
        if not bt and not nt:
            continue
        bse = parse_bse(bt) if bt else None
        nse = parse_nse(nt, _isin_map()) if nt else None
        if nse is not None and _NSE_NAMES:  # NSE's price file has symbols only: take names from NSE's listing file
            nse["name"] = nse["symbol"].map(_NSE_NAMES).fillna(nse["name"])
        via = []
        if bse is None:  # BSE's site often refuses cloud servers: fall back to Yahoo's list of BSE stocks
            from . import live

            bse = live.exchange_quotes("BSE")
            via.append("BSE via Yahoo" if bse is not None else "BSE unavailable")
        out = merge(nse, bse)
        if out.empty:
            continue
        src = " + ".join(x for x, t in (("NSE", nt), ("BSE", bt)) if t) + " bhavcopy" + (f" + {via[0]}" if via else "")
        try:
            pd.to_pickle((out, src), f)
        except Exception:  # noqa: BLE001
            pass
        return out, {"source": src, "date": d}
    # Neither exchange file: build the whole list from Yahoo's screener (both exchanges) before giving up.
    from . import live

    nse, bse = live.exchange_quotes("NSI"), live.exchange_quotes("BSE")
    out = merge(nse, bse)
    if len(out):
        return out, {"source": "Yahoo screener (exchange files unreachable)", "date": day}
    return pd.DataFrame(columns=COLS), {"source": None, "date": None, "error": "NSE and BSE end-of-day files are not reachable right now."}
