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
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": referer, "Accept": "text/csv,*/*"})
        with urllib.request.urlopen(req, timeout=15) as r:
            txt = r.read().decode("utf-8", "ignore")
        return txt if len(txt) > 5000 and "<html" not in txt[:300].lower() else None
    except Exception as e:  # noqa: BLE001
        log.info("bhavcopy %s unreachable: %s", url, e)
        return None


def parse_bse(text: str) -> pd.DataFrame:
    d = pd.read_csv(io.StringIO(text), dtype=str)
    d.columns = [c.strip() for c in d.columns]
    d = d[(d["FinInstrmTp"] == "STK") & d["SctySrs"].isin(BSE_EQUITY_SERIES) & d["ISIN"].str.startswith("INE", na=False)]
    num = lambda c: pd.to_numeric(d[c], errors="coerce")  # noqa: E731
    out = pd.DataFrame({
        "ticker": d["FinInstrmId"].str.strip() + ".BO", "symbol": d["TckrSymb"].str.strip(), "name": d["FinInstrmNm"].str.strip().str.title(),
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
            nm = bse.drop_duplicates("isin").set_index("isin")["name"]
            n["name"] = np.where(n["isin"].isin(nm.index), n["isin"].map(nm), n["name"])
        frames.append(n)
    if bse is not None and len(bse):
        b = bse[~bse["isin"].isin(set(nse["isin"]) if nse is not None and len(nse) else set())]
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
        return dict(zip(u["symbol"], u["isin"]))
    except Exception:  # noqa: BLE001
        return {}


def _fixture() -> pd.DataFrame:
    fx = os.getenv("JINNI_FIXTURES")
    if not fx or not (Path(fx) / "bse_bhavcopy_sample.csv").exists():
        return pd.DataFrame(columns=COLS)
    return merge(None, parse_bse((Path(fx) / "bse_bhavcopy_sample.csv").read_text().replace("\r", "")))


def latest(max_back: int = 7) -> tuple[pd.DataFrame, dict]:
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
                return pd.read_pickle(f), {"source": "NSE + BSE bhavcopy (cached)", "date": d}
            except Exception:  # noqa: BLE001
                pass
        bt = _get(BSE_URL.format(d=d), "https://www.bseindia.com/")
        nt = _get(NSE_URL.format(d=d), "https://www.nseindia.com/")
        if not bt and not nt:
            continue
        bse = parse_bse(bt) if bt else None
        nse = parse_nse(nt, _isin_map()) if nt else None
        out = merge(nse, bse)
        if out.empty:
            continue
        try:
            out.to_pickle(f)
        except Exception:  # noqa: BLE001
            pass
        src = " + ".join(x for x, t in (("NSE", nt), ("BSE", bt)) if t) + " bhavcopy"
        return out, {"source": src, "date": d}
    return pd.DataFrame(columns=COLS), {"source": None, "date": None, "error": "NSE and BSE end-of-day files are not reachable right now."}
