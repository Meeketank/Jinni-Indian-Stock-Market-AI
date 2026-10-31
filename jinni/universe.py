"""Tradable universe, benchmarks and market constants for NSE/BSE."""

from __future__ import annotations

# Indian market constants
TRADING_DAYS = 252
# 10Y G-Sec / 91D T-bill area. Users can override in the app settings.
DEFAULT_RISK_FREE = 0.065
BENCHMARK = "^NSEI"

INDICES = {
    "^NSEI": "NIFTY 50",
    "^BSESN": "SENSEX",
    "^NSEBANK": "BANK NIFTY",
    "^CNXIT": "NIFTY IT",
    "^INDIAVIX": "INDIA VIX",
}

SECTOR_INDICES = {
    "^NSEBANK": "Banks",
    "^CNXIT": "IT",
    "^CNXAUTO": "Auto",
    "^CNXPHARMA": "Pharma",
    "^CNXFMCG": "FMCG",
    "^CNXMETAL": "Metal",
    "^CNXENERGY": "Energy",
    "^CNXREALTY": "Realty",
    "^CNXINFRA": "Infra",
    "^CNXMEDIA": "Media",
    "^CNXPSUBANK": "PSU Banks",
}

MACRO = {
    "USDINR=X": "USD/INR",
    "GC=F": "Gold",
    "BZ=F": "Brent",
}

# NIFTY 50 membership and sectors come live from NSE (see constituents.py); nothing is hard-coded here.
from .constituents import NIFTY50  # noqa: E402,F401

# Historical stress windows replayed against today's portfolio.
STRESS_SCENARIOS = {
    "COVID crash (Feb–Mar 2020)": ("2020-02-19", "2020-03-23"),
    "Russia–Ukraine shock (Jan–Mar 2022)": ("2022-01-17", "2022-03-07"),
    "Global rate-hike selloff (2022 H1)": ("2022-04-04", "2022-06-17"),
    "Adani–Hindenburg (Jan–Feb 2023)": ("2023-01-24", "2023-02-28"),
    "Election-result day (4 Jun 2024)": ("2024-06-03", "2024-06-04"),
    "FPI exodus correction (Sep 2024–Feb 2025)": ("2024-09-26", "2025-02-28"),
}


def normalize_symbol(raw: str) -> str:
    """Turn user input like 'tcs', 'HDFC', 'RELIANCE.BO' into a Yahoo ticker."""
    s = raw.strip().upper().replace(" ", "")
    if not s:
        return s
    if s.startswith("^") or "=" in s:
        return s
    base, dot, suffix = s.partition(".")
    if dot and suffix in {"NS", "BO"}:
        return f"{base}.{suffix}"
    return f"{base}.NS"


def display_name(symbol: str) -> str:
    """Short label for charts and tables."""
    if symbol in INDICES:
        return INDICES[symbol]
    if symbol in SECTOR_INDICES:
        return SECTOR_INDICES[symbol]
    if symbol in MACRO:
        return MACRO[symbol]
    return symbol.replace(".NS", "").replace(".BO", " (BSE)")


def sector_of(symbol: str) -> str:
    return NIFTY50.get(symbol, "Other")  # live NSE industry for index members, else unknown
