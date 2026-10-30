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

# NIFTY 50 constituents (late-2025 composition) with GICS-style sectors.
# Index changes are applied by NSE twice a year; tickers that fail to load are
# skipped automatically, so a stale entry never breaks the app.
NIFTY50 = {
    "ADANIENT.NS": "Metals & Mining",
    "ADANIPORTS.NS": "Services",
    "APOLLOHOSP.NS": "Healthcare",
    "ASIANPAINT.NS": "Consumer Durables",
    "AXISBANK.NS": "Financials",
    "BAJAJ-AUTO.NS": "Auto",
    "BAJFINANCE.NS": "Financials",
    "BAJAJFINSV.NS": "Financials",
    "BEL.NS": "Capital Goods",
    "BHARTIARTL.NS": "Telecom",
    "CIPLA.NS": "Healthcare",
    "COALINDIA.NS": "Energy",
    "DRREDDY.NS": "Healthcare",
    "EICHERMOT.NS": "Auto",
    "ETERNAL.NS": "Consumer Services",
    "GRASIM.NS": "Materials",
    "HCLTECH.NS": "IT",
    "HDFCBANK.NS": "Financials",
    "HDFCLIFE.NS": "Financials",
    "HINDALCO.NS": "Metals & Mining",
    "HINDUNILVR.NS": "FMCG",
    "ICICIBANK.NS": "Financials",
    "INDIGO.NS": "Services",
    "INFY.NS": "IT",
    "ITC.NS": "FMCG",
    "JIOFIN.NS": "Financials",
    "JSWSTEEL.NS": "Metals & Mining",
    "KOTAKBANK.NS": "Financials",
    "LT.NS": "Construction",
    "M&M.NS": "Auto",
    "MARUTI.NS": "Auto",
    "MAXHEALTH.NS": "Healthcare",
    "NESTLEIND.NS": "FMCG",
    "NTPC.NS": "Power",
    "ONGC.NS": "Energy",
    "POWERGRID.NS": "Power",
    "RELIANCE.NS": "Energy",
    "SBILIFE.NS": "Financials",
    "SBIN.NS": "Financials",
    "SHRIRAMFIN.NS": "Financials",
    "SUNPHARMA.NS": "Healthcare",
    "TATACONSUM.NS": "FMCG",
    "TMPV.NS": "Auto",
    "TATASTEEL.NS": "Metals & Mining",
    "TCS.NS": "IT",
    "TECHM.NS": "IT",
    "TITAN.NS": "Consumer Durables",
    "TRENT.NS": "Consumer Services",
    "ULTRACEMCO.NS": "Materials",
    "WIPRO.NS": "IT",
}

# Old tickers that users commonly type, mapped to the current listing.
ALIASES = {
    "HDFC": "HDFCBANK",          # merged into HDFC Bank, July 2023
    "ZOMATO": "ETERNAL",         # renamed March 2025
    "TATAMOTORS": "TMPV",        # demerged Oct 2025 (passenger vehicles)
    "MM": "M&M",
}

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
    base = ALIASES.get(base, base)
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
    return NIFTY50.get(symbol, "Other")
