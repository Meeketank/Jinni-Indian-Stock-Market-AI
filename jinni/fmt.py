"""Number formatting shared by the Streamlit app and the web API (Indian conventions)."""

from __future__ import annotations

import math


def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and not math.isfinite(x))


def inr(x, decimals: int = 0) -> str:
    """₹ with lakh / crore suffixes for big numbers."""
    if _bad(x):
        return "—"
    a, sign = abs(x), "-" if x < 0 else ""
    if a >= 1e7:
        return f"{sign}₹{a / 1e7:,.2f} Cr"
    if a >= 1e5:
        return f"{sign}₹{a / 1e5:,.2f} L"
    return f"{sign}₹{a:,.{decimals}f}"


def pct(x, decimals: int = 2, sign: bool = False) -> str:
    if _bad(x):
        return "—"
    return f"{x * 100:+.{decimals}f}%" if sign else f"{x * 100:.{decimals}f}%"


def num(x, decimals: int = 2) -> str:
    if _bad(x):
        return "—"
    return f"{x:,.{decimals}f}"


def tone(x) -> str:
    try:
        return "up" if x > 0 else "down" if x < 0 else ""
    except TypeError:
        return ""
