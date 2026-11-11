"""Trend state, what history did after the same state, and a multi-horizon price forecast.

Nothing here claims to know the future price. The forecast is a distribution: the middle (median), a likely range
(5th to 95th percentile) and the chance of finishing above or below given levels, from a block-bootstrap of the stock's
own daily bars rescaled to today's volatility. The trend section reports where the stock is now and what it actually did,
historically, in the months after it was last in the same state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import forecast as fc
from . import indicators as ind

LABELS = {2: "Strong uptrend", 1: "Uptrend", 0: "Sideways", -1: "Downtrend", -2: "Strong downtrend"}
HORIZONS = {"1 week": 5, "2 weeks": 10, "1 month": 21, "3 months": 63, "6 months": 126}


def state_series(df: pd.DataFrame) -> pd.Series:
    """Trend state for every day, from -2 (strong downtrend) to +2 (strong uptrend). Uses only past data."""
    c = df["Close"]
    s50, s200 = ind.sma(c, 50), c.rolling(200, min_periods=120).mean()
    slope = np.log(c).diff(63)  # 3-month log change
    pts = (c > s50).astype(int) + (c > s200).astype(int) + (s50 > s200).astype(int) + (slope > 0.03).astype(int) \
        - (slope < -0.03).astype(int)  # 0..4, minus one for a falling quarter
    st = pd.Series(0, index=c.index)
    st[pts >= 4] = 2
    st[pts == 3] = 1
    st[pts == 1] = -1
    st[pts <= 0] = -2
    return st.where(s200.notna())


def describe(df: pd.DataFrame) -> dict:
    """Where the trend stands today, in numbers."""
    c = df["Close"]
    st = state_series(df)
    cur = int(st.iloc[-1]) if pd.notna(st.iloc[-1]) else 0
    y = np.log(c.iloc[-63:].values)
    x = np.arange(len(y))
    k, b = np.polyfit(x, y, 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2) if len(y) > 5 else np.nan
    adx_v = float(ind.adx(df)[0].iloc[-1])
    s50, s200 = float(ind.sma(c, 50).iloc[-1]), float(c.rolling(200, min_periods=120).mean().iloc[-1])
    days = int((st.iloc[::-1] != cur).values.argmax()) if (st != cur).any() else len(st)
    return {"state": cur, "label": LABELS[cur], "slope_month": float(np.expm1(k * 21)), "r2": r2, "adx": adx_v,
            "vs50": float(c.iloc[-1] / s50 - 1), "vs200": float(c.iloc[-1] / s200 - 1) if np.isfinite(s200) else np.nan,
            "days_in_state": days, "hi52": float(c.iloc[-1] / c.iloc[-252:].max() - 1), "lo52": float(c.iloc[-1] / c.iloc[-252:].min() - 1)}


def after_state(df: pd.DataFrame, horizons: dict[str, int] | None = None) -> pd.DataFrame:
    """What this stock did after past days in each trend state (rows) over each horizon: average, up %, count."""
    st, c = state_series(df), df["Close"]
    rows = []
    for name, h in (horizons or {"1 week": 5, "1 month": 21, "3 months": 63}).items():
        fwd = c.shift(-h) / c - 1
        for s, lab in LABELS.items():
            m = (st == s) & fwd.notna()
            if m.sum() >= 20:
                r = fwd[m]
                rows.append({"State": lab, "Horizon": name, "Days": int(m.sum()), "Avg move": float(r.mean()),
                             "Median": float(r.median()), "Up %": float((r > 0).mean()), "Worst 5%": float(r.quantile(0.05)),
                             "s": s})
    return pd.DataFrame(rows)


def price_forecast(df: pd.DataFrame, horizons: dict[str, int] | None = None, n_paths: int = 4000) -> pd.DataFrame:
    """Per horizon: median price, likely range and the chance of finishing above / below a few levels."""
    price = float(df["Close"].iloc[-1])
    rows = []
    for name, h in (horizons or HORIZONS).items():
        try:
            sim = fc.simulate(df, h, 0.05, None, n_paths=n_paths, annual_drift=fc.MARKET_DRIFT)
        except ValueError:
            continue
        r = sim.terminal
        rows.append({"Horizon": name, "Days": h, "Low": price * (1 + float(np.percentile(r, 5))), "Median": price * (1 + float(np.median(r))),
                     "High": price * (1 + float(np.percentile(r, 95))), "P up": float((r > 0).mean()),
                     "P +5%": float((r >= 0.05).mean()), "P +10%": float((r >= 0.10).mean()), "P +20%": float((r >= 0.20).mean()),
                     "P -5%": float((r <= -0.05).mean()), "P -10%": float((r <= -0.10).mean())})
    return pd.DataFrame(rows)
