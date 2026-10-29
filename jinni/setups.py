"""Daily chart setups (adapted from the day-trader-skill scanner for NSE daily bars).

Each detector returns a boolean Series (True on days the setup fired), so the
same rule can be replayed over history: "every past time this setup fired".
A setup is a pattern that matched, not a prediction.
"""

from __future__ import annotations

import pandas as pd

from . import indicators as ind


def _breakout_20(df):
    c = df["Close"]
    return c > c.shift(1).rolling(20).max()


def _high_52w(df):
    c = df["Close"]
    return c >= c.shift(1).rolling(252, min_periods=200).max()


def _bull_flag(df):
    """Strong 10-day pole (+8%), then a tight 5-day pause holding the upper half."""
    c = df["Close"]
    pole = c.shift(5) / c.shift(15) - 1
    hi5 = df["High"].rolling(5).max()
    lo5 = df["Low"].rolling(5).min()
    tight = (hi5 - lo5) / c < 0.06
    holds = lo5 > c.shift(15) + 0.5 * (c.shift(5) - c.shift(15))
    return (pole > 0.08) & tight & holds


def _reclaim_20(df):
    c = df["Close"]
    m = ind.sma(c, 20)
    return (c > m) & (c.shift(1) <= m.shift(1))


def _oversold(df):
    return ind.rsi(df["Close"]) < 30


def _oversold_bounce(df):
    r = ind.rsi(df["Close"])
    c = df["Close"]
    return (r.shift(1) < 30) & (c > df["High"].shift(1))


def _volume_breakout(df):
    v = df["Volume"]
    c = df["Close"]
    return (v > 2 * v.shift(1).rolling(20).mean()) & (c > df["Open"]) & (c.pct_change() > 0.02)


def _pullback_uptrend(df):
    """Uptrend (above 50 & 200-DMA) dipping to the 20-DMA with RSI cooled to 40-50."""
    c = df["Close"]
    r = ind.rsi(c)
    return (c > ind.sma(c, 50)) & (ind.sma(c, 50) > ind.sma(c, 200)) & \
        (df["Low"] <= ind.sma(c, 20) * 1.01) & r.between(38, 52)


SETUPS = {
    "20-day breakout": _breakout_20,
    "52-week high": _high_52w,
    "Bull flag": _bull_flag,
    "Reclaimed 20-DMA": _reclaim_20,
    "Pullback in uptrend": _pullback_uptrend,
    "Volume breakout": _volume_breakout,
    "Oversold (RSI<30)": _oversold,
    "Oversold bounce": _oversold_bounce,
}


def setup_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({k: f(df).fillna(False).astype(bool) for k, f in SETUPS.items()}, index=df.index)


def active_setups(df: pd.DataFrame) -> list[str]:
    fr = setup_frame(df)
    return [k for k, v in fr.iloc[-1].items() if v]
