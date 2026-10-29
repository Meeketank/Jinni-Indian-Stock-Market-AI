"""Technical indicators, vectorised over pandas Series.

RSI, ATR and ADX use Wilder's smoothing (alpha = 1/period), which is the
definition used by TradingView, Zerodha Kite and most charting platforms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .universe import TRADING_DAYS


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0), n)
    loss = _wilder(-delta.clip(upper=0), n)
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # all gains and no losses -> 100; flat -> 50
    out = out.where(loss != 0, np.where(gain > 0, 100.0, 50.0))
    return out.where(gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["Close"].shift(1)
    return pd.concat(
        [df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _wilder(true_range(df), n)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid + k * sd, mid, mid - k * sd


def adx(df: pd.DataFrame, n: int = 14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = _wilder(true_range(df), n)
    plus_di = 100 * _wilder(plus_dm, n) / tr
    minus_di = 100 * _wilder(minus_dm, n) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx, n), plus_di, minus_di


def stochastic(df: pd.DataFrame, n: int = 14, d: int = 3):
    lo = df["Low"].rolling(n).min()
    hi = df["High"].rolling(n).max()
    k = 100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(d).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"]).cumsum()


def realized_vol(close: pd.Series, n: int = 20) -> pd.Series:
    """Annualised close-to-close volatility from log returns."""
    return np.log(close).diff().rolling(n).std() * np.sqrt(TRADING_DAYS)


def vwap(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).rolling(n).sum() / df["Volume"].rolling(n).sum()


def pivot_levels(df: pd.DataFrame) -> dict:
    """Classic floor pivots from the last completed session."""
    last = df.iloc[-1]
    p = (last["High"] + last["Low"] + last["Close"]) / 3
    return {
        "R2": p + (last["High"] - last["Low"]),
        "R1": 2 * p - last["Low"],
        "Pivot": p,
        "S1": 2 * p - last["High"],
        "S2": p - (last["High"] - last["Low"]),
    }


def support_resistance(close: pd.Series, window: int = 120, order: int = 5, tol: float = 0.015) -> dict:
    """Cluster swing highs/lows over the window into nearest support/resistance."""
    s = close.iloc[-window:]
    if len(s) < 2 * order + 1:
        return {"support": None, "resistance": None}
    roll_max = s.rolling(2 * order + 1, center=True).max()
    roll_min = s.rolling(2 * order + 1, center=True).min()
    highs = s[s == roll_max].values
    lows = s[s == roll_min].values
    price = s.iloc[-1]
    res = sorted([h for h in highs if h > price * (1 + tol / 3)])
    sup = sorted([lv for lv in lows if lv < price * (1 - tol / 3)], reverse=True)
    return {"support": sup[0] if sup else None, "resistance": res[0] if res else None}


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with every indicator as a column."""
    out = df.copy()
    c = out["Close"]
    for n in (20, 50, 200):
        out[f"SMA{n}"] = sma(c, n)
    out["EMA21"] = ema(c, 21)
    out["RSI"] = rsi(c)
    out["MACD"], out["MACD_signal"], out["MACD_hist"] = macd(c)
    out["ATR"] = atr(out)
    out["BB_up"], out["BB_mid"], out["BB_lo"] = bollinger(c)
    out["ADX"], out["+DI"], out["-DI"] = adx(out)
    out["STOCH_K"], out["STOCH_D"] = stochastic(out)
    out["OBV"] = obv(out)
    out["VOL20"] = realized_vol(c)
    out["VWAP20"] = vwap(out)
    return out
