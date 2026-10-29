"""Multi-factor stock rating and trade planning.

The rating combines five transparent factor scores (each 0-100):

* Trend      – price vs 50/200-day averages, 50-day slope, ADX strength
* Momentum   – 12-1 month and 3 month returns, relative strength vs NIFTY
* Timing     – RSI / Bollinger %B / MACD (penalises overbought entries)
* Volume     – OBV trend and participation on up days
* Risk       – realised volatility and distance from 52-week high

Scores are computed as full time series so the same rule can be back-tested
(see ``score_history_stats``), which is how the app reports *measured*
hit-rates instead of an invented confidence number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind

WEIGHTS = {"Trend": 0.30, "Momentum": 0.25, "Timing": 0.15, "Volume": 0.10, "Risk": 0.20}


def _scale(x: pd.Series, lo: float, hi: float) -> pd.Series:
    """Linearly map x from [lo, hi] to [0, 100] and clip."""
    return ((x - lo) / (hi - lo) * 100).clip(0, 100)


def factor_scores(df: pd.DataFrame, bench_close: pd.Series | None = None) -> pd.DataFrame:
    """Vectorised factor scores for every bar. ``df`` needs OHLCV columns."""
    c = df["Close"]
    sma50, sma200 = ind.sma(c, 50), ind.sma(c, 200)
    adx_v, pdi, mdi = ind.adx(df)

    # Trend
    above50 = (c > sma50).astype(float)
    above200 = (c > sma200).astype(float)
    golden = (sma50 > sma200).astype(float)
    slope50 = sma50.pct_change(10) * 100
    trend_dir = np.where(pdi > mdi, 1.0, 0.0)
    trend = (
        25 * above50 + 20 * above200 + 20 * golden
        + 20 * _scale(slope50, -3, 3) / 100
        + 15 * trend_dir * _scale(adx_v, 15, 40) / 100
    )

    # Momentum: 12-1 month (skip last month to avoid short-term reversal), 3 month
    mom_12_1 = c.shift(21) / c.shift(252) - 1
    mom_3 = c / c.shift(63) - 1
    parts = [_scale(mom_12_1 * 100, -30, 50), _scale(mom_3 * 100, -15, 20)]
    if bench_close is not None:
        b = bench_close.reindex(c.index).ffill()
        rs = (c / c.shift(63)) / (b / b.shift(63)) - 1
        parts.append(_scale(rs * 100, -15, 15))
    momentum = pd.concat(parts, axis=1).mean(axis=1, skipna=True)

    # Timing: best entries are RSI 45-65 (healthy), not >75 (stretched) or <30 (falling knife)
    r = ind.rsi(c)
    rsi_score = 100 - (r - 57).abs().clip(upper=40) * 2.5
    up, mid, lo = ind.bollinger(c)
    pctb = (c - lo) / (up - lo)
    bb_score = 100 - (pctb - 0.6).abs().clip(upper=1) * 100
    _, _, hist = ind.macd(c)
    macd_score = np.where(hist > 0, 70.0, 30.0) + np.where(hist > hist.shift(1), 30.0, 0.0)
    timing = (rsi_score + bb_score + pd.Series(macd_score, index=c.index).clip(0, 100)) / 3

    # Volume: OBV above its 20-day mean, up-volume share over 20 days
    o = ind.obv(df)
    obv_up = (o > o.rolling(20).mean()).astype(float) * 50
    up_day = (c.diff() > 0).astype(float)
    up_vol_share = (df["Volume"] * up_day).rolling(20).sum() / df["Volume"].rolling(20).sum()
    volume = obv_up + _scale(up_vol_share, 0.35, 0.65) / 2

    # Risk: lower vol and shallower drawdown from 52w high score better
    vol = ind.realized_vol(c, 60)
    dd_52 = c / c.rolling(252, min_periods=60).max() - 1
    risk = (100 - _scale(vol, 0.15, 0.55)) * 0.5 + _scale(dd_52 * 100, -35, 0) * 0.5

    out = pd.DataFrame(
        {"Trend": trend, "Momentum": momentum, "Timing": timing, "Volume": volume, "Risk": risk},
        index=c.index,
    )
    out["Composite"] = sum(out[k] * w for k, w in WEIGHTS.items())
    return out


def fundamental_score(info: dict) -> tuple[float | None, dict]:
    """Quality/value score (0-100) from a Yahoo info dict. None if no data."""
    def num(k):
        v = info.get(k)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    parts = {}
    pe = num("trailingPE") or num("forwardPE")
    if pe is not None and pe > 0:
        parts["Valuation (P/E)"] = float(np.clip(100 - (pe - 12) * 2, 0, 100))
    roe = num("returnOnEquity")
    if roe is not None:
        parts["Profitability (ROE)"] = float(np.clip(roe * 100 / 25 * 100, 0, 100))
    de = num("debtToEquity")
    if de is not None:
        parts["Balance sheet (D/E)"] = float(np.clip(100 - de / 2, 0, 100))
    g = num("earningsGrowth")
    if g is None:
        g = num("revenueGrowth")
    if g is not None:
        parts["Growth"] = float(np.clip(50 + g * 200, 0, 100))
    pm = num("profitMargins")
    if pm is not None:
        parts["Margins"] = float(np.clip(pm * 100 / 25 * 100, 0, 100))
    if not parts:
        return None, {}
    return float(np.mean(list(parts.values()))), parts


def rating_from_score(score: float) -> str:
    if score >= 72:
        return "STRONG BUY"
    if score >= 60:
        return "BUY"
    if score >= 42:
        return "HOLD"
    if score >= 30:
        return "SELL"
    return "STRONG SELL"


@dataclass
class TradePlan:
    side: str
    entry: float
    stop: float
    targets: list[float]
    risk_per_share: float
    reward_risk: list[float]
    qty: int
    capital_at_risk: float

    def as_rows(self) -> list[dict]:
        rows = [{"Level": "Entry", "Price": self.entry, "Move %": 0.0, "R multiple": 0.0}]
        rows.append({
            "Level": "Stop-loss", "Price": self.stop,
            "Move %": (self.stop / self.entry - 1) * 100, "R multiple": -1.0,
        })
        for i, (t, rr) in enumerate(zip(self.targets, self.reward_risk), 1):
            rows.append({"Level": f"Target {i}", "Price": t, "Move %": (t / self.entry - 1) * 100, "R multiple": rr})
        return rows


def trade_plan(df: pd.DataFrame, side: str, capital: float = 1_000_000, risk_pct: float = 0.01) -> TradePlan:
    """ATR + structure based plan. Stop sits beyond the nearest swing level or
    2×ATR, whichever is tighter-but-safe; targets are 1.5R / 2.5R / 4R, with T1
    pulled to resistance/support if that is closer."""
    price = float(df["Close"].iloc[-1])
    a = float(ind.atr(df).iloc[-1])
    if not np.isfinite(a) or a <= 0:
        a = price * 0.02
    sr = ind.support_resistance(df["Close"])
    long = side != "SHORT"
    if long:
        atr_stop = price - 2 * a
        struct = sr["support"] - 0.25 * a if sr["support"] else None
        stop = max(atr_stop, struct) if struct and struct < price - 0.75 * a else atr_stop
    else:
        atr_stop = price + 2 * a
        struct = sr["resistance"] + 0.25 * a if sr["resistance"] else None
        stop = min(atr_stop, struct) if struct and struct > price + 0.75 * a else atr_stop
    risk = abs(price - stop)
    sign = 1 if long else -1
    multiples = [1.5, 2.5, 4.0]
    targets = [price + sign * m * risk for m in multiples]
    level = sr["resistance"] if long else sr["support"]
    if level and (level - price) * sign > risk and (level - targets[0]) * sign < 0:
        targets[0] = level
    rr = [abs(t - price) / risk for t in targets]
    qty = int((capital * risk_pct) // risk) if risk > 0 else 0
    qty = min(qty, int(capital // price))  # cannot exceed capital
    return TradePlan("LONG" if long else "SHORT", price, stop, targets, risk, rr, qty, qty * risk)


def detect_events(df: pd.DataFrame, lookback: int = 10) -> list[str]:
    """Human-readable list of notable recent technical events."""
    c = df["Close"]
    ev = []
    s50, s200 = ind.sma(c, 50), ind.sma(c, 200)
    diff = (s50 - s200).dropna()
    if len(diff) > lookback:
        recent = np.sign(diff.iloc[-lookback - 1:])
        if (recent.diff().fillna(0) > 0).any():
            ev.append("Golden cross (50-DMA crossed above 200-DMA) in the last two weeks")
        if (recent.diff().fillna(0) < 0).any():
            ev.append("Death cross (50-DMA crossed below 200-DMA) in the last two weeks")
    hi52 = c.rolling(252, min_periods=60).max().iloc[-1]
    lo52 = c.rolling(252, min_periods=60).min().iloc[-1]
    if c.iloc[-1] >= hi52 * 0.98:
        ev.append("Trading within 2% of its 52-week high")
    if c.iloc[-1] <= lo52 * 1.03:
        ev.append("Trading within 3% of its 52-week low")
    r = ind.rsi(c).iloc[-1]
    if r >= 75:
        ev.append(f"RSI {r:.0f}: overbought, pullback risk elevated")
    elif r <= 28:
        ev.append(f"RSI {r:.0f}: oversold, watch for reversal confirmation")
    v = df["Volume"]
    if len(v) > 21 and v.iloc[-21:-1].mean() > 0 and v.iloc[-1] > 2.5 * v.iloc[-21:-1].mean():
        ev.append(f"Volume spike: {v.iloc[-1] / v.iloc[-21:-1].mean():.1f}× the 20-day average")
    up, _, lo = ind.bollinger(c)
    width = ((up - lo) / c).dropna()
    if len(width) > 120 and width.iloc[-1] <= width.iloc[-120:].quantile(0.1):
        ev.append("Bollinger squeeze: volatility at a 6-month low, breakout likely")
    return ev or ["No notable technical events"]


def score_history_stats(scores: pd.Series, close: pd.Series, horizon: int = 20) -> pd.DataFrame:
    """How did forward returns behave in each rating bucket historically?

    This is an in-sample sanity check of the scoring rule on this stock — it
    shows whether higher scores were actually followed by better returns.
    """
    fwd = close.shift(-horizon) / close - 1
    df = pd.DataFrame({"score": scores, "fwd": fwd}).dropna()
    if df.empty:
        return pd.DataFrame()
    df["Rating"] = df["score"].apply(rating_from_score)
    order = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
    g = df.groupby("Rating")["fwd"]
    out = pd.DataFrame({
        "Days": g.size(),
        f"Avg {horizon}d return %": g.mean() * 100,
        "Hit rate % (ret>0)": g.apply(lambda x: (x > 0).mean() * 100),
        f"Worst {horizon}d %": g.min() * 100,
    }).reindex([o for o in order if o in g.groups])
    return out
