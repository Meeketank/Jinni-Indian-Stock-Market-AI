"""Vectorised single-stock strategy backtester with Indian transaction costs.

Signals are computed on day *t*'s close and the position is taken from day
*t+1*, so there is no look-ahead. Costs are charged on every change in
position (turnover).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from .risk import performance_stats
from .signals import factor_scores

# Delivery equity, round trip ≈ STT 0.1%×2 + exchange/SEBI/stamp/GST + slippage.
DEFAULT_COST_BPS = 15.0  # per side

STRATEGIES = {
    "trend": "Trend: long when 50-DMA > 200-DMA",
    "composite": "JINNI composite score ≥ 60 (exit < 45)",
    "rsi_reversion": "RSI(2) mean reversion above 200-DMA",
    "breakout": "Donchian 55-day breakout, 20-day exit",
    "buy_hold": "Buy & hold",
}


def positions(df: pd.DataFrame, strategy: str, bench: pd.Series | None = None) -> pd.Series:
    c = df["Close"]
    if strategy == "buy_hold":
        pos = pd.Series(1.0, index=c.index)
    elif strategy == "trend":
        pos = (ind.sma(c, 50) > ind.sma(c, 200)).astype(float)
    elif strategy == "composite":
        s = factor_scores(df, bench)["Composite"]
        pos = pd.Series(np.nan, index=c.index)
        pos[s >= 60] = 1.0
        pos[s < 45] = 0.0
        pos = pos.ffill().fillna(0.0)
    elif strategy == "rsi_reversion":
        r2 = ind.rsi(c, 2)
        regime = c > ind.sma(c, 200)
        pos = pd.Series(np.nan, index=c.index)
        pos[(r2 < 10) & regime] = 1.0
        pos[(r2 > 70) | ~regime] = 0.0
        pos = pos.ffill().fillna(0.0)
    elif strategy == "breakout":
        hi = df["High"].rolling(55).max().shift(1)
        lo = df["Low"].rolling(20).min().shift(1)
        pos = pd.Series(np.nan, index=c.index)
        pos[c > hi] = 1.0
        pos[c < lo] = 0.0
        pos = pos.ffill().fillna(0.0)
    else:
        raise ValueError(strategy)
    return pos


def run(df: pd.DataFrame, strategy: str, bench: pd.Series | None = None,
        cost_bps: float = DEFAULT_COST_BPS, warmup: int = 200) -> dict:
    c = df["Close"]
    pos = positions(df, strategy, bench).shift(1).fillna(0.0)  # trade next bar
    r = c.pct_change().fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    strat = pos * r - turnover * cost_bps / 1e4
    strat = strat.iloc[warmup:]
    bh = r.iloc[warmup:]
    b = bench.pct_change().reindex(strat.index) if bench is not None else None
    trades = int((pos.diff().iloc[warmup:] > 0).sum())
    # per-trade stats
    entries = pos.iloc[warmup:].diff() > 0
    trade_id = entries.cumsum()[pos.iloc[warmup:] > 0]
    per_trade = (1 + strat.loc[trade_id.index]).groupby(trade_id).prod() - 1
    stats = performance_stats(strat, b)
    stats.update({
        "Trades": trades,
        "Win rate": float((per_trade > 0).mean()) if len(per_trade) else np.nan,
        "Avg trade": float(per_trade.mean()) if len(per_trade) else np.nan,
        "Time in market": float(pos.iloc[warmup:].mean()),
        "Costs paid": float((turnover.iloc[warmup:] * cost_bps / 1e4).sum()),
    })
    return {
        "equity": (1 + strat).cumprod(),
        "buy_hold": (1 + bh).cumprod(),
        "position": pos.iloc[warmup:],
        "returns": strat,
        "stats": stats,
    }
