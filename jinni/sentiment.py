"""Sentiment: headline scoring and a market mood (fear/greed) index.

Headline scoring uses a finance-specific lexicon (in the spirit of the
Loughran-McDonald dictionary, plus Indian-market terms) with negation and
intensifier handling. It is fast, transparent and needs no model download.
It reads tone, not truth: a positive headline is not a forecast.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import indicators as ind

POS = {
    "beat": 2, "beats": 2, "surge": 2, "surges": 2, "soar": 2, "soars": 2, "jump": 1.5, "jumps": 1.5, "rally": 1.5,
    "rallies": 1.5, "gain": 1, "gains": 1, "rise": 1, "rises": 1, "up": 0.5, "high": 0.5, "record": 1.5,
    "strong": 1.5, "robust": 1.5, "growth": 1, "profit": 1, "profits": 1, "upgrade": 2, "upgraded": 2,
    "outperform": 2, "overweight": 1.5, "buy": 1, "bullish": 2, "positive": 1, "optimistic": 1.5, "boost": 1.5,
    "boosts": 1.5, "wins": 1.5, "win": 1, "order": 0.5, "orders": 0.5, "contract": 0.5, "approval": 1.5,
    "approved": 1.5, "expansion": 1, "expands": 1, "dividend": 1, "bonus": 1, "buyback": 1.5, "recovery": 1,
    "rebound": 1.5, "rebounds": 1.5, "upbeat": 1.5, "outlook": 0.3, "raises": 1, "raised": 1, "hike": 0.3,
    "breakout": 1.5, "multibagger": 1.5, "top": 0.5, "best": 1, "improves": 1, "improved": 1, "margin": 0.3,
    "inflows": 1, "accumulate": 1, "stake": 0.3, "acquires": 0.5, "partnership": 1, "launch": 0.5, "launches": 0.5,
    "cut": 0, "easing": 1, "stimulus": 1, "steady": 0.5, "resilient": 1,
}
NEG = {
    "miss": 2, "misses": 2, "fall": 1.5, "falls": 1.5, "drop": 1.5, "drops": 1.5, "plunge": 2.5, "plunges": 2.5,
    "crash": 3, "slump": 2, "slumps": 2, "tumble": 2, "tumbles": 2, "decline": 1.5, "declines": 1.5, "down": 0.5,
    "loss": 2, "losses": 2, "weak": 1.5, "weaker": 1.5, "downgrade": 2, "downgraded": 2, "underperform": 2,
    "underweight": 1.5, "sell": 1, "bearish": 2, "negative": 1, "concern": 1, "concerns": 1, "worry": 1.5,
    "worries": 1.5, "fear": 1.5, "fears": 1.5, "risk": 0.5, "risks": 0.5, "probe": 2, "raid": 2.5, "fraud": 3,
    "penalty": 2, "fine": 1, "fined": 2, "ban": 2, "banned": 2, "default": 3, "debt": 0.5, "pledge": 1.5,
    "pledged": 1.5, "resigns": 2, "resignation": 2, "lawsuit": 2, "scam": 3, "delay": 1, "delayed": 1,
    "slowdown": 1.5, "recession": 2, "inflation": 0.5, "outflows": 1.5, "selloff": 2, "sell-off": 2, "volatile": 0.5,
    "cuts": 0.5, "lowers": 1, "lowered": 1, "warning": 1.5, "warns": 1.5, "tariff": 1.5, "tariffs": 1.5,
    "war": 2, "sanctions": 1.5, "strike": 1, "shutdown": 1.5, "insolvency": 3, "bankruptcy": 3, "halt": 1.5,
    "worst": 1.5, "low": 0.5, "lower": 0.5, "pressure": 1, "dip": 1, "dips": 1, "slips": 1, "slip": 1,
    "sebi": 0.3, "notice": 0.5, "uncertainty": 1.5, "turmoil": 2,
}
NEGATORS = {"not", "no", "never", "without", "despite", "fails", "failed", "lack", "hardly"}
INTENS = {"sharply": 1.5, "significantly": 1.4, "massive": 1.5, "huge": 1.4, "strongly": 1.3, "slightly": 0.6,
          "marginally": 0.6, "record": 1.3}


def score_headline(text: str) -> float:
    """Tone in [-1, 1]."""
    words = re.findall(r"[a-z][a-z\-']*", text.lower())
    total = 0.0
    for i, w in enumerate(words):
        v = POS.get(w, 0) - NEG.get(w, 0)
        if v == 0:
            continue
        window = words[max(0, i - 3):i]
        if any(x in NEGATORS for x in window):
            v = -0.7 * v
        mult = max([INTENS.get(x, 1.0) for x in window] + [1.0])
        total += v * mult
    return float(np.tanh(total / 3))


def score_news(news: pd.DataFrame, half_life_hours: float = 48) -> dict:
    if news is None or news.empty:
        return {"score": np.nan, "n": 0, "pos": 0, "neg": 0, "label": "No news"}
    s = news["Headline"].map(score_headline)
    if "Time" in news and news["Time"].notna().any():
        now = pd.Timestamp.now(tz=news["Time"].dropna().iloc[0].tz)
        age = news["Time"].map(lambda t: (now - t).total_seconds() / 3600 if t is not None else 999)
        w = 0.5 ** (age / half_life_hours)
    else:
        w = pd.Series(1.0, index=news.index)
    score = float((s * w).sum() / w.sum()) if w.sum() > 0 else float(s.mean())
    label = ("Very positive" if score > 0.35 else "Positive" if score > 0.1 else "Very negative" if score < -0.35
             else "Negative" if score < -0.1 else "Neutral")
    return {"score": score, "n": len(s), "pos": int((s > 0.1).sum()), "neg": int((s < -0.1).sum()), "label": label,
            "scores": s}


# ------------------------------------------------------------ market mood


def _pct_rank(s: pd.Series, window: int = 252) -> float:
    s = s.dropna().iloc[-window:]
    if len(s) < 20:
        return np.nan
    return float((s < s.iloc[-1]).mean() * 100)


def market_mood(nifty: pd.Series, vix: pd.Series | None, stocks_close: pd.DataFrame | None,
                gold: pd.Series | None = None) -> dict:
    """0 = extreme fear, 100 = extreme greed. Each component is today's reading
    ranked against its own last year (percentile), so it adapts to regimes."""
    comp = {}
    mom = nifty / nifty.rolling(125).mean() - 1
    comp["Momentum (NIFTY vs 125-DMA)"] = _pct_rank(mom)
    if vix is not None and vix.notna().sum() > 60:
        rel = vix / vix.rolling(50).mean()
        comp["Volatility (India VIX, inverted)"] = 100 - _pct_rank(rel)
    if stocks_close is not None and stocks_close.shape[1] >= 10:
        breadth = (stocks_close > stocks_close.rolling(50).mean()).mean(axis=1)
        comp["Breadth (% above 50-DMA)"] = _pct_rank(breadth)
        hi = (stocks_close >= stocks_close.rolling(252, min_periods=120).max() * 0.98).sum(axis=1)
        lo = (stocks_close <= stocks_close.rolling(252, min_periods=120).min() * 1.02).sum(axis=1)
        comp["New highs vs lows"] = _pct_rank((hi - lo).rolling(5).mean())
    if gold is not None and gold.notna().sum() > 60:
        g = gold.reindex(nifty.index).ffill()
        rel = nifty.pct_change(20) - g.pct_change(20)
        comp["Safe-haven demand (equity vs gold)"] = _pct_rank(rel)
    rsi = ind.rsi(nifty)
    comp["Short-term strength (RSI)"] = float(np.clip(rsi.iloc[-1], 0, 100)) if rsi.notna().any() else np.nan
    vals = [v for v in comp.values() if np.isfinite(v)]
    score = float(np.mean(vals)) if vals else np.nan
    label = ("Extreme fear" if score < 20 else "Fear" if score < 40 else "Neutral" if score <= 60
             else "Greed" if score <= 80 else "Extreme greed")
    return {"score": score, "label": label, "components": comp}


def mood_history(nifty: pd.Series, vix: pd.Series | None, stocks_close: pd.DataFrame | None,
                 window: int = 252) -> pd.Series:
    """Rolling version of the mood score for charting (momentum, VIX, breadth)."""
    parts = []
    mom = nifty / nifty.rolling(125).mean() - 1
    parts.append(mom.rolling(window, min_periods=60).rank(pct=True) * 100)
    if vix is not None and vix.notna().sum() > 60:
        rel = (vix / vix.rolling(50).mean()).reindex(nifty.index)
        parts.append(100 - rel.rolling(window, min_periods=60).rank(pct=True) * 100)
    if stocks_close is not None and stocks_close.shape[1] >= 10:
        b = (stocks_close > stocks_close.rolling(50).mean()).mean(axis=1).reindex(nifty.index)
        parts.append(b.rolling(window, min_periods=60).rank(pct=True) * 100)
    return pd.concat(parts, axis=1).mean(axis=1)
