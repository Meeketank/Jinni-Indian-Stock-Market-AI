"""Out-of-sample accuracy of the "chance of reaching +X% within N days" machinery and of the ranking signals.

For every stock and every ``step``-th day the model sees only the data up to that day, states a probability, and the
outcome is read from the following ``h`` days (did the intraday high touch +X%?). Nothing is fitted on the outcomes:
this is a pure walk-forward test, so the numbers are what a user would actually have experienced.

Reported: calibration (does "40%" mean 40%?), Brier score and AUC against simple baselines, and how each signal
(technical score, momentum, relative strength, volatility, the Time-Machine probability) ranks the stocks that went on to
hit the target. Sentiment and fundamentals are not here because free sources have no point-in-time history to test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import forecast as fc
from . import indicators as ind
from . import signals
from .mlcore import roc_auc


def quick_touch(sig_daily: float, target: float, h: int) -> float:
    """Closed-form driftless probability of touching +target within h days (reflection principle)."""
    from scipy.stats import norm

    if not np.isfinite(sig_daily) or sig_daily <= 0:
        return np.nan
    return float(2 * (1 - norm.cdf(np.log1p(target) / (sig_daily * np.sqrt(h)))))


def trade_return(df: pd.DataFrame, t: int, h: int, target: float, stop: float, cost: float = fc.COST_ROUND_TRIP) -> float:
    """Result of buying at day t's close with a target and a stop, bar by bar over the next h days (stop wins ties)."""
    entry = float(df["Close"].iloc[t])
    tp, sl = entry * (1 + target), entry * (1 - stop)
    for k in range(1, h + 1):
        lo, hi = float(df["Low"].iloc[t + k]), float(df["High"].iloc[t + k])
        if lo <= sl:
            return -stop - cost
        if hi >= tp:
            return target - cost
    return float(df["Close"].iloc[t + h]) / entry - 1 - cost


def collect(dfs: dict[str, pd.DataFrame], bench: pd.Series | None, target: float = 0.05, h: int = 21, step: int = 10,
            n_paths: int = 500, min_hist: int = 300) -> pd.DataFrame:
    """One row per (stock, origin day) with every signal known on that day and the realised outcome."""
    rows = []
    for tk, df in dfs.items():
        if len(df) < min_hist + h + 10:
            continue
        sc = signals.factor_scores(df, bench)
        c, hi = df["Close"], df["High"]
        lr = np.log(c).diff()
        vol60 = lr.rolling(60).std()
        mom63 = c / c.shift(63) - 1
        mom21 = c / c.shift(21) - 1
        rs = None
        if bench is not None:
            b = bench.reindex(c.index).ffill()
            rs = (c / c.shift(63)) / (b / b.shift(63)) - 1
        fwd_hi = pd.concat([hi.shift(-k) for k in range(1, h + 1)], axis=1).max(axis=1)
        realised = (fwd_hi / c - 1) >= target
        for t in range(min_hist, len(df) - h - 1, step):
            past = df.iloc[: t + 1]
            try:
                sim = fc.simulate(past, h, target, None, n_paths=n_paths, annual_drift=fc.MARKET_DRIFT)
            except ValueError:
                continue
            stop = fc.auto_stop_pct(past, h, target)
            ret_trade = trade_return(df, t, h, target, stop)
            trail = (past["High"].rolling(h).max().shift(-h) / past["Close"] - 1).dropna()
            trail_rate = float((trail.iloc[-504:] >= target).mean()) if len(trail) > 60 else np.nan
            rows.append({"ticker": tk, "date": df.index[t], "p_sim": sim.p_touch, "p_trail": trail_rate,
                         "p_quick": quick_touch(float(vol60.iloc[t]), target, h), "tech": float(sc["Composite"].iloc[t]),
                         "mom63": float(mom63.iloc[t]), "mom21": float(mom21.iloc[t]),
                         "rs63": float(rs.iloc[t]) if rs is not None else np.nan, "vol": float(vol60.iloc[t] * np.sqrt(252)),
                         "hit": float(realised.iloc[t]), "stop": stop, "ret_trade": ret_trade})
    return pd.DataFrame(rows).dropna(subset=["p_sim", "hit"])


def brier(p: pd.Series, y: pd.Series) -> float:
    m = p.notna() & y.notna()
    return float(((p[m] - y[m]) ** 2).mean())


def calibration(p: pd.Series, y: pd.Series, bins: int = 5) -> pd.DataFrame:
    m = p.notna()
    d = pd.DataFrame({"p": p[m], "y": y[m]})
    d["bin"] = pd.qcut(d["p"], bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(Samples=("y", "size"), Predicted=("p", "mean"), Actual=("y", "mean"))
    return g.reset_index(drop=True)


def signal_table(s: pd.DataFrame) -> pd.DataFrame:
    """AUC of each signal for 'will this stock reach the target?' plus the hit rate of its top / bottom 20%."""
    base = float(s["hit"].mean())
    rows = []
    for name, col, sign in [("Time-Machine probability", "p_sim", 1), ("Closed-form (volatility only)", "p_quick", 1),
                            ("Own past hit rate", "p_trail", 1), ("Volatility", "vol", 1), ("Technical score", "tech", 1),
                            ("3-month momentum", "mom63", 1), ("1-month momentum", "mom21", 1), ("Relative strength", "rs63", 1)]:
        x = s[col]
        m = x.notna()
        if m.sum() < 100 or s.loc[m, "hit"].nunique() < 2:
            continue
        hi_cut, lo_cut = x[m].quantile(0.8), x[m].quantile(0.2)
        rows.append({"Signal": name, "AUC": roc_auc(s.loc[m, "hit"], x[m] * sign), "Top 20% hit": float(s.loc[m & (x >= hi_cut), "hit"].mean()),
                     "Bottom 20% hit": float(s.loc[m & (x <= lo_cut), "hit"].mean()), "Base rate": base, "Samples": int(m.sum())})
    return pd.DataFrame(rows)


def summary(s: pd.DataFrame) -> dict:
    y = s["hit"]
    out = {"n": len(s), "stocks": s["ticker"].nunique(), "base": float(y.mean()), "auc_sim": roc_auc(y, s["p_sim"]),
           "brier_sim": brier(s["p_sim"], y), "brier_base": brier(pd.Series(y.mean(), index=y.index), y),
           "mean_pred": float(s["p_sim"].mean()), "from": s["date"].min(), "to": s["date"].max()}
    top = s[s["p_sim"] >= s["p_sim"].quantile(0.8)]
    out["ret_all"], out["ret_top"] = float(s["ret_trade"].mean()), float(top["ret_trade"].mean())
    out["hit_top"], out["n_top"] = float(top["hit"].mean()), len(top)
    out["ret_t"] = float(s["ret_trade"].mean() / s["ret_trade"].std() * np.sqrt(len(s))) if s["ret_trade"].std() > 0 else np.nan
    out["brier_trail"] = brier(s["p_trail"], y) if s["p_trail"].notna().sum() > 50 else np.nan
    return out
