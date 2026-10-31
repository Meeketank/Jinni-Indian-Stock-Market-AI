"""Systematic stock selection: factor signals, walk-forward validation, risk-aware portfolio construction.

The approach mirrors how institutional quant desks work, at a scale one person can run:

1. **Signals** - cross-sectional factors for every stock on every date (momentum, reversal, low volatility, trend,
   52-week-high proximity, relative strength), each z-scored across the universe.
2. **Weights learned without peeking** - at each rebalance the weight of a factor is its average rank-IC (correlation
   with the following period's return) over the *previous* 24 months only; factors that haven't worked get zero.
3. **Walk-forward backtest** - the composite is traded as a top-quintile basket, net of costs, against the equal-weight
   universe, so the reported edge is out-of-sample by construction.
4. **Construction** - the best names go through a risk model (Ledoit-Wolf covariance) with position and sector limits.

Limits that cannot be engineered away with free data: the universe is *today's* listed stocks (survivorship bias flatters
every backtest), and there is no point-in-time fundamentals history, so fundamentals only screen the final names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FACTORS = {
    "Momentum 12-1": "12-month return skipping the latest month",
    "Momentum 6-1": "6-month return skipping the latest month",
    "Momentum 3M": "3-month return",
    "Short-term reversal": "last month's loser tends to bounce (negative 1-month return)",
    "Low volatility": "calmer stocks (lower 60-day volatility)",
    "52-week-high proximity": "price close to its 52-week high",
    "Trend (vs 200-DMA)": "price above its 200-day average",
    "Relative strength": "6-month return minus NIFTY's",
}
COST_PER_SIDE = 0.0015       # 15 bps, matches the rebalance-cost assumption elsewhere
IC_WINDOW = 24               # rebalances of IC history used for weights (24 monthly = 2 years)
MIN_IC_OBS = 12
TOP_SHARE = 0.2


def raw_factors(close: pd.DataFrame, bench: pd.Series | None) -> dict[str, pd.DataFrame]:
    lr = np.log(close).diff()
    f = {
        "Momentum 12-1": close.shift(21) / close.shift(252) - 1,
        "Momentum 6-1": close.shift(21) / close.shift(126) - 1,
        "Momentum 3M": close / close.shift(63) - 1,
        "Short-term reversal": -(close / close.shift(21) - 1),
        "Low volatility": -lr.rolling(60, min_periods=40).std(),
        "52-week-high proximity": close / close.rolling(252, min_periods=200).max(),
        "Trend (vs 200-DMA)": close / close.rolling(200, min_periods=150).mean() - 1,
    }
    if bench is not None:
        b = bench.reindex(close.index).ffill()
        f["Relative strength"] = close.pct_change(126).sub(b.pct_change(126), axis=0)
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in f.items()}


def zscores(f: pd.DataFrame) -> pd.DataFrame:
    z = f.sub(f.mean(axis=1), axis=0).div(f.std(axis=1).replace(0, np.nan), axis=0)
    return z.clip(-3, 3)


def _rank_ic(a: pd.Series, b: pd.Series, min_n: int = 30) -> float:
    m = a.notna() & b.notna()
    if m.sum() < min_n:
        return np.nan
    return float(a[m].rank().corr(b[m].rank()))


def learn_weights(ic_hist: pd.DataFrame) -> pd.Series:
    """Factor weights from past IC only: positive mean IC (with enough observations), normalised to sum to 1."""
    h = ic_hist.dropna(how="all").iloc[-IC_WINDOW:]
    if len(h) < MIN_IC_OBS:
        return pd.Series(0.0, index=ic_hist.columns)
    w = h.mean().clip(lower=0)
    return w / w.sum() if w.sum() > 0 else pd.Series(0.0, index=ic_hist.columns)


@dataclass
class Validation:
    h: int
    ic: pd.DataFrame            # rebalance date x factor
    ic_comp: pd.Series          # composite IC per rebalance
    weights_now: pd.Series
    curve: pd.DataFrame         # cumulative net growth: Selection vs Universe
    stats: dict
    factor_table: pd.DataFrame
    z: dict = field(default_factory=dict)
    names_per_period: int = 0


def validate(close: pd.DataFrame, bench: pd.Series | None, h: int = 21, top_share: float = TOP_SHARE) -> Validation:
    raw = raw_factors(close, bench)
    z = {k: zscores(v) for k, v in raw.items()}
    fwd = close.shift(-h) / close - 1
    n = len(close)
    first = 252
    rebal = list(range(first, n - h, h))
    if len(rebal) < MIN_IC_OBS + 6:
        raise ValueError("Not enough price history to validate the selection model (need about 3 years).")
    names = list(z)
    ic = pd.DataFrame({k: [_rank_ic(z[k].iloc[i], fwd.iloc[i]) for i in rebal] for k in names},
                      index=close.index[rebal])
    rows, comp_ic, prev_picks = [], [], set()
    for j, i in enumerate(rebal):
        w = learn_weights(ic.iloc[:j])  # strictly earlier rebalances
        if w.sum() == 0:
            continue
        score = sum(w[k] * z[k].iloc[i].fillna(0) for k in names if w[k] > 0)
        valid = fwd.iloc[i].notna() & z[names[0]].iloc[i].notna()
        s, r = score[valid], fwd.iloc[i][valid]
        if len(s) < 30:
            continue
        k_top = max(10, int(len(s) * top_share))
        picks = set(s.nlargest(k_top).index)
        turnover = 1 - len(picks & prev_picks) / len(picks) if prev_picks else 1.0
        gross, uni = float(r[list(picks)].mean()), float(r.mean())
        net = gross - 2 * COST_PER_SIDE * turnover
        bottom = float(r[s.nsmallest(k_top).index].mean())
        rows.append({"date": close.index[i], "Selection": net, "Universe": uni, "Gross": gross, "Bottom": bottom})
        comp_ic.append((close.index[i], _rank_ic(score[valid], r)))
        prev_picks = picks
    if len(rows) < 6:
        raise ValueError("The validation produced too few out-of-sample periods.")
    bt = pd.DataFrame(rows).set_index("date")
    ppy = 252 / h
    sh = lambda x: float(x.mean() / x.std() * np.sqrt(ppy)) if x.std() > 0 else np.nan  # noqa: E731
    exc = bt["Selection"] - bt["Universe"]
    cic = pd.Series(dict(comp_ic))
    stats = {
        "periods": len(bt), "from": bt.index[0], "to": bt.index[-1],
        "ret_sel": float((1 + bt["Selection"]).prod() ** (ppy / len(bt)) - 1),
        "ret_uni": float((1 + bt["Universe"]).prod() ** (ppy / len(bt)) - 1),
        "sharpe_sel": sh(bt["Selection"]), "sharpe_uni": sh(bt["Universe"]),
        "hit": float((exc > 0).mean()), "excess_t": float(exc.mean() / exc.std() * np.sqrt(len(exc))) if exc.std() > 0 else np.nan,
        "ic": float(cic.mean()), "ic_t": float(cic.mean() / cic.std() * np.sqrt(len(cic))) if cic.std() > 0 else np.nan,
        "spread": float((bt["Gross"] - bt["Bottom"]).mean()), "worst": float(bt["Selection"].min()),
        "cost_drag": float((bt["Gross"] - bt["Selection"]).mean() * ppy),
    }
    curve = (1 + bt[["Selection", "Universe"]]).cumprod()
    wn = learn_weights(ic)
    ft = pd.DataFrame({"Factor": names, "What it measures": [FACTORS[k] for k in names],
                       "Mean IC": [float(ic[k].mean()) for k in names],
                       "t-stat": [float(ic[k].mean() / ic[k].std() * np.sqrt(ic[k].notna().sum())) if ic[k].std() > 0 else np.nan for k in names],
                       "Worked %": [float((ic[k].dropna() > 0).mean()) for k in names],
                       "Weight now": [float(wn[k]) for k in names]})
    return Validation(h, ic, cic, wn, curve, stats, ft, z, int(max(10, len(close.columns) * top_share)))


def live_scores(v: Validation, close: pd.DataFrame) -> pd.DataFrame:
    """Composite score and per-factor contributions for the latest date."""
    last = close.index[-1]
    w = v.weights_now
    if w.sum() == 0:
        w = pd.Series(1.0 / len(w), index=w.index)  # no factor has proven itself: equal weight, flagged by the caller
    contrib = pd.DataFrame({k: w[k] * v.z[k].loc[last].fillna(0) for k in w.index if w[k] > 0})
    out = pd.DataFrame({"Score": contrib.sum(axis=1)})
    out["Percentile"] = out["Score"].rank(pct=True)
    return out.join(contrib.add_prefix("c_"))


def drivers_text(row: pd.Series, top: int = 3) -> str:
    c = row[[k for k in row.index if k.startswith("c_")]].astype(float)
    c.index = [k[2:] for k in c.index]
    best = c[c > 0.05].sort_values(ascending=False).head(top)
    return ", ".join(best.index) if len(best) else "no single standout factor"


def rating(pct_rank: float) -> str:
    return "STRONG BUY" if pct_rank >= 0.95 else "BUY" if pct_rank >= 0.85 else "HOLD" if pct_rank >= 0.4 else "AVOID"


def tilt_weights(score: pd.Series, max_w: float = 0.25, floor: float = 0.2) -> pd.Series:
    """Weights proportional to score (with a floor), capped per name and renormalised."""
    w = score.clip(lower=floor)
    w = w / w.sum()
    for _ in range(20):
        over = w > max_w
        if not over.any():
            break
        excess = (w[over] - max_w).sum()
        w[over] = max_w
        free = ~over
        w[free] += excess * w[free] / w[free].sum()
    return w / w.sum()
