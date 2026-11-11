"""Scenario analysis: "if it goes well the level is about X, if it goes badly about Y".

Two kinds of scenario, both built only from what actually happened before:

* **Event branches** - for a type of event (election result, RBI decision, Budget, Fed...), take every past occurrence,
  sort the market's reactions, and report the favourable branch (typical of the best third), the base case (the middle)
  and the unfavourable branch (typical of the worst third) as price levels from today. It cannot say *which* branch will
  happen: nobody knows an election or policy outcome in advance. It says what each branch has been worth.
* **What-if shocks** - a regression of the stock's weekly returns on market, US equities, crude, the rupee and gold gives
  the move to expect if one of them jumps ("crude +15%"), with an honest error band from what the factors do not explain.

Each has a walk-forward check: did earlier branches contain what happened next, and how much of the stock's movement
did the factor model explain on weeks it had not seen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = {"Event day": 1, "1 week later": 5, "1 month later": 21}
FACTORS = {"^NSEI": "NIFTY 50", "^GSPC": "S&P 500 (US)", "BZ=F": "Brent crude", "USDINR=X": "USD/INR", "GC=F": "Gold"}
PRESETS = {
    "Market crash (NIFTY −10%)": {"^NSEI": -0.10},
    "Market rally (NIFTY +5%)": {"^NSEI": 0.05},
    "US sell-off (S&P −7%, NIFTY −4%)": {"^GSPC": -0.07, "^NSEI": -0.04},
    "Crude spike (+20%)": {"BZ=F": 0.20},
    "Crude slump (−15%)": {"BZ=F": -0.15},
    "Rupee slides 3% (USD/INR +3%)": {"USDINR=X": 0.03},
    "Flight to safety (gold +8%, NIFTY −5%)": {"GC=F": 0.08, "^NSEI": -0.05},
}


def _positions(index: pd.DatetimeIndex, dates, offset: int) -> list[int]:
    out = []
    for d in pd.to_datetime(list(dates)):
        p = int(index.searchsorted(d))
        p += offset
        if 1 <= p < len(index) and (index[p] - d).days <= 5 + offset * 3:
            out.append(p)
    return sorted(set(out))


def reactions(close: pd.Series, dates, offset: int = 0) -> pd.DataFrame:
    """Per past event: move from the close before the event to the event day, 1 week and 1 month later."""
    close = close.dropna()
    rows = []
    for p in _positions(close.index, dates, offset):
        base = float(close.iloc[p - 1])
        row = {"Date": close.index[p]}
        for name, h in HORIZONS.items():
            q = p + h - 1
            row[name] = float(close.iloc[q] / base - 1) if q < len(close) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("Date") if rows else pd.DataFrame(columns=list(HORIZONS))


def _branch_values(x: np.ndarray) -> tuple[float, float, float]:
    """(unfavourable, base, favourable): medians of the worst third, all, and the best third."""
    x = np.sort(x[np.isfinite(x)])
    k = max(1, len(x) // 3)
    return float(np.median(x[:k])), float(np.median(x)), float(np.median(x[-k:]))


def branches(close: pd.Series, dates, offset: int = 0) -> dict | None:
    """Scenario levels from today for each horizon. None if fewer than 3 past events are in the price history."""
    R = reactions(close, dates, offset)
    R = R[R.index < close.dropna().index[-1]]  # an event on the very last bar has no "after" yet
    last = float(close.dropna().iloc[-1])
    out = {"n": int(R["Event day"].notna().sum()), "last": last, "rows": [], "history": R}
    if out["n"] < 3:
        return None
    for name in HORIZONS:
        x = R[name].dropna().values
        if len(x) < 3:
            continue
        lo, mid, hi = _branch_values(x)
        out["rows"].append({"When": name, "n": int(len(x)), "Unfavourable": lo, "Base": mid, "Favourable": hi,
                            "Unfav ₹": last * (1 + lo), "Base ₹": last * (1 + mid), "Fav ₹": last * (1 + hi),
                            "Worst": float(x.min()), "Best": float(x.max()), "Up %": float((x > 0).mean())})
    return out


def branch_backtest(close: pd.Series, dates, offset: int = 0, min_prior: int = 4) -> dict | None:
    """Walk forward through the events: using only earlier ones, did [unfavourable, favourable] contain what came next?"""
    R = reactions(close, dates, offset)
    res = {}
    for name in HORIZONS:
        x = R[name].dropna().values
        inside = wide = n = 0
        for k in range(min_prior, len(x)):
            lo, _, hi = _branch_values(x[:k])
            n += 1
            inside += lo <= x[k] <= hi
            wide += x[:k].min() <= x[k] <= x[:k].max()
        if n >= 3:
            res[name] = {"n": n, "inside": inside / n, "inside_extremes": wide / n}
    return res or None


def event_day_size(close: pd.Series, dates, offset: int = 0) -> dict | None:
    """Are event days bigger than ordinary days? Ratio of average absolute move."""
    r = close.dropna().pct_change()
    pos = _positions(close.dropna().index, dates, offset)
    if len(pos) < 3:
        return None
    ev = r.iloc[pos].abs()
    other = r.drop(r.index[pos]).abs()
    return {"event": float(ev.mean()), "normal": float(other.mean()), "ratio": float(ev.mean() / other.mean())}


# --------------------------------------------------------------- what-if shocks


def weekly_returns(closes: pd.DataFrame) -> pd.DataFrame:
    c = closes.ffill(limit=4)
    return c.iloc[::5].pct_change().dropna(how="all")


def fit_factors(stock: pd.Series, factors: pd.DataFrame, window_weeks: int = 156) -> dict | None:
    """OLS of the stock's weekly return on factor weekly returns (last ~3 years). Returns betas, residual sd, R²."""
    df = weekly_returns(pd.concat([stock.rename("y"), factors], axis=1)).dropna().iloc[-window_weeks:]
    cols = [c for c in factors.columns if c in df and df[c].std() > 0]
    if len(df) < 60 or not cols:
        return None
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
    y = df["y"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else 0.0
    # out-of-sample check: fit on the first 70%, explain the rest
    k = int(len(df) * 0.7)
    b2, *_ = np.linalg.lstsq(X[:k], y[:k], rcond=None)
    pred = X[k:] @ b2
    oos = 1 - ((y[k:] - pred) ** 2).sum() / ((y[k:] - y[:k].mean()) ** 2).sum()
    corr = float(np.corrcoef(pred, y[k:])[0, 1]) if pred.std() > 0 else np.nan
    return {"betas": dict(zip(cols, beta[1:])), "alpha": float(beta[0]), "resid_sd": float(resid.std(ddof=len(cols) + 1)),
            "r2": float(r2), "oos_r2": float(oos), "oos_corr": corr, "weeks": len(df), "oos_weeks": len(df) - k}


def what_if(fit: dict, shocks: dict[str, float]) -> dict:
    """Expected move for the given factor shocks, with a ±1 residual-sd band (what the factors do not explain)."""
    move = sum(fit["betas"].get(k, 0.0) * v for k, v in shocks.items())
    return {"move": float(move), "low": float(move - fit["resid_sd"]), "high": float(move + fit["resid_sd"])}
