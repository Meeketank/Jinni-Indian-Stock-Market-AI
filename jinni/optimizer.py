"""Portfolio construction.

Objectives: max Sharpe, min variance, equal risk contribution (risk parity),
max diversification, and target-volatility max return. Constraints: long-only,
per-name min/max weight, per-sector cap. Covariance uses Ledoit-Wolf
shrinkage; expected returns can be shrunk historical means or CAPM-implied
equilibrium returns (less noisy than raw historical means).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .risk import ledoit_wolf_cov
from .universe import DEFAULT_RISK_FREE, TRADING_DAYS

OBJECTIVES = {
    "max_sharpe": "Maximum Sharpe ratio",
    "min_vol": "Minimum volatility",
    "risk_parity": "Risk parity (equal risk contribution)",
    "max_div": "Maximum diversification",
    "equal": "Equal weight (benchmark)",
}


def expected_returns(rets: pd.DataFrame, bench: pd.Series | None = None, method: str = "shrunk",
                     rf: float = DEFAULT_RISK_FREE, market_premium: float = 0.07,
                     shrink: float = 0.5) -> pd.Series:
    """Annualised expected returns.

    * ``historical`` – geometric mean of the sample (noisy, prone to chasing).
    * ``shrunk``     – historical shrunk 50% toward the cross-sectional mean.
    * ``capm``       – rf + beta × equity risk premium (India ERP ≈ 7%).
    """
    clean = rets.dropna()
    hist = (1 + clean).prod() ** (TRADING_DAYS / len(clean)) - 1
    if method == "historical":
        return hist
    if method == "capm" and bench is not None:
        b = bench.reindex(clean.index).dropna()
        x = clean.loc[b.index]
        betas = x.apply(lambda s: np.cov(s, b)[0, 1] / b.var())
        return rf + betas * market_premium
    return shrink * hist.mean() + (1 - shrink) * hist


@dataclass
class OptResult:
    weights: pd.Series
    exp_return: float
    volatility: float
    sharpe: float
    objective: str


def _constraints(tickers, sectors: dict | None, sector_cap: float | None):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if sectors and sector_cap and sector_cap < 1:
        groups: dict[str, list[int]] = {}
        for i, t in enumerate(tickers):
            groups.setdefault(sectors.get(t, "Other"), []).append(i)
        for idx in groups.values():
            idx_arr = np.array(idx)
            cons.append({"type": "ineq", "fun": lambda w, ix=idx_arr: sector_cap - w[ix].sum()})
    return cons


def optimize(rets: pd.DataFrame, objective: str = "max_sharpe", mu: pd.Series | None = None,
             cov: pd.DataFrame | None = None, rf: float = DEFAULT_RISK_FREE, min_w: float = 0.0,
             max_w: float = 0.25, sectors: dict | None = None, sector_cap: float | None = None) -> OptResult:
    tickers = list(rets.columns)
    n = len(tickers)
    if n == 0:
        raise ValueError("No assets to optimise")
    cov = cov if cov is not None else ledoit_wolf_cov(rets)
    C = cov.loc[tickers, tickers].values * TRADING_DAYS
    m = (mu if mu is not None else expected_returns(rets)).reindex(tickers).fillna(0).values
    max_w = max(max_w, 1.0 / n + 1e-9)  # keep the problem feasible
    min_w = min(min_w, 1.0 / n)
    bounds = [(min_w, max_w)] * n
    w0 = np.full(n, 1.0 / n)
    cons = _constraints(tickers, sectors, sector_cap)
    sd = np.sqrt(np.diag(C))

    def vol(w):
        return np.sqrt(max(w @ C @ w, 1e-16))

    if objective == "equal":
        w = w0
    else:
        if objective == "max_sharpe":
            f = lambda w: -(w @ m - rf) / vol(w)  # noqa: E731
        elif objective == "min_vol":
            f = lambda w: w @ C @ w * 1e4  # noqa: E731
        elif objective == "risk_parity":
            def f(w):
                rc = w * (C @ w) / vol(w)
                return ((rc - vol(w) / n) ** 2).sum() * 1e4
        elif objective == "max_div":
            f = lambda w: -(w @ sd) / vol(w)  # noqa: E731
        else:
            raise ValueError(f"Unknown objective {objective}")
        res = minimize(f, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 1000, "ftol": 1e-12})
        w = res.x if res.success or np.isfinite(res.fun) else w0
    w = np.clip(w, 0, None)
    w[w < 1e-4] = 0
    w = w / w.sum()
    ws = pd.Series(w, index=tickers)
    er, v = float(w @ m), float(vol(w))
    return OptResult(ws, er, v, (er - rf) / v if v > 0 else np.nan, objective)


def efficient_frontier(rets: pd.DataFrame, mu: pd.Series, cov: pd.DataFrame | None = None,
                       n_points: int = 30, max_w: float = 1.0, rf: float = DEFAULT_RISK_FREE) -> pd.DataFrame:
    tickers = list(rets.columns)
    cov = cov if cov is not None else ledoit_wolf_cov(rets)
    C = cov.loc[tickers, tickers].values * TRADING_DAYS
    m = mu.reindex(tickers).values
    n = len(tickers)
    max_w = max(max_w, 1.0 / n + 1e-9)
    bounds = [(0, max_w)] * n
    lo = optimize(rets, "min_vol", mu, cov, rf, max_w=max_w)
    # highest achievable return under the cap: fill best assets up to the cap
    order = np.argsort(-m)
    w_hi, left = np.zeros(n), 1.0
    for i in order:
        take = min(max_w, left)
        w_hi[i] = take
        left -= take
        if left <= 1e-12:
            break
    r_hi = float(w_hi @ m)
    pts = []
    for target in np.linspace(lo.exp_return, r_hi, n_points):
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w, t=target: w @ m - t}]
        res = minimize(lambda w: w @ C @ w, np.full(n, 1 / n), method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 500})
        if res.success:
            v = float(np.sqrt(res.x @ C @ res.x))
            pts.append({"Return": target, "Volatility": v, "Sharpe": (target - rf) / v})
    return pd.DataFrame(pts)


def discrete_allocation(weights: pd.Series, prices: pd.Series, capital: float) -> tuple[pd.DataFrame, float]:
    """Whole-share allocation: floor each position, then spend leftover cash
    greedily on the names furthest below their target weight."""
    w = weights[weights > 0]
    px = prices.reindex(w.index)
    shares = np.floor(w * capital / px).astype(int)
    cash = capital - float((shares * px).sum())
    for _ in range(10_000):
        current = shares * px / capital
        gap = (w - current).sort_values(ascending=False)
        bought = False
        for t in gap.index:
            if px[t] <= cash and gap[t] > 0:
                shares[t] += 1
                cash -= px[t]
                bought = True
                break
        if not bought:
            break
    df = pd.DataFrame({"Shares": shares, "Price ₹": px, "Value ₹": shares * px})
    df["Actual weight"] = df["Value ₹"] / capital
    df["Target weight"] = w
    return df.sort_values("Value ₹", ascending=False), cash


def rebalance_trades(current_shares: pd.Series, target: pd.DataFrame, cost_bps: float = 15.0) -> pd.DataFrame:
    """Trade list to move from current holdings to the target allocation.
    ``cost_bps`` approximates brokerage + STT + exchange + stamp + GST + slippage."""
    tickers = sorted(set(current_shares.index) | set(target.index))
    cur = current_shares.reindex(tickers).fillna(0)
    tgt = target["Shares"].reindex(tickers).fillna(0)
    px = target["Price ₹"].reindex(tickers)
    delta = (tgt - cur).astype(int)
    trades = pd.DataFrame({"Current": cur.astype(int), "Target": tgt.astype(int), "Trade": delta, "Price ₹": px})
    trades["Side"] = np.where(delta > 0, "BUY", np.where(delta < 0, "SELL", "—"))
    trades["Notional ₹"] = (delta.abs() * px).fillna(0)
    trades["Est. cost ₹"] = trades["Notional ₹"] * cost_bps / 1e4
    return trades[trades["Trade"] != 0]
