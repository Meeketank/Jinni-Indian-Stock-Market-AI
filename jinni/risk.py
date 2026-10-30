"""Portfolio risk analytics.

Conventions
-----------
* Returns are simple daily returns; annualisation uses 252 trading days.
* Portfolio returns use *drifting* buy-and-hold weights only where noted;
  otherwise daily-rebalanced constant weights (standard for risk models).
* VaR / CVaR are reported as positive loss numbers (fraction of portfolio).
* Multi-day VaR is computed from overlapping h-day returns (historical) or
  simulated paths (Monte Carlo), never with the sqrt-of-time shortcut on a
  non-normal distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .universe import DEFAULT_RISK_FREE, TRADING_DAYS


# ------------------------------------------------------------------ basics


def returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return prices.pct_change(fill_method=None).dropna(how="all")


def portfolio_returns(rets: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(rets.columns).fillna(0.0)
    # If an asset has no data on a day, renormalise across those that do.
    avail = rets.notna().astype(float) * w
    scale = avail.sum(axis=1).replace(0, np.nan)
    return (rets.fillna(0.0) * w).sum(axis=1) / scale * w.sum()


def ledoit_wolf_cov(rets: pd.DataFrame) -> pd.DataFrame:
    """Shrinkage covariance (Ledoit-Wolf), daily units. Robust with few observations."""
    from .mlcore import ledoit_wolf_cov as _lw

    clean = rets.dropna()
    return pd.DataFrame(_lw(clean.values), index=rets.columns, columns=rets.columns)


def ewma_cov(rets: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """RiskMetrics EWMA covariance (daily). Reacts faster to regime changes."""
    x = rets.dropna().values
    x = x - x.mean(axis=0)
    w = lam ** np.arange(len(x) - 1, -1, -1)
    w /= w.sum()
    cov = (x * w[:, None]).T @ x
    return pd.DataFrame(cov, index=rets.columns, columns=rets.columns)


# ------------------------------------------------------------ performance


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1 + r.fillna(0)).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r: pd.Series) -> dict:
    dd = drawdown_series(r)
    if dd.empty:
        return {"max_drawdown": 0.0, "peak": None, "trough": None, "recovery": None, "days_underwater": 0}
    trough = dd.idxmin()
    wealth = (1 + r.fillna(0)).cumprod()
    peak = wealth.loc[:trough].idxmax()
    after = wealth.loc[trough:]
    rec = after[after >= wealth.loc[peak]]
    recovery = rec.index[0] if len(rec) else None
    end = recovery or dd.index[-1]
    return {
        "max_drawdown": float(dd.min()),
        "peak": peak,
        "trough": trough,
        "recovery": recovery,
        "days_underwater": int(len(dd.loc[peak:end]) - 1),
    }


def performance_stats(r: pd.Series, bench: pd.Series | None = None, rf: float = DEFAULT_RISK_FREE) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 2:
        return {}
    years = n / TRADING_DAYS
    total = float((1 + r).prod() - 1)
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    rf_d = (1 + rf) ** (1 / TRADING_DAYS) - 1
    ex = r - rf_d
    sharpe = float(ex.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = float(ex.mean() * TRADING_DAYS / downside) if downside > 0 else np.nan
    mdd = max_drawdown(r)["max_drawdown"]
    out = {
        "Total return": total,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "Skew": float(r.skew()),
        "Excess kurtosis": float(r.kurt()),
        "Best day": float(r.max()),
        "Worst day": float(r.min()),
        "% positive days": float((r > 0).mean()),
    }
    if bench is not None:
        b = bench.reindex(r.index).dropna()
        rr = r.reindex(b.index)
        if len(b) > 20 and b.var() > 0:
            beta = float(np.cov(rr, b)[0, 1] / b.var())
            alpha_d = (rr - rf_d).mean() - beta * (b - rf_d).mean()
            active = rr - b
            te = float(active.std() * np.sqrt(TRADING_DAYS))
            up = b > 0
            out.update({
                "Beta": beta,
                "Alpha (ann.)": float(alpha_d * TRADING_DAYS),
                "Correlation": float(rr.corr(b)),
                "R²": float(rr.corr(b) ** 2),
                "Tracking error": te,
                "Information ratio": float(active.mean() * TRADING_DAYS / te) if te > 0 else np.nan,
                "Up capture": float(rr[up].mean() / b[up].mean()) if b[up].mean() else np.nan,
                "Down capture": float(rr[~up].mean() / b[~up].mean()) if b[~up].mean() else np.nan,
            })
    return out


def rolling_beta(r: pd.Series, bench: pd.Series, window: int = 63) -> pd.Series:
    df = pd.concat([r, bench], axis=1).dropna()
    cov = df.iloc[:, 0].rolling(window).cov(df.iloc[:, 1])
    return (cov / df.iloc[:, 1].rolling(window).var()).dropna()


# --------------------------------------------------------------- VaR/CVaR


def _h_day(r: pd.Series, h: int) -> pd.Series:
    """Overlapping compounded h-day returns."""
    if h == 1:
        return r.dropna()
    return (np.log1p(r.dropna()).rolling(h).sum().dropna()).pipe(np.expm1)


def historical_var(r: pd.Series, conf: float = 0.95, h: int = 1) -> tuple[float, float]:
    x = _h_day(r, h)
    q = np.quantile(x, 1 - conf)
    tail = x[x <= q]
    return float(-q), float(-tail.mean()) if len(tail) else float(-q)


def parametric_var(r: pd.Series, conf: float = 0.95, h: int = 1, cornish_fisher: bool = True) -> tuple[float, float]:
    """Gaussian VaR, optionally with Cornish-Fisher skew/kurtosis adjustment."""
    x = r.dropna()
    mu, sd = x.mean() * h, x.std() * np.sqrt(h)
    z = stats.norm.ppf(1 - conf)
    if cornish_fisher:
        s, k = stats.skew(x), stats.kurtosis(x)  # excess kurtosis
        z = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    var = -(mu + z * sd)
    # Gaussian ES; CF-ES approximated by scaling with the same ratio
    z0 = stats.norm.ppf(1 - conf)
    es_gauss = -(mu - sd * stats.norm.pdf(z0) / (1 - conf))
    es = es_gauss * (var / -(mu + z0 * sd)) if cornish_fisher and (mu + z0 * sd) != 0 else es_gauss
    return float(var), float(max(es, var))


def monte_carlo_paths(
    rets: pd.DataFrame,
    weights: pd.Series,
    horizon: int = 10,
    n_sims: int = 10_000,
    method: str = "student-t",
    dof: float = 5.0,
    seed: int = 7,
) -> np.ndarray:
    """Simulate portfolio cumulative returns over ``horizon`` days.

    method="student-t": multivariate Student-t with Ledoit-Wolf covariance
    (captures fat tails & co-movement). method="bootstrap": resample whole
    historical days (keeps empirical joint distribution)."""
    clean = rets.dropna()
    w = weights.reindex(clean.columns).fillna(0).values
    rng = np.random.default_rng(seed)
    if method == "bootstrap":
        idx = rng.integers(0, len(clean), size=(n_sims, horizon))
        port = clean.values[idx] @ w  # (n_sims, horizon)
    else:
        mu = clean.mean().values
        cov = ledoit_wolf_cov(clean).values
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(cov)))
        z = rng.standard_normal((n_sims, horizon, len(mu)))
        chi = rng.chisquare(dof, size=(n_sims, horizon, 1)) / dof
        t = z / np.sqrt(chi) * np.sqrt((dof - 2) / dof)  # unit variance
        asset = mu + t @ L.T
        port = asset @ w
    return np.cumprod(1 + port, axis=1) - 1


def monte_carlo_var(paths: np.ndarray, conf: float = 0.95) -> tuple[float, float]:
    final = paths[:, -1]
    q = np.quantile(final, 1 - conf)
    return float(-q), float(-final[final <= q].mean())


@dataclass
class VaRReport:
    table: pd.DataFrame
    paths: np.ndarray


def var_report(rets: pd.DataFrame, weights: pd.Series, value: float, conf: float = 0.95, h: int = 1,
               n_sims: int = 10_000) -> VaRReport:
    pr = portfolio_returns(rets, weights)
    hv, hes = historical_var(pr, conf, h)
    pv, pes = parametric_var(pr, conf, h, cornish_fisher=False)
    cv, ces = parametric_var(pr, conf, h, cornish_fisher=True)
    paths = monte_carlo_paths(rets, weights, horizon=max(h, 1), n_sims=n_sims)
    mv, mes = monte_carlo_var(paths, conf)
    rows = [
        ("Historical simulation", hv, hes),
        ("Parametric (Gaussian)", pv, pes),
        ("Parametric (Cornish-Fisher)", cv, ces),
        ("Monte Carlo (Student-t, LW cov)", mv, mes),
    ]
    tbl = pd.DataFrame(rows, columns=["Method", "VaR %", "CVaR %"]).set_index("Method")
    tbl["VaR ₹"] = tbl["VaR %"] * value
    tbl["CVaR ₹"] = tbl["CVaR %"] * value
    return VaRReport(tbl, paths)


def var_backtest(r: pd.Series, conf: float = 0.95, window: int = 250) -> dict:
    """Rolling 1-day historical VaR backtest with Kupiec proportion-of-failures test."""
    r = r.dropna()
    var = -r.rolling(window).quantile(1 - conf).shift(1)
    df = pd.DataFrame({"r": r, "var": var}).dropna()
    if df.empty:
        return {}
    breaches = (df["r"] < -df["var"])
    n, x = len(df), int(breaches.sum())
    p = 1 - conf
    phat = x / n if n else 0
    if 0 < x < n:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - phat) - x * np.log(phat))
    else:
        lr = -2 * (n * np.log(1 - p)) if x == 0 else np.inf
    pval = float(1 - stats.chi2.cdf(lr, 1))
    return {
        "observations": n,
        "breaches": x,
        "expected": n * p,
        "breach_rate": phat,
        "kupiec_p": pval,
        "model_ok": pval > 0.05,
        "series": df.assign(breach=breaches),
    }


# ------------------------------------------------------ risk decomposition


def risk_contributions(cov: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Euler decomposition of portfolio volatility (annualised)."""
    w = weights.reindex(cov.index).fillna(0).values
    C = cov.values * TRADING_DAYS
    port_vol = float(np.sqrt(w @ C @ w))
    mrc = C @ w / port_vol if port_vol > 0 else np.zeros_like(w)
    crc = w * mrc
    return pd.DataFrame({
        "Weight": w,
        "Asset vol": np.sqrt(np.diag(C)),
        "Marginal contrib.": mrc,
        "Risk contrib.": crc,
        "% of risk": crc / port_vol if port_vol > 0 else 0,
    }, index=cov.index)


def diversification_ratio(cov: pd.DataFrame, weights: pd.Series) -> float:
    w = weights.reindex(cov.index).fillna(0).values
    sd = np.sqrt(np.diag(cov.values))
    pv = np.sqrt(w @ cov.values @ w)
    return float((w @ sd) / pv) if pv > 0 else np.nan


def effective_n(weights: pd.Series) -> float:
    w = weights[weights > 0]
    return float(1 / (w**2).sum()) if len(w) else 0.0


def factor_exposures(rets: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    """Per-asset market beta, idiosyncratic vol and correlation to benchmark."""
    rows = {}
    b = bench.dropna()
    for t in rets.columns:
        x = pd.concat([rets[t], b], axis=1).dropna()
        if len(x) < 30:
            continue
        beta = np.cov(x.iloc[:, 0], x.iloc[:, 1])[0, 1] / x.iloc[:, 1].var()
        resid = x.iloc[:, 0] - beta * x.iloc[:, 1]
        rows[t] = {
            "Beta": beta,
            "Correlation": x.iloc[:, 0].corr(x.iloc[:, 1]),
            "Total vol": x.iloc[:, 0].std() * np.sqrt(TRADING_DAYS),
            "Idiosyncratic vol": resid.std() * np.sqrt(TRADING_DAYS),
            "Systematic % of var": 1 - resid.var() / x.iloc[:, 0].var(),
        }
    return pd.DataFrame(rows).T


# -------------------------------------------------------------- stress tests


def historical_stress(prices: pd.DataFrame, weights: pd.Series, scenarios: dict[str, tuple[str, str]],
                      bench: pd.Series | None = None) -> pd.DataFrame:
    """Replay historical windows on today's weights (buy-and-hold over the window)."""
    rows = []
    for name, (start, end) in scenarios.items():
        win = prices.loc[start:end]
        if len(win) < 2:
            rows.append({"Scenario": name, "Portfolio %": np.nan, "NIFTY %": np.nan, "Coverage %": 0.0})
            continue
        asset_ret = win.ffill().iloc[-1] / win.bfill().iloc[0] - 1
        w = weights.reindex(prices.columns).fillna(0)
        covered = asset_ret.notna() & (w > 0)
        cov_w = w[covered].sum()
        port = float((asset_ret[covered] * w[covered]).sum() / cov_w) if cov_w > 0 else np.nan
        bret = np.nan
        if bench is not None:
            bw = bench.loc[start:end].dropna()
            if len(bw) > 1:
                bret = float(bw.iloc[-1] / bw.iloc[0] - 1)
        rows.append({"Scenario": name, "Portfolio %": port * 100, "NIFTY %": bret * 100,
                     "Coverage %": cov_w / w.sum() * 100 if w.sum() else 0})
    return pd.DataFrame(rows).set_index("Scenario")


def factor_shock(betas: pd.Series, weights: pd.Series, shock: float) -> float:
    """Portfolio move implied by a market shock via each holding's beta."""
    w = weights.reindex(betas.index).fillna(0)
    return float((w * betas).sum() / w.sum() * shock) if w.sum() else np.nan


def liquidity_profile(weights: pd.Series, value: float, prices: pd.DataFrame, volumes: pd.DataFrame,
                      participation: float = 0.2) -> pd.DataFrame:
    """Days to liquidate each position trading at most ``participation`` of 20-day ADV."""
    adv_value = (prices * volumes).iloc[-20:].mean()
    pos_value = weights * value
    days = pos_value / (adv_value.reindex(weights.index) * participation)
    return pd.DataFrame({
        "Position ₹": pos_value,
        "20d ADV ₹": adv_value.reindex(weights.index),
        "% of ADV": pos_value / adv_value.reindex(weights.index) * 100,
        "Days to exit": days,
    })
