"""Unit tests for the jinni engine. Run: pytest -q"""

import os

import numpy as np
import pandas as pd
import pytest

os.environ["JINNI_DEMO"] = "1"

from jinni import backtest, data, indicators as ind, ml, optimizer as opt, risk, signals  # noqa: E402
from jinni.universe import BENCHMARK, NIFTY50, normalize_symbol  # noqa: E402


@pytest.fixture(scope="module")
def market():
    tickers = list(NIFTY50)[:8]
    return tickers, data.get_prices(tickers + [BENCHMARK], "5y")


def test_normalize_symbol():
    assert normalize_symbol("tcs") == "TCS.NS"
    assert normalize_symbol("HDFC") == "HDFCBANK.NS"  # merged ticker
    assert normalize_symbol("reliance.bo") == "RELIANCE.BO"
    assert normalize_symbol("^NSEI") == "^NSEI"


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert ind.rsi(up).iloc[-1] == pytest.approx(100)
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert ind.rsi(down).iloc[-1] == pytest.approx(0)
    r = ind.rsi(pd.Series(np.random.default_rng(0).normal(0, 1, 500).cumsum() + 100)).dropna()
    assert r.between(0, 100).all()


def test_atr_constant_range():
    df = pd.DataFrame({"High": [11.0] * 50, "Low": [9.0] * 50, "Close": [10.0] * 50})
    assert ind.atr(df).iloc[-1] == pytest.approx(2.0)


def test_historical_var_matches_quantile():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 100_000))
    var, es = risk.historical_var(r, 0.95)
    assert var == pytest.approx(0.01645, rel=0.03)
    assert es == pytest.approx(0.02063, rel=0.03)  # normal ES at 95%
    pv, pes = risk.parametric_var(r, 0.95, cornish_fisher=False)
    assert pv == pytest.approx(var, rel=0.03)


def test_risk_contributions_sum_to_vol(market):
    tickers, pdx = market
    rets = risk.returns(pdx.close[tickers])
    w = pd.Series(np.linspace(1, 2, len(tickers)), index=tickers)
    w /= w.sum()
    cov = risk.ledoit_wolf_cov(rets)
    rc = risk.risk_contributions(cov, w)
    port_vol = np.sqrt(w.values @ cov.values @ w.values * 252)
    assert rc["Risk contrib."].sum() == pytest.approx(port_vol)
    assert rc["% of risk"].sum() == pytest.approx(1.0)


def test_max_drawdown_simple():
    r = pd.Series([0.1, -0.5, 0.2, 1.0])
    assert risk.max_drawdown(r)["max_drawdown"] == pytest.approx(-0.5)


def test_beta_of_benchmark_is_one(market):
    _, pdx = market
    b = risk.returns(pdx.close[BENCHMARK])
    assert risk.performance_stats(b, b)["Beta"] == pytest.approx(1.0)


@pytest.mark.parametrize("objective", list(opt.OBJECTIVES))
def test_optimizer_constraints(market, objective):
    tickers, pdx = market
    rets = risk.returns(pdx.close[tickers]).dropna()
    res = opt.optimize(rets, objective, max_w=0.25, sectors=NIFTY50, sector_cap=0.5)
    assert res.weights.sum() == pytest.approx(1.0)
    assert (res.weights >= -1e-9).all()
    if objective != "equal":
        assert res.weights.max() <= 0.25 + 1e-4
    sec = res.weights.groupby(pd.Series(NIFTY50).reindex(tickers)).sum()
    assert sec.max() <= 0.5 + 1e-4


def test_min_vol_beats_equal_weight(market):
    tickers, pdx = market
    rets = risk.returns(pdx.close[tickers]).dropna()
    assert opt.optimize(rets, "min_vol", max_w=1).volatility <= opt.optimize(rets, "equal").volatility + 1e-9


def test_discrete_allocation_within_budget(market):
    tickers, pdx = market
    w = pd.Series(1 / len(tickers), index=tickers)
    alloc, cash = opt.discrete_allocation(w, pdx.close[tickers].iloc[-1], 1_000_000)
    assert cash >= 0
    assert alloc["Value ₹"].sum() + cash == pytest.approx(1_000_000)


def test_trade_plan_geometry(market):
    tickers, pdx = market
    df = pdx.ohlcv(tickers[0])
    p = signals.trade_plan(df, "LONG", 1_000_000, 0.01)
    assert p.stop < p.entry < p.targets[0] <= p.targets[1] < p.targets[2]
    assert p.capital_at_risk <= 10_000 + 1e-6
    s = signals.trade_plan(df, "SHORT")
    assert s.stop > s.entry > s.targets[2]


def test_scores_in_range(market):
    tickers, pdx = market
    sc = signals.factor_scores(pdx.ohlcv(tickers[0]), pdx.close[BENCHMARK]).dropna()
    assert sc.min().min() >= 0 and sc.max().max() <= 100


def test_backtest_has_no_lookahead():
    """A strategy given tomorrow's sign would be perfect; ours must not be."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", periods=800)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800))), index=idx)
    df = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6})
    res = backtest.run(df, "trend", cost_bps=0)
    assert res["stats"]["Sharpe"] < 2  # random walk: no free lunch


def test_walk_forward_no_leak_on_random_data(market):
    """On synthetic random-walk data an honest validation must find ~no edge."""
    tickers, pdx = market
    res = ml.walk_forward({t: pdx.ohlcv(t) for t in tickers[:3]}, pdx.close[BENCHMARK], horizon=5, n_splits=3)
    assert abs(res.summary["auc"] - 0.5) < 0.08
    assert set(res.latest["prob_up"].between(0, 1)) == {True}
