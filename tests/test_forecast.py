"""Tests for the Return Finder engines, sizing/risk gate and setups."""

import os

import numpy as np
import pandas as pd
import pytest

os.environ["JINNI_DEMO"] = "1"

from jinni import data, forecast as fc, riskcheck as rc, setups  # noqa: E402
from jinni.universe import BENCHMARK, NIFTY50  # noqa: E402


def gbm(n=1500, vol=0.015, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, vol, n)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100, c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 3, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 3, n)))
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": c, "Volume": 1e6}, index=idx)


def test_touch_probability_matches_reflection_principle():
    """Zero-drift Brownian motion: P(max over T >= b) = 2*(1 - Phi(b / (sigma*sqrt(T))))."""
    from scipy.stats import norm

    df = gbm(vol=0.015)
    T, b = 21, 0.05
    sim = fc.simulate(df, T, b, None, n_paths=20000, annual_drift=0.0)
    sigma = np.log(df["Close"]).diff().std()
    exact = 2 * (1 - norm.cdf(np.log1p(b) / (sigma * np.sqrt(T))))
    # daily bars with intrabar highs touch slightly more often than the continuous close-only formula
    assert sim.p_touch == pytest.approx(exact, abs=0.08)


def test_simulation_agrees_with_history_on_random_walk():
    df = gbm(seed=3)
    mu = np.log(df["Close"]).diff().mean() * 252  # give the simulation the sample's own drift
    sim = fc.simulate(df, 21, 0.05, 0.04, n_paths=10000, annual_drift=mu)
    hist = fc.history_odds(df, 21, 0.05, 0.04)["all"]
    assert sim.p_target == pytest.approx(hist["p_target"], abs=0.07)
    assert sim.p_target + sim.p_stop + sim.p_neither == pytest.approx(1.0)


def test_replay_outcomes_hand_case():
    idx = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame({"Open": [100, 101, 103, 99, 98], "High": [100, 102, 106, 100, 99],
                       "Low": [100, 100, 102, 95, 97], "Close": [100, 101, 105, 96, 98], "Volume": 1}, index=idx)
    rep = fc.replay_outcomes(df, 3, 0.05, 0.04)
    assert rep.iloc[0]["outcome"] == 1 and rep.iloc[0]["days"] == 2  # hits 105 on day 2
    # from 101: day1 high 106 >= 106.05? no; day2 low 95 <= 96.96 -> stop
    rep2 = fc.replay_outcomes(df, 3, 0.0505, 0.04)
    assert rep2.iloc[1]["outcome"] == -1


def test_same_bar_conflict_counts_as_stop():
    idx = pd.bdate_range("2024-01-01", periods=3)
    df = pd.DataFrame({"Open": [100, 100, 100], "High": [100, 110, 100], "Low": [100, 90, 100],
                       "Close": [100, 100, 100], "Volume": 1}, index=idx)
    assert fc.replay_outcomes(df, 2, 0.05, 0.05).iloc[0]["outcome"] == -1


def test_verdict_breakeven():
    v, be, share = fc.verdict(0.30, 0.60, 0.10, 0.05)  # 2R target needs 33%; got 33%
    assert be == pytest.approx(1 / 3)
    assert "break-even" in v
    v2, _, _ = fc.verdict(0.6, 0.2, 0.05, 0.05)
    assert v2.startswith("Strongly toward the target")


def test_evaluate_tracked():
    idx = pd.bdate_range("2024-01-01", periods=6)
    df = pd.DataFrame({"Open": 100, "High": [100, 101, 103, 106, 100, 100], "Low": 99, "Close": 100,
                       "Volume": 1}, index=idx)
    ev = fc.evaluate_tracked(df, idx[0], 100, 105, 95, 5)
    assert ev["status"] == "Target hit" and ev["days"] == 3


def test_sizing_and_gate():
    r = dict(rc.DEFAULT_RULES)
    sz = rc.size_position(1000, 950, 1_000_000, 1_000_000, r)
    assert sz.qty == 100 and sz.deciding_limit == "Position cap"  # 10% of 10L = 100 shares; loss cap allows 200
    far = rc.size_position(1000, 1, 1_000_000, 1_000_000, {**r, "max_loss_per_trade": 500})
    assert far.qty == 0 and far.refused
    g = rc.gate(1000, 950, 1100, sz.qty, "IT", 0.5, 1_000_000, 1_000_000, {}, 0, r)
    assert g.status == "PASS"
    g2 = rc.gate(1000, 950, 1050, sz.qty, "IT", 0.5, 1_000_000, 1_000_000, {}, 0, r)  # R:R 1.0 < 1.5
    assert g2.status == "BLOCK"


def test_full_forecast_pipeline():
    tickers = list(NIFTY50)[:5]
    pdx = data.get_prices(tickers + [BENCHMARK], "5y")
    dfs = {t: pdx.ohlcv(t) for t in tickers}
    ml = fc.ml_hit_model(dfs, pdx.close[BENCHMARK], 10, 0.04, 0.03, n_splits=3)
    assert 0 <= ml.weight <= 1
    f = fc.forecast_stock(tickers[0], dfs[tickers[0]], pdx.close[BENCHMARK], 10, 0.04, 0.03, ml, n_paths=2000)
    assert 0 <= f.p_target <= 1 and f.p_target + f.p_stop <= 1 + 1e-9
    assert f.grade in "ABCD"
    sf = setups.setup_frame(dfs[tickers[0]])
    assert sf.dtypes.eq(bool).all()
