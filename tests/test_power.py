"""Tests for the indicator library, pattern engine, sentiment, event lab and market scan."""

import os

import numpy as np
import pandas as pd
import pytest

os.environ["JINNI_DEMO"] = "1"

from jinni import data, forecast as fc, market, patterns as pt, seasonality as se, sentiment as sm, ta  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return data.get_prices(["RELIANCE.NS"], "5y").ohlcv("RELIANCE.NS")


def test_every_indicator_runs_and_aligns(df):
    assert len(ta.CATALOG) >= 80
    for name, i in ta.CATALOG.items():
        out = i.fn(df)
        assert len(out) == len(df), name
    tbl, v = ta.summary(df)
    assert set(v) == set(ta.CATEGORIES) | {"Overall"}
    assert v["Overall"]["buy"] + v["Overall"]["sell"] + v["Overall"]["neutral"] == tbl["Votes"].sum()


def test_classic_pivots():
    d = pd.DataFrame({"Open": [100], "High": [110], "Low": [90], "Close": [105], "Volume": [1]},
                     index=pd.bdate_range("2024-01-01", periods=1))
    p = ta.pivots_all(d)["Classic"]
    P = (110 + 90 + 105) / 3
    assert p["Pivot"] == pytest.approx(P) and p["R1"] == pytest.approx(2 * P - 90) and p["R2"] == pytest.approx(P + 20)


def test_supertrend_flips_with_trend():
    idx = pd.bdate_range("2024-01-01", periods=120)
    c = np.r_[np.linspace(100, 150, 60), np.linspace(150, 90, 60)]
    d = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1e6}, index=idx)
    st = ta.supertrend(d)
    assert st["ST_dir"].iloc[55] == 1 and st["ST_dir"].iloc[-1] == -1


def _bars(rows):
    idx = pd.bdate_range("2024-01-01", periods=len(rows))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx).assign(Volume=1e6)


def test_hammer_and_engulfing_detected():
    down = [(110 - i, 111 - i, 108.5 - i, 109 - i) for i in range(20)]  # steady decline
    hammer = [(89.5, 90.8, 85.0, 90.6)]  # long lower shadow, small real body at the top
    cf = pt.candlestick_frame(_bars(down + hammer))
    assert cf["Hammer"].iloc[-1]
    eng = [(89.0, 89.5, 87.0, 87.5), (87.0, 91.0, 86.8, 90.5)]
    cf2 = pt.candlestick_frame(_bars(down + eng))
    assert cf2["Bullish engulfing"].iloc[-1]


def test_zigzag_has_no_lookahead(df):
    pivs = pt.zigzag(df)
    assert pivs and all(p.confirmed >= p.i for p in pivs)
    kinds = [p.kind for p in pivs]
    assert all(a != b for a, b in zip(kinds, kinds[1:]))  # alternating highs and lows


def test_double_bottom_found():
    seg = lambda a, b, n: list(np.linspace(a, b, n))  # noqa: E731
    c = seg(130, 100, 25) + seg(100, 115, 15) + seg(115, 100.5, 15) + seg(100.5, 118, 15)
    d = _bars([(x, x * 1.004, x * 0.996, x) for x in c])
    names = [p.name for p in pt.detect_chart_patterns(d, pt.zigzag(d, 0.05))]
    assert "Double bottom" in names


def test_sentiment_signs():
    assert sm.score_headline("Shares surge after record profit, analysts upgrade") > 0.3
    assert sm.score_headline("Stock plunges on fraud probe, SEBI penalty") < -0.3
    assert abs(sm.score_headline("Company to hold board meeting on Tuesday")) < 0.1


def test_nth_weekday_and_expiry_dates():
    idx = pd.bdate_range("2024-01-01", "2025-12-31")
    second_fri = se.nth_weekday_dates(idx, 4, 2)
    assert pd.Timestamp("2024-01-12") in second_fri and pd.Timestamp("2025-03-14") in second_fri
    exp = se.monthly_expiry_dates(idx)
    assert pd.Timestamp("2024-01-25") in exp      # last Thursday (old regime)
    assert pd.Timestamp("2025-10-28") in exp      # last Tuesday (from Sep 2025)


def test_event_study_detects_planted_effect():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=2500)
    r = rng.normal(0, 0.01, len(idx))
    fridays = idx.weekday == 4
    r[fridays] += 0.006  # plant a +0.6% Friday effect
    close = pd.Series(100 * np.cumprod(1 + r), index=idx)
    res = se.event_study(close, idx[fridays], 0, 3, "Friday")
    assert res.summary["p_mean"] < 0.01 and res.summary["mean"] > 0.4
    flat = se.event_study(close, se.nth_weekday_dates(idx, 1, 2), 0, 3, "2nd Tue")
    assert flat.summary["p_mean"] > 0.01


def test_prefilter_excludes_unaffordable_and_illiquid():
    pdx = data.get_prices(["RELIANCE.NS", "MARUTI.NS", "ITC.NS"], "2y")
    pf = market.prefilter(pdx, budget=1000, target=0.05, horizon=5, min_value_cr=0.01)
    assert "budget" in pf.loc["MARUTI.NS", "Excluded because"]
    assert pf.loc["ITC.NS", "Excluded because"] == ""


def test_basket_simulation_and_allocation():
    pdx = data.get_prices(["ITC.NS", "HDFCBANK.NS"], "5y")
    dfs = {t: pdx.ohlcv(t) for t in pdx.tickers}
    c = pd.DataFrame({"ticker": list(dfs), "Price": [float(d["Close"].iloc[-1]) for d in dfs.values()]})
    al = fc.allocate_budget(c, 5000, 2)
    assert al["Invest ₹"].sum() + al.attrs["cash"] == pytest.approx(5000)
    b = fc.simulate_basket(dfs, dict(zip(al["ticker"], al["Shares"])), 10, 0.05, {t: 0.04 for t in dfs}, n_paths=2000)
    assert 0 <= b["p_reach"] <= 1 and b["p5"] <= b["p50"] <= b["p95"]
