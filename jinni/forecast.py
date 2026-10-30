"""Return Finder: odds that a stock reaches +X% before -S% within N trading days.

Three independent engines, then an honest blend:

1. Time Machine (Monte Carlo). Rebuilds the next N days thousands of times
   from the stock's own daily bars (open/high/low/close as ratios to the
   previous close), sampled in 5-day blocks and rescaled from the volatility
   of the day they came from to today's volatility (filtered historical
   simulation). Each path is walked bar by bar: stop checked before target,
   a gap through either level fills at the open.
2. History replay. The same plan (entry at the close, same target, same stop,
   same horizon) applied to every past day: unconditional, and only on days
   that looked like today (same trend regime, similar JINNI score), shrunk
   toward the unconditional rate when the sample is small.
3. ML model. Gradient boosting trained on the replay outcomes across the
   whole universe, validated walk-forward with a purge gap, then calibrated
   (isotonic) on its own out-of-sample predictions. Its weight in the blend
   scales with its measured skill (AUC); no skill means zero weight.

Verdict logic follows the day-trader-skill Time Machine: a target S/X away
needs to come first in S/(X+S) of the paths that hit a side to break even.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from .signals import factor_scores
from .universe import TRADING_DAYS

COST_ROUND_TRIP = 0.003  # ~0.3%: STT both sides, charges, slippage (delivery)
# Long-run log drift for Indian large caps (~11%/yr). The simulation strips each
# stock's own past trend and uses this instead, so it does not chase winners.
MARKET_DRIFT = 0.11
HORIZONS = {"1 week": 5, "2 weeks": 10, "1 month": 21, "3 months": 63}


# ----------------------------------------------------------------- helpers


def atr_stop_pct(df: pd.DataFrame, mult: float = 2.0) -> float:
    a = float(ind.atr(df).iloc[-1] / df["Close"].iloc[-1])
    return float(np.clip(mult * a, 0.015, 0.15))


def auto_stop_pct(df: pd.DataFrame, horizon: int, target: float, min_rr: float = 1.5) -> float:
    """Volatility-aware stop: 1.5×ATR scaled by sqrt(horizon/5), no wider than
    target/min_rr (so reward:risk holds), never tighter than 1×ATR (noise)."""
    a = float(ind.atr(df).iloc[-1] / df["Close"].iloc[-1])
    s = 1.5 * a * np.sqrt(max(horizon, 5) / 5)
    s = min(s, target / min_rr)
    s = max(s, a)
    return float(np.clip(s, 0.01, 0.20))


def _ewma_vol(x: np.ndarray, lam: float = 0.94) -> np.ndarray:
    v = np.empty_like(x)
    s2 = np.nanvar(x[:60]) if len(x) > 60 else np.nanvar(x)
    for i, r in enumerate(x):
        v[i] = np.sqrt(s2)
        if np.isfinite(r):
            s2 = lam * s2 + (1 - lam) * r * r
    return v, np.sqrt(s2)


def bar_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Log ratios of each bar's O/H/L/C to the previous close."""
    pc = df["Close"].shift(1)
    out = pd.DataFrame({
        "o": np.log(df["Open"] / pc), "h": np.log(df["High"] / pc),
        "l": np.log(df["Low"] / pc), "c": np.log(df["Close"] / pc),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    # Repair bad bars so H >= max(O,C) and L <= min(O,C)
    out["h"] = out[["h", "o", "c"]].max(axis=1)
    out["l"] = out[["l", "o", "c"]].min(axis=1)
    return out


# ----------------------------------------------------------- Time Machine


@dataclass
class SimResult:
    p_target: float
    p_stop: float
    p_neither: float
    p_touch: float           # target touched at any point (ignoring the stop)
    p_green: float           # target first, or above entry at horizon
    ev: float                # mean realised return per path (before costs)
    mean_neither: float      # mean return of paths that hit neither
    median_days_to_target: float
    terminal_pct: dict
    fan: pd.DataFrame        # percentiles of price per day
    n_paths: int
    seed: int


def simulate(df: pd.DataFrame, horizon: int, target: float, stop: float | None, n_paths: int = 5000,
             seed: int = 19, block: int = 5, lookback: int = 1260, annual_drift: float = 0.0) -> SimResult:
    """Monte Carlo odds of hitting +target before -stop within horizon days.

    ``target`` and ``stop`` are fractions (0.05 = 5%). ``stop=None`` = no stop.
    Historical drift is removed and replaced by ``annual_drift`` so the
    simulation does not simply extrapolate the stock's past trend.
    """
    r = bar_ratios(df.iloc[-lookback - 1:])
    if len(r) < 120:
        raise ValueError("Need at least ~6 months of daily bars.")
    R = r[["o", "h", "l", "c"]].values.copy()
    vol_t, vol_now = _ewma_vol(R[:, 3])
    scale = (vol_now / np.maximum(vol_t, 1e-6))[:, None]
    mean_c = R[:, 3].mean()
    R = R - mean_c  # de-mean all four (shift whole bar)
    R = R * np.clip(scale, 0.33, 3.0)
    R = R + annual_drift / TRADING_DAYS

    rng = np.random.default_rng(seed)
    T = len(R)
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, T - block, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :horizon]
    bars = R[idx]  # (n, horizon, 4)
    close_log = np.cumsum(bars[:, :, 3], axis=1)
    prev = np.concatenate([np.zeros((n_paths, 1)), close_log[:, :-1]], axis=1)
    o, h, lo, c = prev + bars[:, :, 0], prev + bars[:, :, 1], prev + bars[:, :, 2], close_log

    lt = np.log1p(target)
    ls = np.log1p(-stop) if stop else -np.inf
    hit_t = h >= lt
    hit_s = lo <= ls
    # same bar: stop wins unless the bar opened above the target (gap up)
    both = hit_t & hit_s
    hit_t = hit_t & ~(both & (o < lt))
    hit_s = hit_s & ~(both & (o >= lt))
    big = horizon + 1
    ft = np.where(hit_t.any(1), hit_t.argmax(1), big)
    fs = np.where(hit_s.any(1), hit_s.argmax(1), big)
    tgt_first = ft < fs
    stp_first = fs < ft
    neither = ~(tgt_first | stp_first)

    rows = np.arange(n_paths)
    ret = np.expm1(c[:, -1])
    fill_t = np.expm1(np.maximum(o[rows, np.minimum(ft, horizon - 1)], lt))
    fill_s = np.expm1(np.minimum(o[rows, np.minimum(fs, horizon - 1)], ls)) if stop else ret
    realised = np.where(tgt_first, fill_t, np.where(stp_first, fill_s, ret))

    price0 = float(df["Close"].iloc[-1])
    q = [5, 25, 50, 75, 95]
    fan = pd.DataFrame(np.percentile(price0 * np.exp(c), q, axis=0).T, columns=[f"p{x}" for x in q])
    fan.index = np.arange(1, horizon + 1)
    term = np.percentile(np.expm1(c[:, -1]), q)
    return SimResult(
        p_target=float(tgt_first.mean()), p_stop=float(stp_first.mean()), p_neither=float(neither.mean()),
        p_touch=float((h >= lt).any(1).mean()),
        p_green=float((tgt_first | (neither & (ret > 0))).mean()),
        ev=float(realised.mean()),
        mean_neither=float(ret[neither].mean()) if neither.any() else 0.0,
        median_days_to_target=float(np.median(ft[tgt_first] + 1)) if tgt_first.any() else np.nan,
        terminal_pct=dict(zip(q, term)), fan=fan, n_paths=n_paths, seed=seed,
    )


# --------------------------------------------------------- history replay


def replay_outcomes(df: pd.DataFrame, horizon: int, target: float, stop: float | None) -> pd.DataFrame:
    """For every past day t: enter at close, outcome over the next ``horizon`` bars.

    Returns a frame indexed by entry date with columns: outcome (1 target,
    -1 stop, 0 neither), ret (realised), days (bars until exit)."""
    c = df["Close"].values
    o = df["Open"].values
    h = df["High"].values
    lo = df["Low"].values
    n = len(c)
    m = n - horizon
    if m <= 0:
        return pd.DataFrame(columns=["outcome", "ret", "days"])
    from numpy.lib.stride_tricks import sliding_window_view as sw

    H = sw(h[1:], horizon)[:m] / c[:m, None]
    L = sw(lo[1:], horizon)[:m] / c[:m, None]
    O = sw(o[1:], horizon)[:m] / c[:m, None]
    C_end = c[horizon:horizon + m] / c[:m]
    ut, us = 1 + target, (1 - stop) if stop else -np.inf
    ht = H >= ut
    hs = L <= us
    both = ht & hs
    ht = ht & ~(both & (O < ut))
    hs = hs & ~(both & (O >= ut))
    big = horizon + 1
    ft = np.where(ht.any(1), ht.argmax(1), big)
    fs = np.where(hs.any(1), hs.argmax(1), big)
    rows = np.arange(m)
    out = np.where(ft < fs, 1, np.where(fs < ft, -1, 0))
    ret = np.where(out == 1, np.maximum(O[rows, np.minimum(ft, horizon - 1)], ut) - 1,
                   np.where(out == -1, np.minimum(O[rows, np.minimum(fs, horizon - 1)], us) - 1 if stop else 0,
                            C_end - 1))
    days = np.where(out == 1, ft + 1, np.where(out == -1, fs + 1, horizon))
    return pd.DataFrame({"outcome": out, "ret": ret, "days": days}, index=df.index[:m])


def _rates(x: pd.DataFrame) -> dict:
    n = len(x)
    if n == 0:
        return {"n": 0, "p_target": np.nan, "p_stop": np.nan, "p_neither": np.nan, "avg_ret": np.nan}
    return {"n": n, "p_target": float((x["outcome"] == 1).mean()), "p_stop": float((x["outcome"] == -1).mean()),
            "p_neither": float((x["outcome"] == 0).mean()), "avg_ret": float(x["ret"].mean())}


def history_odds(df: pd.DataFrame, horizon: int, target: float, stop: float | None,
                 scores: pd.Series | None = None, prior_strength: float = 40.0) -> dict:
    """Unconditional and 'days like today' replay odds, with shrinkage."""
    rep = replay_outcomes(df, horizon, target, stop)
    allr = _rates(rep)
    c = df["Close"]
    regime = (c > ind.sma(c, 200)).reindex(rep.index)
    today_regime = bool(c.iloc[-1] > ind.sma(c, 200).iloc[-1])
    mask = regime == today_regime
    if scores is not None and scores.notna().iloc[-1]:
        s_now = scores.iloc[-1]
        mask &= (scores.reindex(rep.index) - s_now).abs() <= 10
    cond = _rates(rep[mask.fillna(False)])
    k = cond["n"]
    w = k / (k + prior_strength)  # shrink small samples toward the base rate
    shr = {key: (w * cond[key] + (1 - w) * allr[key]) if k else allr[key]
           for key in ("p_target", "p_stop", "p_neither", "avg_ret")}
    return {"all": allr, "similar": cond, "shrunk": shr, "outcomes": rep, "mask": mask}


def setup_history(df: pd.DataFrame, setup_frame: pd.DataFrame, horizon: int, target: float,
                  stop: float | None) -> pd.DataFrame:
    rep = replay_outcomes(df, horizon, target, stop)
    rows = []
    for name in setup_frame.columns:
        fired = setup_frame[name].reindex(rep.index).fillna(False)
        x = rep[fired]
        r = _rates(x)
        rows.append({"Setup": name, "Times fired": r["n"], "Target first": r["p_target"],
                     "Stop first": r["p_stop"], "Neither": r["p_neither"], "Avg return": r["avg_ret"]})
    return pd.DataFrame(rows).set_index("Setup")


# ---------------------------------------------------------------- ML engine


@dataclass
class MLHit:
    auc: float
    brier: float
    base_rate: float
    n_oos: int
    latest: pd.Series            # calibrated P(target first) per ticker
    reliability: pd.DataFrame    # predicted vs actual by bucket
    weight: float                # blend weight earned by skill (0..1)


def ml_hit_model(dfs: dict[str, pd.DataFrame], bench: pd.Series | None, horizon: int, target: float,
                 stop: float | None, n_splits: int = 4) -> MLHit:
    from .ml import FEATURES, build_features, make_model
    from .mlcore import IsotonicCalibrator, brier as brier_score_loss, roc_auc as roc_auc_score

    frames, latest = [], []
    for t, df in dfs.items():
        if len(df) < 400:
            continue
        X = build_features(df, bench)
        st_ = stop.get(t) if isinstance(stop, dict) else stop
        rep = replay_outcomes(df, horizon, target, st_)
        y = (rep["outcome"] == 1).astype(float).reindex(X.index)
        frames.append(X.assign(label=y, ticker=t))
        latest.append(X.iloc[[-1]].assign(ticker=t))
    if not frames:
        raise ValueError("Not enough history for the ML engine.")
    data = pd.concat(frames).dropna(subset=FEATURES + ["label"]).sort_index()
    dates = np.array(sorted(data.index.unique()))
    min_train = 252
    block = (len(dates) - min_train) // n_splits
    oos = []
    for k in range(n_splits):
        ts = min_train + k * block
        te = len(dates) if k == n_splits - 1 else ts + block
        tr = data[data.index <= dates[max(ts - horizon - 1, 0)]]
        tt = data[(data.index >= dates[ts]) & (data.index <= dates[te - 1])]
        if tr["label"].nunique() < 2 or tt.empty:
            continue
        mdl = make_model().fit(tr[FEATURES], tr["label"])
        oos.append(tt[["label"]].assign(p=mdl.predict_proba(tt[FEATURES])[:, 1]))
    if not oos:
        raise ValueError("ML walk-forward produced no folds.")
    oos = pd.concat(oos)
    auc = roc_auc_score(oos["label"], oos["p"]) if oos["label"].nunique() > 1 else 0.5
    iso = IsotonicCalibrator(y_min=0.001, y_max=0.999).fit(oos["p"], oos["label"])
    cal = iso.predict(oos["p"])
    brier = brier_score_loss(oos["label"], cal)
    oos["bucket"] = pd.qcut(oos["p"], 5, labels=False, duplicates="drop")
    rel = oos.assign(cal=cal).groupby("bucket").agg(Predicted=("cal", "mean"), Actual=("label", "mean"),
                                                    Count=("label", "size"))
    final = make_model().fit(data[FEATURES], data["label"])
    lat = pd.concat(latest).dropna(subset=FEATURES)
    p_latest = pd.Series(iso.predict(final.predict_proba(lat[FEATURES])[:, 1]), index=lat["ticker"].values)
    weight = float(np.clip((auc - 0.52) / 0.08, 0, 1))  # 0 at AUC .52, full at .60
    return MLHit(float(auc), float(brier), float(oos["label"].mean()), len(oos), p_latest, rel, weight)


# ------------------------------------------------------------------- blend


@dataclass
class Forecast:
    ticker: str
    price: float
    target_pct: float
    stop_pct: float | None
    horizon: int
    target_price: float
    stop_price: float | None
    p_target: float
    p_stop: float
    p_touch: float
    engines: dict
    ev: float
    breakeven: float
    win_share: float
    verdict: str
    grade: str
    days_to_target: float
    as_of: str = ""
    sim: SimResult | None = None
    hist: dict | None = None
    notes: list = field(default_factory=list)


def verdict(p_t: float, p_s: float, target: float, stop: float | None) -> tuple[str, float, float]:
    """Day-trader-skill style: compare win share among decided paths to break-even."""
    if not stop:
        return ("High odds" if p_t >= 0.6 else "Moderate odds" if p_t >= 0.35 else "Low odds"), np.nan, np.nan
    decided = p_t + p_s
    be = stop / (target + stop)
    if decided < 0.05:
        return "Mostly neither: price likely stays in range", be, np.nan
    share = p_t / decided
    d = share - be
    if abs(d) < 0.05:
        v = "Close to a coin flip" if abs(target - stop) < 1e-9 else "Close to break-even"
    elif abs(d) < 0.15:
        v = "Leans toward the target" if d > 0 else "Leans toward the stop"
    else:
        v = "Strongly toward the target" if d > 0 else "Strongly toward the stop"
    return v, be, share


def grade(ev: float, share: float, be: float, spread: float, stop: float | None, p_t: float) -> str:
    edge = (share - be) if stop and np.isfinite(share) else p_t - 0.5
    if ev > 0.01 and edge >= 0.10 and spread < 0.15:
        return "A"
    if ev > 0.003 and edge >= 0.04:
        return "B"
    if ev > -0.003:
        return "C"
    return "D"


def forecast_stock(ticker: str, df: pd.DataFrame, bench: pd.Series | None, horizon: int, target: float,
                   stop: float | None, ml: MLHit | None = None, n_paths: int = 5000, seed: int = 19) -> Forecast:
    sim = simulate(df, horizon, target, stop, n_paths=n_paths, seed=seed, annual_drift=MARKET_DRIFT)
    scores = factor_scores(df, bench)["Composite"]
    hist = history_odds(df, horizon, target, stop, scores)
    hs = hist["shrunk"]

    engines = {"Time Machine": sim.p_target, "History replay": hs["p_target"]}
    w = {"Time Machine": 0.5, "History replay": 0.5}
    ml_p = None
    if ml is not None and ticker in ml.latest.index and ml.weight > 0:
        ml_p = float(ml.latest[ticker])
        engines["ML (calibrated)"] = ml_p
        w["ML (calibrated)"] = 0.5 * ml.weight
    elif ml is not None and ticker in ml.latest.index:
        engines["ML (calibrated)"] = float(ml.latest[ticker])  # shown, weight 0
    tot = sum(w.values())
    p_t = sum(engines[k] * w[k] for k in w) / tot
    p_s = 0.5 * sim.p_stop + 0.5 * hs["p_stop"] if stop else 0.0
    if p_t + p_s > 1:
        p_s = 1 - p_t
    neither_ret = 0.5 * sim.mean_neither + 0.5 * (hist["all"]["avg_ret"] if np.isfinite(hist["all"]["avg_ret"]) else 0)
    ev = p_t * target - (p_s * stop if stop else 0) + (1 - p_t - p_s) * neither_ret - COST_ROUND_TRIP
    v, be, share = verdict(p_t, p_s, target, stop)
    vals = [engines[k] for k in w]
    spread = max(vals) - min(vals)
    g = grade(ev, share, be, spread, stop, p_t)
    notes = []
    if spread >= 0.15:
        notes.append(f"Engines disagree by {spread:.0%}: treat the blend with caution.")
    if hist["similar"]["n"] < 40:
        notes.append(f"Only {hist['similar']['n']} similar past days, so history is shrunk toward the base rate.")
    if sim.p_neither > 0.4 and stop:
        notes.append(f"{sim.p_neither:.0%} of simulations hit neither level within {horizon} days.")
    price = float(df["Close"].iloc[-1])
    return Forecast(
        ticker=ticker, price=price, target_pct=target, stop_pct=stop, horizon=horizon,
        target_price=price * (1 + target), stop_price=price * (1 - stop) if stop else None,
        p_target=float(p_t), p_stop=float(p_s), p_touch=sim.p_touch, engines=engines, ev=float(ev),
        breakeven=float(be) if np.isfinite(be) else np.nan, win_share=float(share) if np.isfinite(share) else np.nan,
        verdict=v, grade=g, days_to_target=sim.median_days_to_target, as_of=str(df.index[-1].date()), sim=sim, hist=hist, notes=notes,
    )


def odds_matrix(df: pd.DataFrame, targets=(0.02, 0.05, 0.10, 0.15, 0.20), horizons=HORIZONS,
                n_paths: int = 3000, seed: int = 19) -> pd.DataFrame:
    """P(touch +X% within horizon), no stop: blend of simulation and history."""
    rows = {}
    for name, hz in horizons.items():
        rows[name] = {}
        for t in targets:
            s = simulate(df, hz, t, None, n_paths=n_paths, seed=seed, annual_drift=MARKET_DRIFT).p_touch
            h = replay_outcomes(df, hz, t, None)
            hp = float((h["outcome"] == 1).mean()) if len(h) else s
            rows[name][f"+{t:.0%}"] = 0.5 * s + 0.5 * hp
    return pd.DataFrame(rows).T


def evaluate_tracked(df: pd.DataFrame, entry_date, entry: float, target_price: float,
                     stop_price: float | None, horizon: int) -> dict:
    """Score a tracked prediction against what the price actually did afterwards."""
    after = df.loc[df.index > pd.Timestamp(entry_date)].iloc[:horizon]
    if after.empty:
        return {"status": "Open", "exit_date": None, "ret": 0.0, "days": 0}
    for i, (d, row) in enumerate(after.iterrows(), 1):
        hit_s = stop_price is not None and row["Low"] <= stop_price
        hit_t = row["High"] >= target_price
        if hit_s and hit_t:
            if row["Open"] >= target_price:
                hit_s = False
            else:
                hit_t = False
        if hit_s:
            px = min(row["Open"], stop_price)
            return {"status": "Stop hit", "exit_date": d, "ret": px / entry - 1, "days": i}
        if hit_t:
            px = max(row["Open"], target_price)
            return {"status": "Target hit", "exit_date": d, "ret": px / entry - 1, "days": i}
    last = after.iloc[-1]
    if len(after) >= horizon:
        return {"status": "Expired", "exit_date": after.index[-1], "ret": last["Close"] / entry - 1, "days": horizon}
    return {"status": "Open", "exit_date": None, "ret": last["Close"] / entry - 1, "days": len(after)}


# ---------------------------------------------------------- budget baskets


def allocate_budget(cands: pd.DataFrame, budget: float, max_names: int = 3) -> pd.DataFrame:
    """Whole-share allocation of ``budget`` across the best candidates.

    ``cands`` needs columns ticker and Price, sorted best first. Walks down the
    ranking buying names that still fit (aiming for an equal rupee split across
    up to ``max_names``), then spends leftover cash on the chosen names in
    rank order."""
    per = budget / max_names
    cash = budget
    shares: dict[str, int] = {}
    px = dict(zip(cands["ticker"], cands["Price"]))
    for t, p in zip(cands["ticker"], cands["Price"]):
        if len(shares) >= max_names:
            break
        if p <= cash:
            q = max(1, int(min(per, cash) // p))
            shares[t] = q
            cash -= q * p
    changed = True
    while changed:
        changed = False
        for t in list(shares):
            if px[t] <= cash:
                shares[t] += 1
                cash -= px[t]
                changed = True
    out = pd.DataFrame({"ticker": list(shares), "Price": [px[t] for t in shares], "Shares": list(shares.values())})
    out["Invest ₹"] = out["Price"] * out["Shares"]
    out.attrs["cash"] = cash
    return out


def simulate_basket(dfs: dict[str, pd.DataFrame], shares: dict[str, int], horizon: int, target: float,
                    stops: dict[str, float | None], n_paths: int = 4000, seed: int = 19, block: int = 5,
                    lookback: int = 750) -> dict:
    """Joint Monte Carlo of a basket: every stock replays the SAME historical
    days in each path, so their co-movement is kept. Each position exits at its
    own target/stop; the basket result is the rupee-weighted sum."""
    tick = [t for t in shares if shares[t] > 0 and t in dfs]
    ratios = {t: bar_ratios(dfs[t].iloc[-lookback - 1:]) for t in tick}
    common = None
    for r in ratios.values():
        common = r.index if common is None else common.intersection(r.index)
    if common is None or len(common) < 120:
        raise ValueError("Not enough overlapping history for the basket.")
    rng = np.random.default_rng(seed)
    T = len(common)
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, T - block, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, -1)[:, :horizon]
    value0 = sum(shares[t] * float(dfs[t]["Close"].iloc[-1]) for t in tick)
    pnl = np.zeros(n_paths)
    for t in tick:
        R = ratios[t].loc[common, ["o", "h", "l", "c"]].values.copy()
        vol_t, vol_now = _ewma_vol(R[:, 3])
        R = (R - R[:, 3].mean()) * np.clip(vol_now / np.maximum(vol_t, 1e-6), 0.33, 3.0)[:, None]
        R = R + MARKET_DRIFT / TRADING_DAYS
        bars = R[idx]
        cl = np.cumsum(bars[:, :, 3], axis=1)
        prev = np.concatenate([np.zeros((n_paths, 1)), cl[:, :-1]], axis=1)
        o, h, lo = prev + bars[:, :, 0], prev + bars[:, :, 1], prev + bars[:, :, 2]
        lt = np.log1p(target)
        st_ = stops.get(t)
        ls = np.log1p(-st_) if st_ else -np.inf
        ht, hs = h >= lt, lo <= ls
        both = ht & hs
        ht, hs = ht & ~(both & (o < lt)), hs & ~(both & (o >= lt))
        big = horizon + 1
        ft = np.where(ht.any(1), ht.argmax(1), big)
        fs = np.where(hs.any(1), hs.argmax(1), big)
        rows = np.arange(n_paths)
        ret = np.expm1(cl[:, -1])
        rt = np.expm1(np.maximum(o[rows, np.minimum(ft, horizon - 1)], lt))
        rs = np.expm1(np.minimum(o[rows, np.minimum(fs, horizon - 1)], ls)) if st_ else ret
        realised = np.where(ft < fs, rt, np.where(fs < ft, rs, ret))
        pnl += shares[t] * float(dfs[t]["Close"].iloc[-1]) * (realised - COST_ROUND_TRIP)
    r = pnl / value0
    return {"value": value0, "p_reach": float((r >= target - COST_ROUND_TRIP).mean()), "p_loss": float((r < 0).mean()),
            "exp": float(r.mean()), "p5": float(np.percentile(r, 5)), "p50": float(np.percentile(r, 50)),
            "p95": float(np.percentile(r, 95)), "returns": r}
