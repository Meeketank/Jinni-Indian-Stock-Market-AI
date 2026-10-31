"""Short-horizon direction outlook for the market or a stock, with honest validation.

The model is trained on what a trader could know at the close of day t: the stock's own trend and volatility, the
US session that ends *after* India closes (S&P 500, Nasdaq, US VIX), Asian markets, crude, gold, the dollar, the
rupee, US yields and India VIX. It predicts whether the next ``h`` trading days end higher. It is validated
walk-forward (train on the past, test on the next block, with a purge gap) and reports its out-of-sample AUC
against a coin flip, so the screen can say plainly when it has no edge.

News and event tone is *not* part of the trained model (free news has no deep history to train on). It is applied
afterwards as a small, capped tilt and labelled unvalidated; ``tone_vs_next_day`` reports how GDELT's tone history
actually lined up with next-day NIFTY returns over the window it covers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import indicators as ind
from .mlcore import GBClassifier, IsotonicCalibrator, accuracy, roc_auc

MARKET = "^NSEI"
CUES = {  # ticker -> short name used in feature names
    "^GSPC": "sp500", "^IXIC": "nasdaq", "^N225": "nikkei", "^HSI": "hangseng", "^VIX": "us_vix", "^INDIAVIX": "india_vix",
    "CL=F": "wti", "BZ=F": "brent", "GC=F": "gold", "DX-Y.NYB": "dollar", "USDINR=X": "usdinr", "^TNX": "us10y",
}
TICKERS = [MARKET, *CUES]
FRIENDLY = {
    "sp500": "S&P 500 (US close after India shut)", "nasdaq": "Nasdaq (US close)", "nikkei": "Nikkei", "hangseng": "Hang Seng",
    "us_vix": "US VIX", "india_vix": "India VIX", "wti": "WTI crude", "brent": "Brent crude", "gold": "Gold",
    "dollar": "US dollar index", "usdinr": "USD/INR", "us10y": "US 10-yr yield",
}
CAP = 0.03  # max probability shift from news/event tone, in absolute terms (3 points)


def _friendly(feat: str) -> str:
    for k, v in FRIENDLY.items():
        if feat.startswith(k + "_"):
            return f"{v} · {feat[len(k) + 1:].replace('r1', '1-day move').replace('r5', '5-day move').replace('lvl', 'level').replace('chg', 'change')}"
    return {"ret_1": "Own 1-day return", "ret_5": "Own 5-day return", "ret_21": "Own 1-month return",
            "rsi": "RSI(14)", "dist_sma20": "Distance from 20-DMA", "dist_sma50": "Distance from 50-DMA",
            "dist_sma200": "Distance from 200-DMA", "vol20": "20-day volatility", "vol_ratio": "Volatility trend (20d vs 60d)",
            "range_pos": "Position in 3-month range", "volume_z": "Volume vs normal", "rel_21": "1-month strength vs NIFTY",
            }.get(feat, feat)


def build_features(target: pd.DataFrame, cues: pd.DataFrame, is_market: bool, bench: pd.Series | None = None
                   ) -> pd.DataFrame:
    """Features known at the close of each date. ``target``: OHLCV of the thing being predicted; ``cues``: closes."""
    c = target["Close"]
    f = pd.DataFrame(index=target.index)
    for n in (1, 5, 21):
        f[f"ret_{n}"] = c.pct_change(n)
    f["rsi"] = ind.rsi(c) / 100
    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = c / ind.sma(c, n) - 1
    f["vol20"] = ind.realized_vol(c, 20)
    f["vol_ratio"] = f["vol20"] / ind.realized_vol(c, 60)
    hi, lo = target["High"].rolling(63).max(), target["Low"].rolling(63).min()
    f["range_pos"] = (c - lo) / (hi - lo)
    if "Volume" in target and target["Volume"].sum() > 0:
        lv = np.log1p(target["Volume"])
        f["volume_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    if not is_market and bench is not None:
        b = bench.reindex(target.index).ffill()
        f["rel_21"] = c.pct_change(21) - b.pct_change(21)
    cu = cues.reindex(target.index.union(cues.index)).ffill(limit=4).reindex(target.index)
    for tk, nm in CUES.items():
        if tk not in cu or cu[tk].notna().sum() < 250:
            continue
        s = cu[tk]
        if nm in ("us_vix", "india_vix"):
            f[f"{nm}_lvl"] = s.rolling(252, min_periods=60).rank(pct=True)
            f[f"{nm}_chg"] = s.pct_change()
        elif nm == "us10y":
            f[f"{nm}_chg"] = s.diff()
            f[f"{nm}_r5"] = s.diff(5)
        else:
            f[f"{nm}_r1"] = s.pct_change()
            if nm in ("sp500", "nasdaq", "dollar", "usdinr"):
                f[f"{nm}_r5"] = s.pct_change(5)
    return f.replace([np.inf, -np.inf], np.nan)


def _model() -> GBClassifier:
    return GBClassifier(max_depth=3, learning_rate=0.08, n_estimators=70, min_samples_leaf=60, l2=2.0, subsample=0.4, random_state=7)


@dataclass
class Outlook:
    h: int
    as_of: pd.Timestamp
    p_raw: float
    p_cal: float
    auc: float
    accuracy: float
    base_rate: float
    n_oos: int
    folds: pd.DataFrame
    drivers: pd.DataFrame
    buckets: pd.DataFrame
    snapshot: dict = field(default_factory=dict)
    exp_move: float = np.nan   # one-sigma move over h days, from current volatility
    train_rows: int = 0

    @property
    def edge(self) -> str:
        if not np.isfinite(self.auc) or self.auc < 0.52:
            return "none"
        return "weak" if self.auc < 0.56 else "modest" if self.auc < 0.62 else "useful"


def fit_outlook(target: pd.DataFrame, cues: pd.DataFrame, h: int = 1, is_market: bool = True,
                bench: pd.Series | None = None, n_splits: int = 5, min_train: int = 750) -> Outlook:
    X = build_features(target, cues, is_market, bench)
    y = (target["Close"].shift(-h) / target["Close"] - 1)
    label = (y > 0).astype(float).where(y.notna())
    cols = [c for c in X.columns if X[c].notna().mean() > 0.9]
    X = X[cols]
    live = X.iloc[[-1]].ffill().fillna(X.median())
    data = X.assign(_y=label, _r=y).dropna(subset=cols + ["_y"])
    if len(data) < min_train + 4 * h + 100:
        raise ValueError("Not enough history to train and validate an outlook for this symbol.")
    n = len(data)
    block = (n - min_train) // n_splits
    folds, oos = [], []
    for k in range(n_splits):
        a = min_train + k * block
        b = n if k == n_splits - 1 else a + block
        tr, te = data.iloc[: a - h], data.iloc[a:b]  # purge h days so overlapping labels never leak
        if tr["_y"].nunique() < 2 or te.empty:
            continue
        m = _model().fit(tr[cols], tr["_y"])
        p = m.predict_proba(te[cols])[:, 1]
        base = max(te["_y"].mean(), 1 - te["_y"].mean())
        folds.append({"Test from": te.index[0].date(), "Test to": te.index[-1].date(), "Days": len(te),
                      "Accuracy": accuracy(te["_y"], p >= 0.5), "Base rate": base,
                      "AUC": roc_auc(te["_y"], p) if te["_y"].nunique() > 1 else np.nan})
        oos.append(pd.DataFrame({"p": p, "y": te["_y"].values, "r": te["_r"].values}, index=te.index))
    if not oos:
        raise ValueError("Walk-forward validation produced no folds.")
    o = pd.concat(oos)
    auc = roc_auc(o["y"], o["p"])
    acc = accuracy(o["y"], o["p"] >= 0.5)
    base = max(o["y"].mean(), 1 - o["y"].mean())
    cal = IsotonicCalibrator(0.02, 0.98).fit(o["p"].values, o["y"].values)
    final = _model().fit(data[cols], data["_y"])
    p_raw = float(final.predict_proba(live[cols])[:, 1][0])
    p_cal = float(np.clip(cal.predict([p_raw])[0], 0.05, 0.95))
    # How much each input is pushing today's call: replace it by its typical value and see the probability move.
    med = data[cols].median()
    rows = []
    for c_ in cols:
        alt = live.copy()
        alt[c_] = med[c_]
        d = p_raw - float(final.predict_proba(alt[cols])[:, 1][0])
        rows.append({"Driver": _friendly(c_), "Today": float(live[c_].iloc[0]), "Typical": float(med[c_]), "Push": d})
    drivers = pd.DataFrame(rows)
    drivers = drivers.reindex(drivers["Push"].abs().sort_values(ascending=False).index).head(8).reset_index(drop=True)
    o["bucket"] = pd.qcut(o["p"], 5, duplicates="drop")
    bk = o.groupby("bucket", observed=True).agg(Calls=("y", "size"), Avg_p=("p", "mean"), Up=("y", "mean"), Avg_move=("r", "mean"))
    bk.index = [f"Q{i + 1}" for i in range(len(bk))]
    vol = float(live["vol20"].iloc[0]) if "vol20" in live else np.nan
    return Outlook(h, target.index[-1], p_raw, p_cal, auc, acc, base, len(o), pd.DataFrame(folds), drivers, bk.reset_index(names="Bucket"),
                   {c_: float(live[c_].iloc[0]) for c_ in cols}, vol * np.sqrt(h / 252) if np.isfinite(vol) else np.nan, len(data))


def tilt(p: float, tone: float, n: int, heat: float = 0.0) -> dict:
    """Small news/event adjustment: tone in [-1, 1] scaled by how many headlines back it up, capped at ±CAP.
    Unvalidated: shown separately so the reader can see the trained model and the tilt apart."""
    if not np.isfinite(tone) or n < 8:
        return {"shift": 0.0, "p": p, "tone": tone, "n": n}
    conf = min(1.0, n / 60)                       # few headlines -> small weight
    shift = float(np.clip(CAP * np.tanh(2.0 * tone) * conf, -CAP, CAP))
    return {"shift": shift, "p": float(np.clip(p + shift, 0.03, 0.97)), "tone": tone, "n": n}


def tone_vs_next_day(tone: pd.Series | None, market_close: pd.Series) -> dict | None:
    """Did GDELT's daily news tone line up with the NEXT NIFTY session's return? (Spearman, with its p-value.)"""
    if tone is None or len(tone) < 25:
        return None
    r = market_close.pct_change().shift(-1)  # return from day t's close to the next close
    z = pd.DataFrame({"tone": tone.diff(), "r": r}).dropna()  # change in tone: is news getting better or worse?
    z = z[z.index.isin(market_close.index)]
    if len(z) < 25:
        return None
    rho, pv = stats.spearmanr(z["tone"], z["r"])
    return {"n": int(len(z)), "rho": float(rho), "p": float(pv), "from": z.index[0].date(), "to": z.index[-1].date()}
