"""Direction-prediction model with honest, leak-free validation.

The old app trained on overlapping windows and evaluated with a *random*
train/test split, so the test set contained days sandwiched between training
days — inflating accuracy. Here:

* features are scale-free (returns, ratios, oscillators) so one model can be
  pooled across stocks;
* labels are forward ``horizon``-day returns;
* validation is walk-forward (train on the past, test on the next block) with
  a purge gap of ``horizon`` days between train and test so overlapping labels
  never leak;
* results are reported against the naive base rate ("always predict up").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, roc_auc_score

from . import indicators as ind

FEATURES = [
    "ret_1", "ret_5", "ret_21", "ret_63", "rsi", "macd_hist_n", "atr_n", "dist_sma20",
    "dist_sma50", "dist_sma200", "bb_pctb", "vol20", "vol_ratio", "volume_z", "adx",
    "rs_21", "mkt_ret_5", "mkt_ret_21", "mkt_vol20",
]


def build_features(df: pd.DataFrame, bench: pd.Series | None = None) -> pd.DataFrame:
    c = df["Close"]
    f = pd.DataFrame(index=df.index)
    for n in (1, 5, 21, 63):
        f[f"ret_{n}"] = c.pct_change(n)
    f["rsi"] = ind.rsi(c) / 100
    _, _, hist = ind.macd(c)
    f["macd_hist_n"] = hist / c
    f["atr_n"] = ind.atr(df) / c
    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = c / ind.sma(c, n) - 1
    up, _, lo = ind.bollinger(c)
    f["bb_pctb"] = (c - lo) / (up - lo)
    f["vol20"] = ind.realized_vol(c, 20)
    f["vol_ratio"] = f["vol20"] / ind.realized_vol(c, 60)
    lv = np.log1p(df["Volume"])
    f["volume_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    f["adx"] = ind.adx(df)[0] / 100
    if bench is not None:
        b = bench.reindex(df.index).ffill()
        f["rs_21"] = c.pct_change(21) - b.pct_change(21)
        f["mkt_ret_5"] = b.pct_change(5)
        f["mkt_ret_21"] = b.pct_change(21)
        f["mkt_vol20"] = ind.realized_vol(b, 20)
    else:
        for k in ("rs_21", "mkt_ret_5", "mkt_ret_21", "mkt_vol20"):
            f[k] = 0.0
    return f.replace([np.inf, -np.inf], np.nan)


def build_labels(close: pd.Series, horizon: int, bench: pd.Series | None = None,
                 relative: bool = False, threshold: float = 0.0) -> pd.Series:
    fwd = close.shift(-horizon) / close - 1
    if relative and bench is not None:
        b = bench.reindex(close.index).ffill()
        fwd = fwd - (b.shift(-horizon) / b - 1)
    lab = (fwd > threshold).astype(float)
    lab[fwd.isna()] = np.nan
    return lab


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.04, max_iter=250, min_samples_leaf=60,
        l2_regularization=1.0, random_state=0,
    )


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    oos: pd.DataFrame  # date, ticker, proba, label, fwd_ret
    summary: dict
    latest: pd.DataFrame  # ticker -> probability for the next horizon
    importance: pd.Series


def walk_forward(dfs: dict[str, pd.DataFrame], bench: pd.Series | None, horizon: int = 5,
                 n_splits: int = 5, relative: bool = False, threshold: float = 0.0,
                 min_train_days: int = 252) -> WalkForwardResult:
    """Pooled walk-forward validation across one or more stocks."""
    frames = []
    latest_rows = []
    for t, df in dfs.items():
        if len(df) < min_train_days + 60:
            continue
        X = build_features(df, bench)
        y = build_labels(df["Close"], horizon, bench, relative, threshold)
        fwd = df["Close"].shift(-horizon) / df["Close"] - 1
        d = X.assign(label=y, fwd_ret=fwd, ticker=t)
        frames.append(d)
        latest_rows.append(X.iloc[[-1]].assign(ticker=t))
    if not frames:
        raise ValueError("Not enough history to train (need > 1 year per stock).")
    data = pd.concat(frames).dropna(subset=FEATURES + ["label"]).sort_index()
    dates = np.array(sorted(data.index.unique()))
    if len(dates) < min_train_days + 2 * horizon + 50:
        raise ValueError("Not enough clean history after feature warm-up.")

    test_block = (len(dates) - min_train_days) // n_splits
    folds, oos_parts = [], []
    for k in range(n_splits):
        test_start = min_train_days + k * test_block
        test_end = len(dates) if k == n_splits - 1 else test_start + test_block
        train_end = test_start - horizon  # purge gap
        if train_end < 100:
            continue
        tr = data[data.index <= dates[train_end - 1]]
        te = data[(data.index >= dates[test_start]) & (data.index <= dates[test_end - 1])]
        if tr["label"].nunique() < 2 or te.empty:
            continue
        model = make_model().fit(tr[FEATURES], tr["label"])
        p = model.predict_proba(te[FEATURES])[:, 1]
        pred = (p >= 0.5).astype(int)
        base = max(te["label"].mean(), 1 - te["label"].mean())
        folds.append({
            "Fold": k + 1,
            "Test from": pd.Timestamp(dates[test_start]).date(),
            "Test to": pd.Timestamp(dates[test_end - 1]).date(),
            "Train rows": len(tr),
            "Test rows": len(te),
            "Accuracy": accuracy_score(te["label"], pred),
            "Base rate": base,
            "Edge vs base": accuracy_score(te["label"], pred) - base,
            "AUC": roc_auc_score(te["label"], p) if te["label"].nunique() > 1 else np.nan,
            "Precision (up)": precision_score(te["label"], pred, zero_division=0),
        })
        oos_parts.append(te[["ticker", "label", "fwd_ret"]].assign(proba=p))
    if not folds:
        raise ValueError("Walk-forward produced no valid folds.")
    folds_df = pd.DataFrame(folds).set_index("Fold")
    oos = pd.concat(oos_parts)
    pred = (oos["proba"] >= 0.5).astype(int)
    summary = {
        "accuracy": accuracy_score(oos["label"], pred),
        "base_rate": max(oos["label"].mean(), 1 - oos["label"].mean()),
        "auc": roc_auc_score(oos["label"], oos["proba"]) if oos["label"].nunique() > 1 else np.nan,
        "brier": brier_score_loss(oos["label"], oos["proba"]),
        "n_oos": len(oos),
    }
    summary["edge"] = summary["accuracy"] - summary["base_rate"]
    # bucket analysis: does a higher probability mean higher realised return?
    oos["bucket"] = pd.qcut(oos["proba"], 5, labels=["Q1 (most bearish)", "Q2", "Q3", "Q4", "Q5 (most bullish)"],
                            duplicates="drop")
    summary["buckets"] = oos.groupby("bucket", observed=True).agg(
        Predictions=("proba", "size"), Avg_prob=("proba", "mean"),
        Hit_rate=("label", "mean"), Avg_fwd_ret=("fwd_ret", "mean"),
    )

    # Final model on all labelled data -> live probability
    final = make_model().fit(data[FEATURES], data["label"])
    latest = pd.concat(latest_rows).dropna(subset=FEATURES)
    latest_p = pd.DataFrame({
        "ticker": latest["ticker"].values,
        "as_of": latest.index,
        "prob_up": final.predict_proba(latest[FEATURES])[:, 1] if len(latest) else [],
    }).set_index("ticker")

    # Permutation importance on the most recent fold (cheap, model-agnostic)
    from sklearn.inspection import permutation_importance

    last = data.iloc[-min(len(data), 3000):]
    pi = permutation_importance(final, last[FEATURES], last["label"], n_repeats=3, random_state=0,
                                scoring="roc_auc")
    importance = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
    return WalkForwardResult(folds_df, oos, summary, latest_p, importance)
