"""Small NumPy replacements for the handful of scikit-learn pieces JINNI uses.

scikit-learn adds ~30 MB to a serverless bundle (Vercel's Python limit is 225 MB) for a few functions, so they
live here instead:

* ``ledoit_wolf_cov``        shrinkage covariance (Ledoit & Wolf 2004, constant-variance target)
* ``GBClassifier``           histogram gradient-boosted decision trees (logistic loss, level-wise, depth-limited)
* ``IsotonicCalibrator``     monotone probability calibration (pool-adjacent-violators)
* ``roc_auc`` / ``brier`` / ``permutation_importance``

Each is validated against scikit-learn in tests/test_mlcore.py (scikit-learn is a dev-only dependency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- covariance


def ledoit_wolf_shrinkage(X: np.ndarray) -> float:
    """Shrinkage intensity toward mu*I, same estimator as sklearn.covariance.ledoit_wolf_shrinkage."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    X = X - X.mean(axis=0)
    X2 = X**2
    emp_cov_trace = X2.sum(axis=0) / n
    mu = emp_cov_trace.sum() / p
    beta = (X2.T @ X2).sum()
    delta = ((X.T @ X) ** 2).sum() / n**2
    beta = (beta / n - delta) / (p * n)
    delta = (delta - 2.0 * mu * emp_cov_trace.sum() + p * mu**2) / p
    beta = min(beta, delta)
    return 0.0 if beta == 0 else float(beta / delta)


def ledoit_wolf_cov(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / n
    s = ledoit_wolf_shrinkage(X)
    mu = np.trace(S) / p
    return (1 - s) * S + s * mu * np.eye(p)


# -------------------------------------------------------------------- metrics


def roc_auc(y_true, score) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) identity; ties get average ranks."""
    y = np.asarray(y_true).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(score, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier(y_true, p) -> float:
    y, p = np.asarray(y_true, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision(y_true, y_pred) -> float:
    y, p = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    return float((y & p).sum() / p.sum()) if p.sum() else 0.0


# ------------------------------------------------------------------ isotonic


class IsotonicCalibrator:
    """Monotone increasing fit of y on x (pool adjacent violators), linear interpolation, clipped ends."""

    def __init__(self, y_min: float = 0.0, y_max: float = 1.0):
        self.y_min, self.y_max = y_min, y_max

    def fit(self, x, y) -> "IsotonicCalibrator":
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        ux, inv = np.unique(x, return_inverse=True)
        w = np.bincount(inv).astype(float)
        ym = np.bincount(inv, weights=y) / w
        vals, wts, cnts = [], [], []
        for v, wt in zip(ym, w):
            vals.append(v); wts.append(wt); cnts.append(1)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v2, w2, c2 = vals.pop(), wts.pop(), cnts.pop()
                vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / (wts[-1] + w2)
                wts[-1] += w2
                cnts[-1] += c2
        self.x_ = ux
        self.y_ = np.clip(np.repeat(vals, cnts), self.y_min, self.y_max)
        return self

    def predict(self, x) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=float), self.x_[0], self.x_[-1])
        return np.interp(x, self.x_, self.y_)


# ------------------------------------------------------- gradient boosting


class GBClassifier:
    """Histogram gradient-boosted trees for binary classification.

    Logistic loss with Newton leaf values, level-wise trees of depth ``max_depth``, features quantised into
    ``n_bins`` quantile bins, L2-regularised leaves, a minimum leaf size and row subsampling per tree
    (stochastic gradient boosting: faster and a regulariser). Deterministic for a given ``random_state``.
    """

    def __init__(self, max_depth: int = 3, learning_rate: float = 0.12, n_estimators: int = 80,
                 min_samples_leaf: int = 60, l2: float = 1.0, subsample: float = 0.35, n_bins: int = 32,
                 random_state: int = 0):
        self.max_depth, self.lr, self.n_estimators = max_depth, learning_rate, n_estimators
        self.min_leaf, self.l2, self.subsample, self.n_bins = min_samples_leaf, l2, subsample, n_bins
        self.random_state = random_state

    # -- binning
    def _bin(self, X: np.ndarray) -> np.ndarray:
        cols = [np.searchsorted(e, X[:, j], side="right") for j, e in enumerate(self.edges_)]
        return np.stack(cols, axis=1).astype(np.int16)

    def fit(self, X, y) -> "GBClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = [np.unique(np.quantile(X[:, j], qs)) for j in range(p)]
        B = self._bin(X)
        self.nb_ = max(len(e) for e in self.edges_) + 1
        rate = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
        self.init_ = float(np.log(rate / (1 - rate)))
        F = np.full(n, self.init_)
        rng = np.random.default_rng(self.random_state)
        m = n if self.subsample >= 1 else max(int(n * self.subsample), 4 * self.min_leaf)
        min_leaf = max(1, int(self.min_leaf * m / n))
        self.trees_ = []
        for _ in range(self.n_estimators):
            pr = 1.0 / (1.0 + np.exp(-F))
            g, h = pr - y, pr * (1.0 - pr)
            rows = rng.choice(n, m, replace=False) if m < n else np.arange(n)
            tree = self._grow(B[rows], g[rows], h[rows], min_leaf)
            F += self.lr * self._tree_predict(tree, B)
            self.trees_.append(tree)
        return self

    def _grow(self, B: np.ndarray, g: np.ndarray, h: np.ndarray, min_leaf: int):
        n, p = B.shape
        nb, lam = self.nb_, self.l2
        offs = (np.arange(p) * nb)[None, :]
        node = np.zeros(n, dtype=np.int64)
        feats, thrs = [], []
        rep_g, rep_h = np.repeat(g, p), np.repeat(h, p)
        for d in range(self.max_depth):
            m = 1 << d
            flat = (node[:, None] * (p * nb) + offs + B).ravel()
            size = m * p * nb
            G = np.bincount(flat, weights=rep_g, minlength=size).reshape(m, p, nb)
            H = np.bincount(flat, weights=rep_h, minlength=size).reshape(m, p, nb)
            C = np.bincount(flat, minlength=size).reshape(m, p, nb).astype(float)
            GL, HL, CL = G.cumsum(-1), H.cumsum(-1), C.cumsum(-1)
            Gt, Ht, Ct = GL[..., -1:], HL[..., -1:], CL[..., -1:]
            gain = GL**2 / (HL + lam) + (Gt - GL) ** 2 / (Ht - HL + lam) - Gt**2 / (Ht + lam)
            gain = np.where((CL >= min_leaf) & (Ct - CL >= min_leaf), gain, -np.inf)
            flat_gain = gain.reshape(m, p * nb)
            best = flat_gain.argmax(axis=1)
            ok = flat_gain[np.arange(m), best] > 1e-9
            feat = np.where(ok, best // nb, 0)
            thr = np.where(ok, best % nb, nb)  # no split: everything goes left
            feats.append(feat)
            thrs.append(thr)
            go_right = B[np.arange(n), feat[node]] > thr[node]
            node = 2 * node + go_right
        leaves = 1 << self.max_depth
        Gl = np.bincount(node, weights=g, minlength=leaves)
        Hl = np.bincount(node, weights=h, minlength=leaves)
        return feats, thrs, -Gl / (Hl + lam)

    @staticmethod
    def _tree_predict(tree, B: np.ndarray) -> np.ndarray:
        feats, thrs, values = tree
        node = np.zeros(len(B), dtype=np.int64)
        rows = np.arange(len(B))
        for feat, thr in zip(feats, thrs):
            node = 2 * node + (B[rows, feat[node]] > thr[node])
        return values[node]

    def decision_function(self, X) -> np.ndarray:
        B = self._bin(np.asarray(X, dtype=float))
        F = np.full(len(B), self.init_)
        for t in self.trees_:
            F += self.lr * self._tree_predict(t, B)
        return F

    def predict_proba(self, X) -> np.ndarray:
        p1 = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p1, p1])


def permutation_importance(model, X: pd.DataFrame, y, n_repeats: int = 3, seed: int = 0) -> pd.Series:
    """Drop in ROC-AUC when a feature's column is shuffled (higher = the model relies on it more)."""
    rng = np.random.default_rng(seed)
    Xv = X.to_numpy(dtype=float)
    y = np.asarray(y)
    base = roc_auc(y, model.predict_proba(Xv)[:, 1])
    out = {}
    for j, name in enumerate(X.columns):
        drops = []
        for _ in range(n_repeats):
            Xp = Xv.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - roc_auc(y, model.predict_proba(Xp)[:, 1]))
        out[name] = float(np.mean(drops))
    return pd.Series(out)
