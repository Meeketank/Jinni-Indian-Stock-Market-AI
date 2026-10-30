"""The NumPy replacements for scikit-learn must agree with scikit-learn (when available) and find real signal."""

import numpy as np
import pandas as pd
import pytest

from jinni import mlcore as mc

rng = np.random.default_rng(0)


def test_ledoit_wolf_matches_sklearn():
    sk = pytest.importorskip("sklearn.covariance")
    for n, p in [(500, 10), (250, 40), (60, 30)]:
        X = rng.normal(size=(n, p)) @ rng.normal(size=(p, p)) * 0.01
        np.testing.assert_allclose(mc.ledoit_wolf_cov(X), sk.LedoitWolf().fit(X).covariance_, atol=1e-12)
        assert mc.ledoit_wolf_shrinkage(X) == pytest.approx(sk.ledoit_wolf_shrinkage(X))


def test_shrinkage_covariance_is_psd_and_symmetric():
    X = rng.normal(size=(40, 30))  # fewer observations than parameters is where shrinkage matters
    C = mc.ledoit_wolf_cov(X)
    assert np.allclose(C, C.T) and np.linalg.eigvalsh(C).min() > 0


def test_isotonic_matches_sklearn_and_is_monotone():
    x = rng.random(3000)
    y = (rng.random(3000) < 0.2 + 0.5 * x).astype(float)
    x[:200] = np.round(x[:200], 1)  # ties
    grid = np.linspace(-0.1, 1.1, 60)
    mine = mc.IsotonicCalibrator(0.001, 0.999).fit(x, y).predict(grid)
    assert (np.diff(mine) >= -1e-12).all() and mine.min() >= 0.001 and mine.max() <= 0.999
    iso = pytest.importorskip("sklearn.isotonic")
    ref = iso.IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(x, y).predict(grid)
    np.testing.assert_allclose(mine, ref, atol=1e-12)


def test_metrics_match_sklearn():
    s = rng.random(2000)
    y = (rng.random(2000) < s).astype(int)
    s[:300] = np.round(s[:300], 1)
    m = pytest.importorskip("sklearn.metrics")
    assert mc.roc_auc(y, s) == pytest.approx(m.roc_auc_score(y, s))
    assert mc.brier(y, s) == pytest.approx(m.brier_score_loss(y, s))
    assert mc.precision(y, s > 0.5) == pytest.approx(m.precision_score(y, s > 0.5))
    assert np.isnan(mc.roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))  # undefined with one class


def _planted(n=30000, p=12):
    X = rng.normal(size=(n, p))
    logit = 1.6 * (X[:, 0] > 0.3) * (X[:, 1] < 0) + 1.2 * np.tanh(2 * X[:, 2]) - 0.8  # threshold + interaction
    return X, (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float), logit


def test_gbm_finds_nonlinear_signal_and_matches_sklearn():
    X, y, logit = _planted()
    tr, te = slice(0, 22000), slice(22000, None)
    g = mc.GBClassifier().fit(X[tr], y[tr])
    auc = mc.roc_auc(y[te], g.predict_proba(X[te])[:, 1])
    oracle = mc.roc_auc(y[te], logit[te])
    assert auc > oracle - 0.02  # essentially recovers the true signal
    hgb = pytest.importorskip("sklearn.ensemble")
    ref = hgb.HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=250, min_samples_leaf=60,
                                             l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
    assert auc > mc.roc_auc(y[te], ref.predict_proba(X[te])[:, 1]) - 0.01


def test_gbm_has_no_skill_on_pure_noise():
    X, _, _ = _planted()
    y = (rng.random(len(X)) < 0.45).astype(float)
    g = mc.GBClassifier().fit(X[:22000], y[:22000])
    assert abs(mc.roc_auc(y[22000:], g.predict_proba(X[22000:])[:, 1]) - 0.5) < 0.03


def test_gbm_is_deterministic_and_outputs_probabilities():
    X, y, _ = _planted(6000)
    a = mc.GBClassifier(random_state=3).fit(X, y).predict_proba(X)
    b = mc.GBClassifier(random_state=3).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)
    assert ((a >= 0) & (a <= 1)).all() and np.allclose(a.sum(axis=1), 1)


def test_permutation_importance_names_the_real_features():
    X, y, _ = _planted(20000)
    g = mc.GBClassifier().fit(X[:15000], y[:15000])
    df = pd.DataFrame(X[15000:], columns=[f"f{i}" for i in range(X.shape[1])])
    imp = mc.permutation_importance(g, df, y[15000:])
    assert set(imp.sort_values(ascending=False).index[:3]) == {"f0", "f1", "f2"}
