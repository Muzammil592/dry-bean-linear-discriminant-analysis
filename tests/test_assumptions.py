import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.datasets import load_wine

from fisher_lda import assumptions as A


def _gaussian_classes(equal_cov=True, n=400, seed=0):
    rng = np.random.default_rng(seed)
    p = 4
    S1 = np.array([[1, .5, 0, 0], [.5, 1, 0, 0], [0, 0, 1, .3], [0, 0, .3, 1]])
    S2 = S1 if equal_cov else np.diag([4, .2, 3, .3])
    X = np.vstack([rng.multivariate_normal(np.zeros(p), S1, n),
                   rng.multivariate_normal(np.ones(p), S2, n)])
    y = np.repeat([0, 1], n)
    return pd.DataFrame(X, columns=list("abcd")), y


def test_box_m_accepts_equal_and_rejects_unequal_covariance():
    X, y = _gaussian_classes(equal_cov=True)
    assert A.box_m_test(X, y)["p_value"] > 0.01
    X, y = _gaussian_classes(equal_cov=False)
    assert A.box_m_test(X, y)["p_value"] < 1e-6


def test_box_m_matches_reference_formula_two_groups():
    # Compare against a direct, independent computation of M for 2 groups.
    X, y = _gaussian_classes(equal_cov=False, n=150, seed=3)
    S = [np.cov(X[y == k].values, rowvar=False) for k in (0, 1)]
    n = [150, 150]; N = 300; g = 2; p = 4
    Sp = ((n[0] - 1) * S[0] + (n[1] - 1) * S[1]) / (N - g)
    M = (N - g) * np.log(np.linalg.det(Sp)) - sum((ni - 1) * np.log(np.linalg.det(Si)) for ni, Si in zip(n, S))
    assert np.isclose(A.box_m_test(X, y)["M"], M)


def test_normality_flags_normal_vs_skewed():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"normal": rng.normal(size=600), "skewed": rng.exponential(size=600)})
    y = np.zeros(600)
    res = A.univariate_normality(X, y).set_index("feature")
    assert bool(res.loc["normal", "normal"]) is True
    assert bool(res.loc["skewed", "normal"]) is False
    assert res.loc["skewed", "skew"] > 1


def test_mardia_accepts_gaussian_rejects_heavy_tail():
    X, y = _gaussian_classes(n=600)
    assert A.multivariate_normality(X, y)["p_value"].min() > 0.001
    rng = np.random.default_rng(2)
    Xh = pd.DataFrame(rng.standard_t(df=3, size=(800, 4)), columns=list("abcd"))
    assert A.multivariate_normality(Xh, np.zeros(800))["p_value"].iloc[0] < 0.01


def test_vif_detects_collinearity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    X = pd.DataFrame({"a": a, "b": a * 2 + rng.normal(scale=.05, size=500), "c": rng.normal(size=500)})
    v = A.vif_table(X).set_index("feature")["VIF"]
    assert v["a"] > 100 and v["b"] > 100 and v["c"] < 2
    assert A.condition_number(X) > 10


def test_vif_matches_regression_definition():
    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    X["c"] = X["a"] * .7 + X["c"] * .5
    Xa = X[["a", "b"]].to_numpy(); Xa = np.c_[np.ones(len(Xa)), Xa]
    beta = np.linalg.lstsq(Xa, X["c"], rcond=None)[0]
    r2 = 1 - np.sum((X["c"] - Xa @ beta) ** 2) / np.sum((X["c"] - X["c"].mean()) ** 2)
    assert np.isclose(A.vif_table(X).set_index("feature").loc["c", "VIF"], 1 / (1 - r2))


def test_high_corr_pairs_and_covariance_ratio():
    X, y = _gaussian_classes(equal_cov=False)
    cr = A.covariance_ratio(X, y)
    assert cr["relative_distance"].min() > 0.1
    X2, y2 = _gaussian_classes(equal_cov=True)
    assert A.covariance_ratio(X2, y2)["relative_distance"].max() < 0.2
    pairs = A.high_correlation_pairs(pd.DataFrame({"a": [1, 2, 3, 4.], "b": [2, 4, 6, 8.1]}), 0.9)
    assert len(pairs) == 1


def test_assumption_report_runs_on_real_data():
    d = load_wine(as_frame=True)
    rep = A.assumption_report(d.data, d.target.to_numpy())
    assert set(rep) >= {"box_m", "vif", "normality_summary", "condition_number"}
    assert rep["normality_summary"].shape[0] == 3
