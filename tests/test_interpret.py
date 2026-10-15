import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_iris

from fisher_lda import FisherLDA, interpret as I

Xw, yw = load_wine(return_X_y=True)
names_w = list(load_wine().feature_names)


def test_standardized_coefficients_are_scale_invariant():
    X2 = Xw.copy(); X2[:, 0] *= 1000.0
    a = I.standardized_coefficients(FisherLDA().fit(Xw, yw)).to_numpy()
    b = I.standardized_coefficients(FisherLDA().fit(X2, yw)).to_numpy()
    np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-8)


def test_projected_axes_have_unit_within_class_variance_in_sd_units():
    m = FisherLDA().fit(Xw, yw)
    sd = np.sqrt(np.diag(m.covariance_))
    # coefficient in SD units applied to within-class-standardised features gives the same LD score
    Xs = (Xw - m.xbar_) / sd
    np.testing.assert_allclose(Xs @ I.standardized_coefficients(m).to_numpy(), m.transform(Xw), atol=1e-8)


def test_structure_correlations_match_numpy():
    m = FisherLDA().fit(Xw, yw)
    S = I.structure_correlations(m, Xw, names_w)
    Z = m.transform(Xw)
    ref = np.corrcoef(Xw[:, 3], Z[:, 0])[0, 1]
    assert np.isclose(S.iloc[3, 0], ref)
    assert S.abs().to_numpy().max() <= 1 + 1e-9 and S.shape == (13, 2)


def test_feature_importance_sums_to_100_and_sorted():
    imp = I.feature_importance(FisherLDA().fit(Xw, yw), names_w)
    assert np.isclose(imp.sum(), 100) and imp.is_monotonic_decreasing


def test_centroid_distances_properties():
    m = FisherLDA().fit(Xw, yw)
    D = I.centroid_distances(m)
    np.testing.assert_allclose(D.to_numpy(), D.to_numpy().T)
    assert np.allclose(np.diag(D), 0)
    d = m.means_[0] - m.means_[2]
    assert np.isclose(D.iloc[0, 2], np.sqrt(d @ np.linalg.solve(m.covariance_, d)), rtol=1e-6)
    pairs = I.closest_class_pairs(D, 2)
    assert len(pairs) == 2 and pairs["mahalanobis"].is_monotonic_increasing


def test_bootstrap_loadings_shape_and_stability():
    df, cos = I.bootstrap_loadings(Xw, yw, names_w, n_boot=40, n_axes=2)
    assert len(df) == 13 * 2 and {"ci_low", "ci_high", "boot_std"} <= set(df.columns)
    assert (df["ci_low"] <= df["ci_high"]).all()
    assert cos["LD1"] > 0.9                      # first axis of Wine is well determined


def test_decision_regions_2d():
    X, y = load_iris(return_X_y=True)
    res = I.decision_regions_2d(X, y, grid=60)
    assert res["regions"].shape == (60, 60) and set(np.unique(res["regions"])) <= {0, 1, 2}
    assert 0.8 < res["accuracy_2d"] <= 1 and 0 <= res["accuracy_full"] <= 1
