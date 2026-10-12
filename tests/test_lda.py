import numpy as np
import pytest
from sklearn.datasets import load_wine, load_iris, make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from fisher_lda import FisherLDA


@pytest.fixture(params=["wine", "iris", "synthetic"])
def data(request):
    if request.param == "wine":
        return load_wine(return_X_y=True)
    if request.param == "iris":
        return load_iris(return_X_y=True)
    return make_classification(
        n_samples=600, n_features=8, n_informative=5, n_redundant=2,
        n_classes=4, n_clusters_per_class=1, random_state=0,
    )


def _abs_cols(A):
    """Compare projections up to per-axis sign."""
    return np.abs(A)


def test_projection_matches_sklearn(data):
    X, y = data
    ours = FisherLDA().fit(X, y).transform(X)
    ref = LinearDiscriminantAnalysis(solver="svd").fit(X, y).transform(X)
    assert ours.shape == ref.shape
    np.testing.assert_allclose(_abs_cols(ours), _abs_cols(ref), rtol=1e-6, atol=1e-6)


def test_predictions_match_sklearn(data):
    X, y = data
    ours = FisherLDA().fit(X, y)
    ref = LinearDiscriminantAnalysis(solver="svd").fit(X, y)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


def test_probabilities_match_sklearn(data):
    X, y = data
    ours = FisherLDA().fit(X, y).predict_proba(X)
    ref = LinearDiscriminantAnalysis(solver="svd").fit(X, y).predict_proba(X)
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_explained_variance_ratio_matches_sklearn(data):
    X, y = data
    ours = FisherLDA().fit(X, y).explained_variance_ratio_
    ref = LinearDiscriminantAnalysis(solver="svd").fit(X, y).explained_variance_ratio_
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_fisher_criterion_is_maximised(data):
    """LD1 must achieve J(w) = lambda_1 and no random direction may beat it."""
    X, y = data
    m = FisherLDA().fit(X, y)
    J = lambda w: (w @ m.Sb_ @ w) / (w @ m.Sw_ @ w)
    best = J(m.scalings_[:, 0])
    np.testing.assert_allclose(best, m.eigenvalues_[0], rtol=1e-6)
    rng = np.random.default_rng(0)
    for _ in range(500):
        assert J(rng.normal(size=X.shape[1])) <= best + 1e-9


def test_projected_within_class_covariance_is_identity(data):
    X, y = data
    m = FisherLDA().fit(X, y)
    W = m.scalings_
    np.testing.assert_allclose(W.T @ m.covariance_ @ W, np.eye(W.shape[1]), atol=1e-8)


def test_discriminant_axes_are_uncorrelated(data):
    X, y = data
    Z = FisherLDA().fit_transform(X, y)
    C = np.cov(Z.T)
    off_diag = C - np.diag(np.diag(C))
    # total covariance of projections is diagonal for LDA
    assert np.abs(off_diag).max() < 1e-8 * max(1.0, np.abs(C).max())


def test_n_components_and_validation(data):
    X, y = data
    K = len(np.unique(y))
    assert FisherLDA(n_components=1).fit_transform(X, y).shape[1] == 1
    assert FisherLDA().fit_transform(X, y).shape[1] == min(K - 1, X.shape[1])
    with pytest.raises(ValueError):
        FisherLDA(n_components=K).fit(X, y)


def test_string_labels_and_unfitted_error():
    X, y = load_iris(return_X_y=True)
    labels = np.array(["a", "b", "c"])[y]
    m = FisherLDA().fit(X, labels)
    assert set(m.predict(X)) <= {"a", "b", "c"}
    with pytest.raises(RuntimeError):
        FisherLDA().transform(X)


def test_handles_collinear_features():
    X, y = load_iris(return_X_y=True)
    X = np.hstack([X, X[:, [0]] * 2.0])       # perfectly collinear column
    m = FisherLDA().fit(X, y)
    assert m.score(X, y) > 0.9
