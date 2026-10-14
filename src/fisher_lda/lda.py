"""Fisher's Linear Discriminant Analysis, implemented from scratch with NumPy.

Two views of the same model are implemented:

1. Dimensionality reduction (Fisher's criterion): find directions w that
   maximise  J(w) = (w' S_B w) / (w' S_W w).
2. Classification (Gaussian generative model): assume each class is
   N(mu_k, Sigma) with a *shared* covariance and classify with Bayes' rule.

Solving the generalised eigenproblem  S_B w = lambda S_W w  is done by first
whitening the within-class scatter (symmetric eigendecomposition), which is
numerically more stable than forming inv(S_W) @ S_B explicitly.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


class FisherLDA(ClassifierMixin, TransformerMixin, BaseEstimator):
    """Linear Discriminant Analysis.

    Parameters
    ----------
    n_components : int or None
        Number of discriminant axes to keep. Defaults to min(n_classes - 1, n_features).
    priors : array-like or None
        Class priors. If None, estimated from class frequencies.

    Attributes
    ----------
    classes_ : (K,) unique class labels
    priors_ : (K,) class priors
    means_ : (K, p) class means
    xbar_ : (p,) overall (prior-weighted) mean
    covariance_ : (p, p) pooled within-class covariance, S_W / (n - K)
    Sw_, Sb_ : within- and between-class scatter matrices
    eigenvalues_ : (r,) discriminant eigenvalues, descending
    explained_variance_ratio_ : (r,) share of between-class separation per axis
    scalings_ : (p, r) projection matrix; transform(X) = (X - xbar_) @ scalings_
    """

    def __init__(self, n_components: int | None = None, priors=None):
        self.n_components = n_components
        self.priors = priors

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, p = X.shape
        self.n_features_in_ = p
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("Need at least 2 classes.")
        if n <= K:
            raise ValueError("Need more samples than classes.")

        counts = np.bincount(y_idx, minlength=K)
        self.priors_ = (
            counts / n if self.priors is None else np.asarray(self.priors, dtype=float)
        )
        self.means_ = np.vstack([X[y_idx == k].mean(axis=0) for k in range(K)])
        self.xbar_ = self.priors_ @ self.means_

        # Within-class scatter S_W = sum_k sum_{x in k} (x - mu_k)(x - mu_k)'
        Sw = np.zeros((p, p))
        for k in range(K):
            D = X[y_idx == k] - self.means_[k]
            Sw += D.T @ D
        # Between-class scatter S_B = sum_k n_k (mu_k - xbar)(mu_k - xbar)'
        M = self.means_ - self.xbar_
        Sb = (M * counts[:, None]).T @ M
        self.Sw_, self.Sb_ = Sw, Sb
        self.covariance_ = Sw / (n - K)

        # Whitening: S_W/(n-K) = V diag(s) V'   ->   W = V diag(s^-1/2)
        s, V = np.linalg.eigh(self.covariance_)
        tol = s.max() * p * np.finfo(float).eps
        keep = s > tol
        if not keep.all():
            # Rank-deficient (collinear features): drop null directions.
            s, V = s[keep], V[:, keep]
        self._whiten = V / np.sqrt(s)          # (p, r_w)

        # In whitened space the problem is a plain symmetric eigenproblem.
        Sb_w = self._whiten.T @ (Sb / (n - K)) @ self._whiten
        evals, U = np.linalg.eigh(Sb_w)
        order = np.argsort(evals)[::-1]
        evals, U = evals[order], U[:, order]

        max_comp = min(K - 1, p)
        r = max_comp if self.n_components is None else self.n_components
        if r > max_comp:
            raise ValueError(f"n_components cannot exceed min(n_classes-1, n_features)={max_comp}.")

        self.eigenvalues_ = np.clip(evals[:max_comp], 0, None)
        self.explained_variance_ratio_ = (self.eigenvalues_ / self.eigenvalues_.sum())[:r]
        self.scalings_ = self._whiten @ U[:, :r]
        self.scalings_ = self._fix_signs(self.scalings_)

        # Classifier parameters: delta_k(x) = x'a_k + b_k
        coef = np.linalg.solve(self._cov_regularised(), self.means_.T).T   # (K, p)
        self.coef_ = coef
        self.intercept_ = -0.5 * np.sum(coef * self.means_, axis=1) + np.log(self.priors_)
        return self

    def _cov_regularised(self):
        # Tiny ridge keeps the solve stable if features are almost collinear.
        p = self.covariance_.shape[0]
        return self.covariance_ + 1e-12 * np.trace(self.covariance_) / p * np.eye(p)

    @staticmethod
    def _fix_signs(W):
        """Eigenvectors are sign-ambiguous; make the largest loading positive."""
        W = W.copy()
        for j in range(W.shape[1]):
            if W[np.argmax(np.abs(W[:, j])), j] < 0:
                W[:, j] *= -1
        return W

    # ------------------------------------------------------------ inference
    def _check_fitted(self):
        if not hasattr(self, "scalings_"):
            raise RuntimeError("Call fit() first.")

    def transform(self, X):
        """Project onto the discriminant axes (LD1, LD2, ...)."""
        self._check_fitted()
        return (np.asarray(X, dtype=float) - self.xbar_) @ self.scalings_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def decision_function(self, X):
        """Linear discriminant scores delta_k(x), shape (n, K)."""
        self._check_fitted()
        return np.asarray(X, dtype=float) @ self.coef_.T + self.intercept_

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores -= scores.max(axis=1, keepdims=True)    # numerical stability
        e = np.exp(scores)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))
