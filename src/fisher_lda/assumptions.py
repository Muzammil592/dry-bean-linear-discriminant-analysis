"""Statistical checks for the assumptions behind LDA.

LDA assumes that, within each class,
  1. features are (multivariate) normal,
  2. all classes share the SAME covariance matrix,
and it works best when features are not (almost) perfectly collinear.

Every function returns a pandas object or a plain dict so results are easy to
display, save, and quote in the README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _as_frame(X) -> pd.DataFrame:
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=float))


# --------------------------------------------------------------------------
# 1. Normality inside each class
# --------------------------------------------------------------------------
def univariate_normality(X, y, max_samples: int = 2000, alpha: float = 0.05,
                         random_state: int = 0) -> pd.DataFrame:
    """Shapiro-Wilk, skewness and excess kurtosis for every (class, feature).

    Shapiro-Wilk becomes over-sensitive on large samples, so each group is
    subsampled to `max_samples`. Read skew/kurtosis alongside the p-value:
    |skew| < 1 and |excess kurtosis| < 2 is usually "close enough" for LDA.
    """
    X, y = _as_frame(X), np.asarray(y)
    rng = np.random.default_rng(random_state)
    rows = []
    for c in np.unique(y):
        Xc = X[y == c]
        for col in X.columns:
            v = Xc[col].to_numpy()
            s = v if len(v) <= max_samples else rng.choice(v, max_samples, replace=False)
            w, p = stats.shapiro(s)
            rows.append({"class": c, "feature": col, "n": len(v),
                         "shapiro_W": w, "p_value": p, "normal": p > alpha,
                         "skew": stats.skew(v), "excess_kurtosis": stats.kurtosis(v)})
    return pd.DataFrame(rows)


def normality_summary(norm_df: pd.DataFrame) -> pd.DataFrame:
    """Share of features passing Shapiro-Wilk, and mean |skew|, per class."""
    g = norm_df.groupby("class")
    return pd.DataFrame({
        "pct_features_normal": g["normal"].mean() * 100,
        "mean_abs_skew": g["skew"].apply(lambda s: s.abs().mean()),
        "mean_abs_excess_kurtosis": g["excess_kurtosis"].apply(lambda s: s.abs().mean()),
    }).round(3)


def multivariate_normality(X, y) -> pd.DataFrame:
    """Mardia's kurtosis test per class (and Mahalanobis QQ data via `mahalanobis_qq`).

    Under multivariate normality the mean of squared Mahalanobis distances raised to
    the 2nd power (b2) is approximately p(p+2) with variance 8p(p+2)/n.
    """
    X, y = _as_frame(X), np.asarray(y)
    p = X.shape[1]
    rows = []
    for c in np.unique(y):
        Z = X[y == c].to_numpy()
        n = len(Z)
        d2 = _mahalanobis_sq(Z)
        b2 = np.mean(d2 ** 2)
        z = (b2 - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
        rows.append({"class": c, "n": n, "mardia_b2": b2, "expected_b2": p * (p + 2),
                     "z": z, "p_value": 2 * (1 - stats.norm.cdf(abs(z)))})
    return pd.DataFrame(rows).set_index("class")


def _mahalanobis_sq(Z: np.ndarray) -> np.ndarray:
    D = Z - Z.mean(axis=0)
    S = np.cov(Z, rowvar=False)
    return np.einsum("ij,jk,ik->i", D, np.linalg.pinv(S), D)


def mahalanobis_qq(X, y, cls):
    """(theoretical, observed) quantiles of squared Mahalanobis distance vs chi2(p)."""
    X, y = _as_frame(X), np.asarray(y)
    Z = X[y == cls].to_numpy()
    d2 = np.sort(_mahalanobis_sq(Z))
    n, p = Z.shape
    theo = stats.chi2.ppf((np.arange(1, n + 1) - 0.5) / n, df=p)
    return theo, d2


# --------------------------------------------------------------------------
# 2. Equal covariance across classes
# --------------------------------------------------------------------------
def box_m_test(X, y) -> dict:
    """Box's M test for equality of covariance matrices.

    H0: all class covariance matrices are equal.
    Warning: Box's M is extremely sensitive to non-normality and large n.
    With thousands of samples it will reject almost always, so ALSO inspect
    `covariance_ratio` and the covariance heatmaps before drawing conclusions.
    """
    X, y = _as_frame(X), np.asarray(y)
    classes = np.unique(y)
    g, p, N = len(classes), X.shape[1], len(X)
    covs, ns = [], []
    for c in classes:
        Z = X[y == c].to_numpy()
        covs.append(np.cov(Z, rowvar=False))
        ns.append(len(Z))
    dfs = np.array(ns) - 1
    pooled = sum(d * S for d, S in zip(dfs, covs)) / (N - g)
    logdet = lambda S: np.linalg.slogdet(S)[1]
    M = (N - g) * logdet(pooled) - sum(d * logdet(S) for d, S in zip(dfs, covs))
    c1 = (np.sum(1.0 / dfs) - 1.0 / (N - g)) * (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))
    chi2 = M * (1 - c1)
    df = (g - 1) * p * (p + 1) / 2
    return {"M": M, "chi2": chi2, "df": df, "p_value": stats.chi2.sf(chi2, df),
            "reject_equal_cov_at_5pct": stats.chi2.sf(chi2, df) < 0.05}


def covariance_ratio(X, y) -> pd.DataFrame:
    """Per-class log-determinant and Frobenius distance from the pooled covariance.

    A scale-free, sample-size-free way to see HOW different the classes are.
    `relative_distance` of 0 means identical to pooled; values > ~0.5 are large.
    """
    X, y = _as_frame(X), np.asarray(y)
    classes = np.unique(y)
    covs = {c: np.cov(X[y == c].to_numpy(), rowvar=False) for c in classes}
    N, g = len(X), len(classes)
    pooled = sum((np.sum(y == c) - 1) * covs[c] for c in classes) / (N - g)
    rows = []
    for c in classes:
        rows.append({"class": c,
                     "log_det": np.linalg.slogdet(covs[c])[1],
                     "log_det_minus_pooled": np.linalg.slogdet(covs[c])[1] - np.linalg.slogdet(pooled)[1],
                     "relative_distance": np.linalg.norm(covs[c] - pooled) / np.linalg.norm(pooled)})
    return pd.DataFrame(rows).set_index("class").round(3)


# --------------------------------------------------------------------------
# 3. Multicollinearity
# --------------------------------------------------------------------------
def vif_table(X) -> pd.DataFrame:
    """Variance Inflation Factor per feature (diag of inverse correlation matrix).

    Rule of thumb: VIF > 5 = notable, > 10 = severe.
    """
    X = _as_frame(X)
    corr = np.corrcoef(X.to_numpy(), rowvar=False)
    vif = np.diag(np.linalg.pinv(corr))
    return (pd.DataFrame({"feature": X.columns, "VIF": vif})
            .sort_values("VIF", ascending=False).reset_index(drop=True))


def high_correlation_pairs(X, threshold: float = 0.9) -> pd.DataFrame:
    X = _as_frame(X)
    corr = X.corr()
    iu = np.triu_indices_from(corr, k=1)
    pairs = pd.DataFrame({"feature_a": corr.columns[iu[0]], "feature_b": corr.columns[iu[1]],
                          "correlation": corr.values[iu]})
    return (pairs[pairs["correlation"].abs() >= threshold]
            .sort_values("correlation", key=np.abs, ascending=False).reset_index(drop=True))


def condition_number(X) -> float:
    """Condition number of the standardised feature matrix (> 30 suggests collinearity)."""
    Z = _as_frame(X).to_numpy(dtype=float)
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    return float(np.linalg.cond(Z))


# --------------------------------------------------------------------------
# 4. One-call report
# --------------------------------------------------------------------------
def assumption_report(X, y, max_samples: int = 2000) -> dict:
    """Run every check and return a dict of results, ready to print or save."""
    norm = univariate_normality(X, y, max_samples=max_samples)
    return {
        "univariate_normality": norm,
        "normality_summary": normality_summary(norm),
        "multivariate_normality": multivariate_normality(X, y),
        "box_m": box_m_test(X, y),
        "covariance_ratio": covariance_ratio(X, y),
        "vif": vif_table(X),
        "high_corr_pairs": high_correlation_pairs(X),
        "condition_number": condition_number(X),
    }
