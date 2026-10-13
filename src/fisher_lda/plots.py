"""Plot helpers for Day 2 EDA and assumption diagnostics."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .assumptions import mahalanobis_qq


def plot_class_balance(y, ax=None):
    ax = ax or plt.gca()
    counts = pd.Series(y).value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title("Class balance"); ax.set_xlabel(""); ax.set_ylabel("samples")
    return ax


def plot_correlation_heatmap(X, ax=None):
    ax = ax or plt.gca()
    sns.heatmap(pd.DataFrame(X).corr(), cmap="coolwarm", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": .7}, ax=ax)
    ax.set_title("Feature correlation")
    return ax


def plot_covariance_heatmaps(X, y, ncols: int = 4, standardise: bool = True):
    """One covariance (or correlation) heatmap per class on a shared colour scale.

    Visually similar panels support the equal-covariance assumption.
    """
    X, y = pd.DataFrame(X), np.asarray(y)
    classes = np.unique(y)
    nrows = int(np.ceil(len(classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    Z = (X - X.mean()) / X.std() if standardise else X
    mats = {c: Z[y == c].cov() for c in classes}
    vmax = max(np.abs(m.values).max() for m in mats.values())
    for ax, c in zip(axes, classes):
        sns.heatmap(mats[c], cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax,
                    cbar=False, xticklabels=False, yticklabels=False, ax=ax)
        ax.set_title(str(c))
    for ax in axes[len(classes):]:
        ax.axis("off")
    fig.suptitle("Class covariance matrices (standardised features, shared scale)")
    fig.tight_layout()
    return fig


def plot_feature_distributions(X, y, features, ncols: int = 3):
    """Per-class KDE of selected features (visual normality / spread check)."""
    X = pd.DataFrame(X).copy(); X["class"] = np.asarray(y)
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, features):
        sns.kdeplot(data=X, x=f, hue="class", common_norm=False, ax=ax, legend=(ax is axes[0]))
    for ax in axes[len(features):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_qq_grid(X, y, feature, ncols: int = 4):
    """Q-Q plot of one feature for every class."""
    X, y = pd.DataFrame(X), np.asarray(y)
    classes = np.unique(y)
    nrows = int(np.ceil(len(classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, classes):
        stats.probplot(X.loc[y == c, feature], dist="norm", plot=ax)
        ax.set_title(str(c)); ax.set_xlabel(""); ax.set_ylabel("")
    for ax in axes[len(classes):]:
        ax.axis("off")
    fig.suptitle(f"Normal Q-Q plots of '{feature}' by class")
    fig.tight_layout()
    return fig


def plot_mahalanobis_qq(X, y, ncols: int = 4):
    """Multivariate normality: squared Mahalanobis distance vs chi-square quantiles."""
    classes = np.unique(np.asarray(y))
    nrows = int(np.ceil(len(classes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, classes):
        theo, obs = mahalanobis_qq(X, y, c)
        ax.scatter(theo, obs, s=4, alpha=.5)
        lim = [0, max(theo.max(), obs.max())]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_title(str(c))
    for ax in axes[len(classes):]:
        ax.axis("off")
    fig.suptitle("Multivariate normality: Mahalanobis d² vs χ² quantiles (points on red line = normal)")
    fig.tight_layout()
    return fig


def plot_vif(vif_df, ax=None):
    ax = ax or plt.gca()
    sns.barplot(data=vif_df, y="feature", x="VIF", ax=ax, color="steelblue")
    ax.set_xscale("log")
    ax.axvline(5, color="orange", ls="--", label="VIF = 5")
    ax.axvline(10, color="red", ls="--", label="VIF = 10")
    ax.legend(); ax.set_title("Variance Inflation Factor (log scale)")
    return ax
