"""Plot helpers for the Day 3 benchmark."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_model_comparison(summary: pd.DataFrame, metric: str = "accuracy", ax=None):
    ax = ax or plt.gca()
    d = summary.sort_values(f"{metric}_mean")
    ax.barh(d.index, d[f"{metric}_mean"], xerr=d[f"{metric}_std"], color="steelblue", capsize=3)
    lo = max(0, d[f"{metric}_mean"].min() - 0.05)
    ax.set_xlim(lo, min(1.0, d[f"{metric}_mean"].max() + 0.03))
    for y, (v, s) in enumerate(zip(d[f"{metric}_mean"], d[f"{metric}_std"])):
        ax.text(v + s + 0.001, y, f"{v:.3f}", va="center", fontsize=9)
    ax.set_title(f"Cross-validated {metric.replace('_', ' ')} (mean ± std over folds)")
    return ax


def plot_timing(summary: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    d = summary.sort_values("fit_time_mean")
    ax.barh(d.index, d["fit_time_mean"], color="darkorange")
    ax.set_xscale("log"); ax.set_xlabel("fit time per fold (s, log scale)")
    ax.set_title("Training cost")
    return ax


def plot_confusion_matrix(cm: pd.DataFrame, title: str = "", ax=None):
    ax = ax or plt.gca()
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    return ax


def plot_pca_vs_lda_2d(emb: dict, y, figsize=(13, 5.5)):
    y = np.asarray(y)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, name in zip(axes, ("PCA", "LDA")):
        Z = emb[name]
        for c in np.unique(y):
            m = y == c
            ax.scatter(Z[m, 0], Z[m, 1], s=5, alpha=.5, label=c)
        ax.set_xlabel(f"{name}1"); ax.set_ylabel(f"{name}2")
        ax.set_title(f"{name}: " + ("maximises variance (ignores labels)" if name == "PCA"
                                     else "maximises class separation"))
    axes[1].legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    return fig


def plot_pca_vs_lda_curve(df: pd.DataFrame, ax=None):
    ax = ax or plt.gca()
    for name, g in df.groupby("reducer"):
        ax.errorbar(g["k"], g["accuracy"], yerr=g["accuracy_std"], marker="o", capsize=3, label=name)
    ax.set_xlabel("number of components kept"); ax.set_ylabel("CV accuracy")
    ax.set_title("Downstream accuracy after PCA vs LDA reduction"); ax.legend()
    return ax


def plot_small_sample(summary: pd.DataFrame, metric: str = "accuracy", ax=None):
    ax = ax or plt.gca()
    for name, g in summary.groupby("model"):
        g = g.sort_values("n_per_class")
        ax.plot(g["n_per_class"], g[metric], marker="o", label=name)
        ax.fill_between(g["n_per_class"], g[metric] - g[f"{metric}_std"].fillna(0),
                        g[metric] + g[f"{metric}_std"].fillna(0), alpha=.12)
    ax.set_xscale("log"); ax.set_xlabel("training samples per class (log)")
    ax.set_ylabel(f"test {metric.replace('_', ' ')}")
    ax.set_title("Small-sample behaviour"); ax.legend()
    return ax


def plot_calibration(tables: dict, ax=None):
    """Reliability diagram for several models. `tables` maps model name -> calibration_table()."""
    ax = ax or plt.gca()
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, t in tables.items():
        ax.plot(t["confidence"], t["accuracy"], marker="o", label=f"{name} (ECE={t.attrs['ece']:.3f})")
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("observed accuracy")
    ax.set_title("Calibration (top-label reliability)"); ax.legend(fontsize=8)
    return ax
