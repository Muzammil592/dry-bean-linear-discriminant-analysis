"""Day 3: model benchmarking utilities.

Everything here is leakage-safe: scaling, PCA and LDA are fitted INSIDE each
cross-validation fold through scikit-learn pipelines.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .lda import FisherLDA

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "roc_auc_ovr": "roc_auc_ovr",
}


# --------------------------------------------------------------------------
def get_models(random_state: int = 0, include: list[str] | None = None) -> dict:
    """The model zoo used in the benchmark (each wrapped with standardisation where useful)."""
    models = {
        "LDA (ours)": make_pipeline(StandardScaler(), FisherLDA()),
        "LDA (sklearn)": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "Shrinkage LDA": make_pipeline(
            StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        ),
        "QDA": make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=1e-3)),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=random_state)
        ),
        "SVM (RBF)": make_pipeline(
            StandardScaler(), SVC(probability=True, random_state=random_state)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=random_state
        ),
    }
    if include is not None:
        models = {k: models[k] for k in include}
    return models


# --------------------------------------------------------------------------
def cross_validate_models(X, y, models: dict | None = None, n_splits: int = 5,
                          n_repeats: int = 1, random_state: int = 42):
    """Stratified (repeated) k-fold benchmark.

    Returns
    -------
    summary : DataFrame, one row per model, mean/std of each metric + fit/predict time (seconds)
    raw     : dict model -> DataFrame of per-fold scores (for paired comparisons / boxplots)
    """
    models = models or get_models()
    cv = (StratifiedKFold(n_splits, shuffle=True, random_state=random_state) if n_repeats == 1
          else RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=random_state))
    rows, raw = {}, {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = cross_validate(clone(model), X, y, cv=cv, scoring=SCORING, n_jobs=1)
        df = pd.DataFrame({k.replace("test_", ""): v for k, v in res.items()})
        raw[name] = df
        row = {}
        for m in list(SCORING) + ["fit_time", "score_time"]:
            row[f"{m}_mean"], row[f"{m}_std"] = df[m].mean(), df[m].std()
        rows[name] = row
    summary = pd.DataFrame(rows).T
    return summary.sort_values("accuracy_mean", ascending=False), raw


# --------------------------------------------------------------------------
def oof_proba(model, X, y, n_splits: int = 5, random_state: int = 42):
    """Out-of-fold class probabilities (every sample predicted by a model that never saw it)."""
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba = cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")
    classes = np.unique(np.asarray(y))
    return proba, classes


def confusion_from_proba(y, proba, classes, normalize: bool = True):
    pred = classes[np.argmax(proba, axis=1)]
    cm = confusion_matrix(np.asarray(y), pred, labels=classes)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
    return pd.DataFrame(cm, index=classes, columns=classes)


def top_confusions(cm_normalized: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Largest off-diagonal entries of a row-normalised confusion matrix."""
    rows = [{"true": i, "predicted": j, "rate": cm_normalized.loc[i, j]}
            for i in cm_normalized.index for j in cm_normalized.columns if i != j]
    return pd.DataFrame(rows).sort_values("rate", ascending=False).head(k).reset_index(drop=True)


def calibration_table(y, proba, classes, n_bins: int = 10):
    """Top-label reliability table and Expected Calibration Error (ECE).

    For each bin of predicted confidence, compare mean confidence with actual accuracy.
    """
    y = np.asarray(y)
    conf = proba.max(axis=1)
    correct = (classes[np.argmax(proba, axis=1)] == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "confidence": conf[m].mean(), "accuracy": correct[m].mean(),
                         "count": int(m.sum())})
    tab = pd.DataFrame(rows)
    ece = float(np.sum(tab["count"] / tab["count"].sum() * (tab["confidence"] - tab["accuracy"]).abs()))
    tab.attrs["ece"] = ece
    return tab


# --------------------------------------------------------------------------
def pca_vs_lda_downstream(X, y, ks=None, n_splits: int = 5, random_state: int = 42,
                          classifier=None) -> pd.DataFrame:
    """Accuracy of a downstream classifier after reducing to k dimensions with PCA vs LDA.

    The reducer is fitted inside each CV fold (no leakage). LDA is capped at K-1 axes.
    """
    X = np.asarray(X, dtype=float)
    K = len(np.unique(y))
    max_k = min(K - 1, X.shape[1])
    ks = ks or list(range(1, max_k + 1))
    classifier = classifier or LogisticRegression(max_iter=2000)
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    rows = []
    for k in ks:
        for name, reducer in (("PCA", PCA(n_components=k)), ("LDA", FisherLDA(n_components=k))):
            pipe = make_pipeline(StandardScaler(), reducer, clone(classifier))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
            rows.append({"k": k, "reducer": name,
                         "accuracy": res["test_accuracy"].mean(), "accuracy_std": res["test_accuracy"].std(),
                         "f1_macro": res["test_f1_macro"].mean()})
    return pd.DataFrame(rows)


def embed_2d(X, y):
    """2-D PCA and LDA coordinates for visual comparison (fitted on all data, for plotting only)."""
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    return {"PCA": PCA(n_components=2).fit_transform(Xs),
            "LDA": FisherLDA(n_components=2).fit_transform(Xs, y)}


# --------------------------------------------------------------------------
def small_sample_study(X, y, sizes=(10, 20, 50, 100, 500), models: dict | None = None,
                       n_repeats: int = 10, test_size: float = 0.3, random_state: int = 0) -> pd.DataFrame:
    """Accuracy / macro-F1 vs number of TRAINING samples per class.

    Each repeat: stratified hold-out test set, then n samples per class (capped at what
    exists) are drawn from the remaining pool to train every model.
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y)
    if models is None:
        models = get_models(include=["LDA (sklearn)", "Shrinkage LDA", "QDA",
                                     "Logistic Regression", "Random Forest"])
    rng = np.random.default_rng(random_state)
    rows = []
    for rep in range(n_repeats):
        X_pool, X_te, y_pool, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=int(rng.integers(1e9)))
        for n in sizes:
            idx = np.concatenate([
                rng.choice(np.where(y_pool == c)[0], size=min(n, int(np.sum(y_pool == c))), replace=False)
                for c in np.unique(y_pool)])
            for name, model in models.items():
                m = clone(model)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m.fit(X_pool[idx], y_pool[idx])
                        pred = m.predict(X_te)
                    acc, f1 = accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro")
                except Exception:
                    acc = f1 = np.nan          # e.g. singular covariance with tiny n
                rows.append({"repeat": rep, "n_per_class": n, "model": name,
                             "accuracy": acc, "f1_macro": f1})
    return pd.DataFrame(rows)


def summarise_small_sample(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "n_per_class"])[["accuracy", "f1_macro"]]
    out = g.mean().join(g.std(), rsuffix="_std")
    return out.reset_index()
