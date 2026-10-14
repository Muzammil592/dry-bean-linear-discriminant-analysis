import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fisher_lda import FisherLDA, benchmark as B

X, y = load_wine(return_X_y=True)
FAST = ["LDA (ours)", "LDA (sklearn)", "Shrinkage LDA", "QDA", "Logistic Regression"]


def test_fisherlda_is_sklearn_compatible():
    m = clone(FisherLDA(n_components=2))
    assert m.get_params()["n_components"] == 2
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    ours = cross_val_score(make_pipeline(StandardScaler(), FisherLDA()), X, y, cv=cv)
    ref = cross_val_score(make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()), X, y, cv=cv)
    np.testing.assert_allclose(ours, ref)


def test_cross_validate_models_shape_and_ranges():
    summary, raw = B.cross_validate_models(X, y, B.get_models(include=FAST), n_splits=3)
    assert set(summary.index) == set(FAST) and set(raw) == set(FAST)
    for m in ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr"]:
        assert summary[f"{m}_mean"].between(0, 1).all()
    assert (summary["fit_time_mean"] > 0).all()
    # our LDA must score the same as sklearn's
    assert np.isclose(summary.loc["LDA (ours)", "accuracy_mean"], summary.loc["LDA (sklearn)", "accuracy_mean"])


def test_oof_proba_confusion_and_calibration():
    model = B.get_models(include=["LDA (ours)"])["LDA (ours)"]
    proba, classes = B.oof_proba(model, X, y)
    assert proba.shape == (len(y), 3) and np.allclose(proba.sum(1), 1)
    cm = B.confusion_from_proba(y, proba, classes)
    assert np.allclose(cm.sum(1), 1)
    assert len(B.top_confusions(cm, 3)) == 3
    tab = B.calibration_table(y, proba, classes)
    assert tab["count"].sum() == len(y) and 0 <= tab.attrs["ece"] <= 1


def test_calibration_table_perfectly_calibrated_has_low_ece():
    rng = np.random.default_rng(0)
    n = 20000
    conf = rng.uniform(0.5, 1.0, n)
    correct = rng.uniform(size=n) < conf
    classes = np.array([0, 1])
    pred = np.where(correct, 1, 0)                      # true label is always 1
    proba = np.c_[np.where(pred == 0, conf, 1 - conf), np.where(pred == 1, conf, 1 - conf)]
    tab = B.calibration_table(np.ones(n, dtype=int), proba, classes)
    assert tab.attrs["ece"] < 0.02


def test_pca_vs_lda_downstream():
    df = B.pca_vs_lda_downstream(X, y, n_splits=3)
    assert set(df["reducer"]) == {"PCA", "LDA"} and df["k"].max() == 2
    # supervised reduction should beat unsupervised at 1 component on Wine
    a = df[df.k == 1].set_index("reducer")["accuracy"]
    assert a["LDA"] > a["PCA"]


def test_embed_2d():
    emb = B.embed_2d(X, y)
    assert emb["PCA"].shape == emb["LDA"].shape == (len(y), 2)


def test_small_sample_study_and_summary():
    models = B.get_models(include=["LDA (sklearn)", "Shrinkage LDA"])
    df = B.small_sample_study(X, y, sizes=(5, 1000), models=models, n_repeats=2)
    assert set(df["n_per_class"]) == {5, 1000} and len(df) == 2 * 2 * 2
    s = B.summarise_small_sample(df)
    assert {"model", "n_per_class", "accuracy", "accuracy_std"} <= set(s.columns)
