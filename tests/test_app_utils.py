import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

from fisher_lda import app_utils as U

d = load_wine(as_frame=True)
X, y = d.data, np.array(["c" + str(i) for i in d.target])


def _bundle():
    return U.build_bundle(X, y, n_background=60)


def test_bundle_contents():
    b = _bundle()
    assert b["feature_names"] == list(X.columns) and b["classes"] == ["c0", "c1", "c2"]
    assert b["background"].shape[1] == 2 + 1 and len(b["background"]) <= 70
    assert set(b["background"]["class"]) == set(b["classes"])
    assert b["std_coefficients"].shape == (13, 2) and b["n_train"] == 178


def test_predict_point_consistent_with_model():
    b = _bundle()
    row = X.iloc[10].to_dict()
    r = U.predict_point(b, row)
    lda = b["lda"]
    assert r["label"] == lda.predict(X.iloc[[10]].to_numpy())[0]
    assert np.isclose(r["proba"].sum(), 1)
    np.testing.assert_allclose(r["ld"].to_numpy(), lda.transform(X.iloc[[10]].to_numpy())[0])
    # contributions add up to the LD coordinates
    np.testing.assert_allclose(r["contributions"].sum().to_numpy(), r["ld"].to_numpy(), atol=1e-8)


def test_presets():
    b = _bundle()
    med = U.preset_values(b, "median")
    assert set(med) == set(X.columns) and np.isclose(med["alcohol"], X["alcohol"].median())
    c1 = U.preset_values(b, "c1")
    assert np.isclose(c1["alcohol"], X.loc[y == "c1", "alcohol"].median())
    # a typical bean of a class should usually be predicted as that class
    assert all(U.predict_point(b, U.preset_values(b, c))["label"] == c for c in b["classes"])


def test_input_warnings():
    b = _bundle()
    assert U.input_warnings(b, U.preset_values(b, "median")) == []
    bad = U.preset_values(b, "median"); bad["alcohol"] = 999.0
    assert any("alcohol" in m for m in U.input_warnings(b, bad))


def test_geometry_warnings_on_bean_like_columns():
    rng = np.random.default_rng(0)
    n = 90
    major = rng.uniform(200, 300, n)
    Xb = pd.DataFrame({"MajorAxisLength": major, "MinorAxisLength": major * rng.uniform(.5, .9, n),
                       "Area": rng.uniform(3e4, 9e4, n), "ConvexArea": rng.uniform(9e4, 1e5, n),
                       "Solidity": rng.uniform(.9, .99, n), "Eccentricity": rng.uniform(.4, .9, n)})
    yb = np.repeat(["a", "b", "c"], 30)
    b = U.build_bundle(Xb, yb, n_background=30)
    v = U.preset_values(b, "median")
    assert U.input_warnings(b, v) == []
    v.update(MinorAxisLength=v["MajorAxisLength"] + 1, Solidity=1.2, Eccentricity=1.5)
    assert len(U.input_warnings(b, v)) >= 3


def test_save_load_roundtrip():
    import tempfile, os
    b = _bundle()
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "sub", "bundle.joblib")
        U.save_bundle(b, p)
        b2 = U.load_bundle(p)
    row = X.iloc[0].to_dict()
    assert U.predict_point(b, row)["label"] == U.predict_point(b2, row)["label"]
