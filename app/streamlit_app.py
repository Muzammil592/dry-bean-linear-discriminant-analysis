"""Interactive demo: classify a dry bean with Fisher's LDA and see where it lands in discriminant space.

Run locally:   streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from fisher_lda.app_utils import input_warnings, load_bundle, predict_point, preset_values

BUNDLE_PATH = ROOT / "app" / "model_bundle.joblib"

st.set_page_config(page_title="Dry Bean LDA", page_icon="🫘", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle(BUNDLE_PATH)


if not BUNDLE_PATH.exists():
    st.error("Model bundle not found. Run `python scripts/train_and_export.py` first.")
    st.stop()

bundle = get_bundle()
names, classes, stats = bundle["feature_names"], bundle["classes"], bundle["feature_stats"]

st.title("🫘 Dry Bean Classifier")
st.caption("Fisher's Linear Discriminant Analysis, implemented from scratch in NumPy. "
           "Set the bean's measurements, see the predicted variety and where the bean sits among 7 classes.")


# ---------------------------------------------------------------- sidebar: inputs
def apply_preset():
    vals = preset_values(bundle, st.session_state["preset"])
    for n in names:
        st.session_state[f"f_{n}"] = vals[n]


options = ["median"] + classes
st.sidebar.header("Bean measurements")
st.sidebar.selectbox("Start from a typical bean of…", options, key="preset", on_change=apply_preset,
                     format_func=lambda o: "Dataset median" if o == "median" else o)

default = preset_values(bundle, "median")
values = {}
for n in names:
    lo = float(min(stats.loc[n, "p1"], bundle["class_medians"][n].min()))
    hi = float(max(stats.loc[n, "p99"], bundle["class_medians"][n].max()))
    pad = (hi - lo) * 0.05 or 1.0
    lo, hi = lo - pad, hi + pad
    key = f"f_{n}"
    if key not in st.session_state:
        st.session_state[key] = float(np.clip(default[n], lo, hi))
    values[n] = st.sidebar.slider(n, lo, hi, key=key, step=(hi - lo) / 500, format="%.4g")

# ---------------------------------------------------------------- prediction
res = predict_point(bundle, values)
warns = input_warnings(bundle, values)

left, right = st.columns([1, 1.6])
with left:
    st.subheader("Prediction")
    st.metric("Predicted variety", res["label"], f"{res['proba'].iloc[0]:.1%} confidence")
    st.bar_chart(res["proba"].rename("probability"))
    for w in warns:
        st.warning(w)
    st.caption("Probabilities come from LDA's Gaussian model; they are informative but can be "
               "over-confident when the data are not truly Gaussian (see the project's calibration analysis).")

with right:
    st.subheader("Where the bean lands in discriminant space")
    ld_cols = list(res["ld"].index)
    c1, c2 = st.columns(2)
    ax_x = c1.selectbox("x-axis", ld_cols, index=0)
    ax_y = c2.selectbox("y-axis", ld_cols, index=1)
    evr = bundle["explained_variance_ratio"]
    bg = bundle["background"]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    sns.scatterplot(data=bg, x=ax_x, y=ax_y, hue="class", hue_order=classes, s=10, alpha=.45,
                    linewidth=0, ax=ax)
    ax.scatter(res["ld"][ax_x], res["ld"][ax_y], marker="*", s=420, color="black",
               edgecolor="white", linewidth=1.2, zorder=5, label="your bean")
    ix, iy = ld_cols.index(ax_x), ld_cols.index(ax_y)
    ax.set_xlabel(f"{ax_x} ({evr[ix]:.0%} of separation)")
    ax.set_ylabel(f"{ax_y} ({evr[iy]:.0%} of separation)")
    ax.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    st.pyplot(fig, clear_figure=True)

st.subheader("Why here? Feature contributions")
axis_choice = st.radio("Discriminant axis", ld_cols, horizontal=True)
contrib = res["contributions"][axis_choice].sort_values(key=np.abs, ascending=False).head(8)
st.bar_chart(contrib.rename("contribution to " + axis_choice))
st.caption("Each bar is (feature value − training mean) × coefficient. The bars of all 16 features "
           "add up to the bean's coordinate on this axis.")

with st.expander("About this model"):
    st.markdown(f"""
- **Model:** Fisher's LDA, fitted on {bundle['n_train']:,} beans (all data; cross-validated performance is reported in the repository README).
- **Axes:** with 7 classes there are up to 6 discriminant axes; the first two carry
  {evr[:2].sum():.0%} of the class separation.
- **Data:** UCI Dry Bean dataset, Koklu & Ozkan (2020), *Computers and Electronics in Agriculture* 174, 105507.
- **Limits:** inputs outside the training range are extrapolations; the demo does not check that all 16 features are mutually consistent.
""")
