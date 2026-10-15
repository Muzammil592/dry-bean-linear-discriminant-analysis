# Fisher's Linear Discriminant Analysis from Scratch
### Classifying Dry Bean Varieties with a NumPy Implementation, Assumption Diagnostics, and Model Benchmarks

> Can seven bean varieties be sorted automatically from geometric measurements alone, and how far can a
> classic linear method go against modern classifiers?

**Status:** Day 4 of 5 complete (implementation, diagnostics, benchmark, interpretation, demo app). See the [roadmap](#roadmap).

---

## Why this project
Linear Discriminant Analysis is both a **classifier** and a **supervised dimensionality-reduction** technique,
yet it is usually used as a black box. This project builds it from first principles, verifies it against
scikit-learn, tests whether its assumptions actually hold on real data, and benchmarks it honestly against
QDA, logistic regression, SVM and random forest.

**Dataset:** [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), 13,611 beans,
16 shape features extracted from images, 7 classes. Real-world framing: automated seed quality sorting.

## Highlights
**implementation**
- Pure-NumPy `FisherLDA` with `fit`, `transform`, `predict`, `predict_proba`, `decision_function`
- Numerically stable solver (whitening + symmetric eigendecomposition, no explicit matrix inverse)
- Verified against `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`: projections, predictions,
  probabilities and explained variance agree to ~1e-6
- Property tests: Fisher criterion is maximised, discriminant axes are uncorrelated

**assumption diagnostics**
- Reusable `fisher_lda.assumptions` module: per-class Shapiro-Wilk, skewness and kurtosis, Mardia's
  multivariate normality test, Box's M test, covariance-distance metric, VIF and condition number
- Diagnostic plots: per-class Q-Q grids, Mahalanobis chi-square plots, covariance heatmaps, correlation heatmap
- Statistical tests are validated against synthetic data with known ground truth (equal vs unequal covariance,
  normal vs skewed, collinear vs independent)
- Remedies (power transform, dropping collinear features) compared before and after

**benchmark**
- `FisherLDA` is scikit-learn compatible (`clone`, `Pipeline`, `cross_val_score`), so it is benchmarked side by side
  with the library implementation
- Leakage-safe stratified k-fold comparison of LDA, shrinkage LDA, QDA, logistic regression, SVM and random forest
  on accuracy, balanced accuracy, macro-F1, one-vs-rest ROC-AUC and training time
- Out-of-fold confusion matrices and a calibration analysis (reliability curves, Expected Calibration Error)
- LDA vs PCA as dimensionality reducers (2-D projections and downstream accuracy)
- Small-sample study: accuracy vs training samples per class

**interpretation and demo**
- Feature effects in within-class SD units, structure correlations, importance scores, bootstrap stability of coefficients
- Biplot, 2-D decision regions, and class-to-class Mahalanobis distances that explain which beans get confused
- Streamlit app: enter 16 measurements, get the predicted variety, class probabilities, the bean's position in
  discriminant space and the features that put it there
- App logic is separated from the UI and unit-tested; the exported model is a ~100 KB file, so deployment needs no dataset

## Method in 60 seconds
Fisher's criterion looks for directions `w` that maximise between-class scatter relative to within-class scatter:

```
J(w) = (wᵀ S_B w) / (wᵀ S_W w)     →     S_B w = λ S_W w
```

With K classes there are at most `min(K-1, p)` useful axes (6 here). The same model also yields a Bayes
classifier under the assumption of Gaussian classes with a shared covariance, which is exactly what Day 2 tests.

## Quick start
```bash
git clone https://github.com/Muzammil592/dry-bean-lda.git
cd dry-bean-lda
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
# download the data, see data/README.md
pytest

# demo app
python scripts/train_and_export.py          # builds app/model_bundle.joblib from the dataset
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

```python
from fisher_lda import FisherLDA, load_dry_bean, assumptions

X, y = load_dry_bean("data/raw/Dry_Bean_Dataset.xlsx")

lda = FisherLDA().fit(X, y)
Z = lda.transform(X)                       # (n, 6) discriminant coordinates

from fisher_lda import benchmark
summary, raw = benchmark.cross_validate_models(X, y)      # 7 models, stratified 5-fold CV
summary[["accuracy_mean", "f1_macro_mean", "fit_time_mean"]]

report = assumptions.assumption_report(X, y)
report["box_m"]                            # equality of covariance matrices
report["vif"].head()                       # most collinear features
report["normality_summary"]                # normality per class
```

## Project structure
```
dry-bean-lda/
├── README.md
├── requirements.txt
├── pyproject.toml
├── app/
│   ├── streamlit_app.py           # demo UI
│   ├── model_bundle.joblib        # exported model + background sample (committed)
│   └── requirements.txt           # lightweight deps for deployment
├── scripts/
│   └── train_and_export.py        # train final model, export the bundle
├── data/
│   ├── README.md                  # download instructions and data dictionary
│   └── raw/                       # (git-ignored) place dataset here
├── src/
│   └── fisher_lda/
│       ├── __init__.py
│       ├── lda.py                 # FisherLDA implementation
│       ├── data.py                # dataset loader
│       ├── assumptions.py         # normality, covariance, collinearity tests
│       ├── plots.py               # diagnostic plots
│       ├── benchmark.py           # model zoo, CV, calibration, PCA vs LDA, small-sample study
│       └── benchmark_plots.py     # benchmark plots
└── tests/
    ├── test_lda.py
    ├── test_assumptions.py
    ├── test_benchmark.py
    ├── test_interpret.py
    └── test_app_utils.py
```

### Live demo
_Link added after deployment._

## Roadmap
- From-scratch LDA, unit tests, validation vs scikit-learn
- EDA and assumption checks (normality, covariance homogeneity, multicollinearity)
- Benchmark vs QDA, shrinkage LDA, logistic regression, SVM, random forest; LDA vs PCA; small-sample study
- Interpretation (feature loadings, decision boundaries) and Streamlit demo

## Reference
Koklu, M. & Ozkan, I.A. (2020). *Multiclass classification of dry beans using computer vision and machine
learning techniques.* Computers and Electronics in Agriculture, 174, 105507.
Fisher, R.A. (1936). *The use of multiple measurements in taxonomic problems.* Annals of Eugenics, 7(2), 179-188.

## License
MIT