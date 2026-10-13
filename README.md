# Fisher's Linear Discriminant Analysis from Scratch
### Classifying Dry Bean Varieties with a NumPy Implementation, Assumption Diagnostics, and Model Benchmarks

> Can seven bean varieties be sorted automatically from geometric measurements alone, and how far can a
> classic linear method go against modern classifiers?

---

## Why this project
Linear Discriminant Analysis is both a **classifier** and a **supervised dimensionality-reduction** technique,
yet it is usually used as a black box. This project builds it from first principles, verifies it against
scikit-learn, tests whether its assumptions actually hold on real data, and benchmarks it honestly against
QDA, logistic regression, SVM and random forest.

**Dataset:** [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), 13,611 beans,
16 shape features extracted from images, 7 classes. Real-world framing: automated seed quality sorting.

## Highlights
**Day 1: implementation**
- Pure-NumPy `FisherLDA` with `fit`, `transform`, `predict`, `predict_proba`, `decision_function`
- Numerically stable solver (whitening + symmetric eigendecomposition, no explicit matrix inverse)
- Verified against `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`: projections, predictions,
  probabilities and explained variance agree to ~1e-6
- Property tests: Fisher criterion is maximised, discriminant axes are uncorrelated

**Day 2: assumption diagnostics**
- Reusable `fisher_lda.assumptions` module: per-class Shapiro-Wilk, skewness and kurtosis, Mardia's
  multivariate normality test, Box's M test, covariance-distance metric, VIF and condition number
- Diagnostic plots: per-class Q-Q grids, Mahalanobis chi-square plots, covariance heatmaps, correlation heatmap
- Statistical tests are validated against synthetic data with known ground truth (equal vs unequal covariance,
  normal vs skewed, collinear vs independent)
- Remedies (power transform, dropping collinear features) compared before and after

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
```

```python
from fisher_lda import FisherLDA, load_dry_bean, assumptions

X, y = load_dry_bean("data/raw/Dry_Bean_Dataset.xlsx")

lda = FisherLDA().fit(X, y)
Z = lda.transform(X)                       # (n, 6) discriminant coordinates

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
├── data/
│   ├── README.md                  # download instructions and data dictionary
│   └── raw/                       # (git-ignored) place dataset here
├── src/
│   └── fisher_lda/
│       ├── __init__.py
│       ├── lda.py                 # FisherLDA implementation
│       ├── data.py                # dataset loader
│       ├── assumptions.py         # normality, covariance, collinearity tests
│       └── plots.py               # diagnostic plots
└── tests/
    ├── test_lda.py
    └── test_assumptions.py
```

## Results

### Validation against scikit-learn (Day 1)
| Check | Result |
|---|---|
| Projection match (up to sign) | _fill in: max abs diff_ |
| Prediction agreement | _fill in_ |
| Test accuracy (ours / sklearn) | _fill in_ |

### Assumption diagnostics (Day 2)
| Assumption | Test | Finding on Dry Bean |
|---|---|---|
| Normality within classes | Shapiro-Wilk, skew, Mardia | _fill in_ |
| Equal covariance | Box's M, relative covariance distance | _fill in_ |
| Low collinearity | VIF, condition number | _fill in_ |
| Remedy effect | Power transform / feature drop, 5-fold CV | _fill in_ |

**Takeaway:** _one or two sentences on which assumptions fail and what that implies for Day 3 (e.g. QDA or shrinkage)._

## Reference
Koklu, M. & Ozkan, I.A. (2020). *Multiclass classification of dry beans using computer vision and machine
learning techniques.* Computers and Electronics in Agriculture, 174, 105507.
Fisher, R.A. (1936). *The use of multiple measurements in taxonomic problems.* Annals of Eugenics, 7(2), 179-188.

## License
MIT
