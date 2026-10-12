# Fisher's Linear Discriminant Analysis from Scratch
### Classifying Dry Bean Varieties with a NumPy Implementation, Assumption Diagnostics, and Model Benchmarks

> Can seven bean varieties be sorted automatically from geometric measurements alone, and how far can a
> classic linear method go against modern classifiers?
> 
---

## Why this project
Linear Discriminant Analysis is both a **classifier** and a **supervised dimensionality-reduction** technique,
yet it is usually used as a black box. This project builds it from first principles, verifies it against
scikit-learn, tests whether its assumptions actually hold on real data, and benchmarks it honestly against
QDA, logistic regression, SVM and random forest.

**Dataset:** [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), 13,611 beans,
16 shape features extracted from images, 7 classes. Real-world framing: automated seed quality sorting.

## Highlights
- Pure-NumPy `FisherLDA` with `fit`, `transform`, `predict`, `predict_proba`, `decision_function`
- Numerically stable solver (whitening + symmetric eigendecomposition; no explicit matrix inverse)
- Handles collinear features by discarding null directions of the within-class scatter
- Verified against `sklearn.discriminant_analysis.LinearDiscriminantAnalysis`: projections, predictions,
  probabilities and explained variance agree to ~1e-6 across 3 datasets
- Property-based tests: Fisher criterion is maximised, discriminant axes are uncorrelated, projected
  within-class covariance is the identity

## Method in 60 seconds
Fisher's criterion looks for directions `w` that maximise between-class scatter relative to within-class scatter:

```
J(w) = (wᵀ S_B w) / (wᵀ S_W w)     →     S_B w = λ S_W w
```

With K classes there are at most `min(K-1, p)` useful axes (6 here). The same model also yields a Bayes
classifier under the assumption of Gaussian classes with a shared covariance. Full derivation in
[`docs/day1_theory.md`](docs/day1_theory.md).

## Quick start
```bash
git clone https://github.com/Muzammil592/dry-bean-lda.git
cd dry-bean-lda
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
# download the data, see data/README.md
pytest                                                   # run the test-suite
jupyter notebook notebooks/01_lda_from_scratch_validation.ipynb
```

```python
from fisher_lda import FisherLDA, load_dry_bean

X, y = load_dry_bean("data/raw/Dry_Bean_Dataset.xlsx")
lda = FisherLDA().fit(X, y)
Z = lda.transform(X)                 # (n, 6) discriminant coordinates
lda.explained_variance_ratio_        # share of separation per axis
lda.predict_proba(X[:5])
```

## Project structure
```
dry-bean-lda/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── data/
│   ├── README.md                  # download instructions and data dictionary
│   └── raw/                       # (git-ignored) place dataset here
├── docs/
│   └── day1_theory.md             # scatter matrices, Fisher criterion, derivation
├── notebooks/
│   └── 01_lda_from_scratch_validation.ipynb
├── reports/
│   └── figures/
├── src/
│   └── fisher_lda/
│       ├── __init__.py
│       ├── lda.py                 # FisherLDA implementation
│       └── data.py                # dataset loader
└── tests/
    └── test_lda.py
```

## Results

### Validation against scikit-learn (Day 1)
| Check | Result |
|---|---|
| Projection match (up to sign) | _fill in: max abs diff_ |
| Prediction agreement | _fill in_ |
| Max probability difference | _fill in_ |
| Test accuracy (ours / sklearn) | _fill in_ |

![LD1 vs LD2](reports/figures/day1_ld1_ld2.png)

_Run the code, paste your real numbers, and keep the figure it saves._

## Reference
Koklu, M. & Ozkan, I.A. (2020). *Multiclass classification of dry beans using computer vision and machine
learning techniques.* Computers and Electronics in Agriculture, 174, 105507.
Fisher, R.A. (1936). *The use of multiple measurements in taxonomic problems.* Annals of Eugenics, 7(2), 179-188.
