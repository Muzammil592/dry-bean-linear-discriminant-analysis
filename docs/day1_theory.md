# Day 1 Theory Notes: Fisher's LDA

## Setup
Data `X` (n x p), labels in K classes, class sizes n_k, class means mu_k, overall mean mu.

## Scatter matrices
- Within-class: `S_W = sum_k sum_{x in class k} (x - mu_k)(x - mu_k)^T`
- Between-class: `S_B = sum_k n_k (mu_k - mu)(mu_k - mu)^T`

## Fisher's criterion
Find the direction `w` that maximises
`J(w) = (w^T S_B w) / (w^T S_W w)`
i.e. push class means apart while keeping each class tight.

## Solution
Setting the gradient to zero gives the generalised eigenproblem `S_B w = lambda S_W w`,
equivalently the eigenvectors of `S_W^{-1} S_B`. Because rank(S_B) <= K - 1, there are at most
**min(K - 1, p)** useful axes (6 for Dry Bean).

## Numerically stable implementation (what the code does)
1. Eigendecompose the pooled covariance `S = S_W/(n-K) = V diag(s) V^T` (symmetric, so `eigh`).
2. Whiten: `W = V diag(s)^(-1/2)`, so that `W^T S W = I`.
3. Eigendecompose the whitened between-class matrix `W^T S_B W` (symmetric again).
4. Projection matrix = `W @ U`. Axes are uncorrelated and have unit within-class variance.

This avoids explicitly inverting `S_W` and tolerates collinear features by dropping null directions.

## Classification view
Assume `x | class k ~ N(mu_k, Sigma)` with shared Sigma. Bayes' rule gives a linear score
`delta_k(x) = x^T Sigma^-1 mu_k - 0.5 mu_k^T Sigma^-1 mu_k + log pi_k`; predict `argmax_k`.
Equal covariances is exactly the assumption we test on Day 2.

## Facts to explain in an interview
- Why at most K-1 axes?
- Why does LDA give *linear* boundaries (and QDA quadratic)?
- LDA vs PCA: supervised (maximises class separation) vs unsupervised (maximises variance).
- What breaks when features are collinear or n < p? (S_W singular -> shrinkage, Day 3.)
- Eigenvector sign and scale are arbitrary, which is why tests compare up to sign.
