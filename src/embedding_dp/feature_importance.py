"""Per-feature importance measured as cross-entropy gain.

For each embedding dimension j we compare two predictors of the label y:
  * baseline: the class prior p(y)                    -> cross-entropy CE_prior
  * feature:  p(y | bin(x_j)), x_j split in quantiles -> cross-entropy CE_j
The importance is the reduction in cross-entropy, gain_j = CE_prior - CE_j (in nats),
an estimate of the mutual information I(x_j; y). Both predictors are fitted and scored
with K-fold cross-fitting, so a feature that only memorizes the training data gets ~0 gain.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold


def _cross_entropy(y: np.ndarray, proba_of_true: np.ndarray) -> float:
    return float(-np.log(np.clip(proba_of_true, 1e-12, 1.0)).mean())


def cross_entropy_gain(X: np.ndarray, y: np.ndarray, n_bins: int = 16, n_folds: int = 5,
                       smoothing: float = 1.0, seed: int = 0) -> np.ndarray:
    """Return gain_j >= 0 for every column of X (higher = more informative about y)."""
    n, d = X.shape
    k = int(y.max()) + 1
    ce_prior = np.zeros(n_folds)
    ce_feat = np.zeros((n_folds, d))

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(folds.split(X, y)):
        y_tr, y_te = y[tr], y[te]

        prior = (np.bincount(y_tr, minlength=k) + smoothing) / (len(y_tr) + k * smoothing)
        ce_prior[f] = _cross_entropy(y_te, prior[y_te])

        for j in range(d):
            # Quantile bin edges from the training fold only (inner edges -> n_bins bins).
            edges = np.unique(np.quantile(X[tr, j], np.linspace(0, 1, n_bins + 1)[1:-1]))
            b_tr = np.searchsorted(edges, X[tr, j])
            b_te = np.searchsorted(edges, X[te, j])
            n_b = len(edges) + 1

            counts = np.zeros((n_b, k))
            np.add.at(counts, (b_tr, y_tr), 1.0)
            cond = (counts + smoothing) / (counts.sum(axis=1, keepdims=True) + k * smoothing)
            ce_feat[f, j] = _cross_entropy(y_te, cond[b_te, y_te])

    gain = ce_prior.mean() - ce_feat.mean(axis=0)
    return np.maximum(gain, 0.0)


def importance_weights(gain: np.ndarray, power: float = 1.0, top_k: int | None = None) -> np.ndarray:
    """Turn gains into budget weights w_j >= 0 with sum(w) = 1.

    power = 0 -> uniform over the kept features; larger power -> more budget to the best ones.
    top_k     -> only the top_k features get budget (the rest are not released).
    """
    keep = np.ones(len(gain), dtype=bool)
    if top_k is not None and top_k < len(gain):
        keep[:] = False
        keep[np.argsort(gain, kind="stable")[::-1][:top_k]] = True
    w = (np.power(gain, power) if power > 0 else np.ones_like(gain)) * keep
    if w.sum() <= 0:  # no informative feature among the kept ones: uniform over the kept set
        w = keep.astype(float)
    return w / w.sum()
