"""Laplace mechanisms for node-level differential privacy.

uniform (traditional)
    Clip each embedding to ||x||_1 <= C and add Laplace(0, 2C/ε) to every coordinate.
    Every feature gets the same noise, important or not.

importance (cross-entropy weighted)
    Each feature j receives a share w_j of the L1 budget (sum w_j = 1, w_j from the
    cross-entropy gain of the feature). Map z_j = x_j * w_j / m_j, where m_j = mean |x_j| on
    the training nodes, so a typical row has ||z||_1 ≈ 1; clip ||z||_1 <= C and add
    Laplace(0, 2C/ε) to z; map back x̂_j = z̃_j * m_j / w_j.

    Privacy: replacing one node changes z by at most 2C in L1, so releasing z̃ is ε-DP, and
    mapping back is post-processing. The noise on feature j in the original scale is
    2C m_j / (ε w_j): features with a larger budget share get proportionally less noise.
    Features with w_j = 0 are not released at all (dropped from the output).

    Caveat: w and m are computed from the training split (raw embeddings and labels). The
    guarantee above holds given w and m; the choice of w and m itself is not privatized.
"""

from __future__ import annotations

import numpy as np

from .log import logger


def clip_l1(X: np.ndarray, clip_norm: float) -> np.ndarray:
    """Rescale each row so that its L1 norm is <= clip_norm."""
    norms = np.abs(X).sum(axis=1, keepdims=True)
    return X * np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))


def _n_clipped(Z: np.ndarray, clip_norm: float) -> int:
    return int((np.abs(Z).sum(axis=1) > clip_norm).sum())


def laplace_mechanism(
    X: np.ndarray, epsilon: float, clip_norm: float, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Traditional per-record ε-DP. Returns (privatized X as float32, noise scale)."""
    X = X.astype(np.float64)
    n_clipped = _n_clipped(X, clip_norm)
    Xc = clip_l1(X, clip_norm)

    if np.isinf(epsilon):
        logger.info("[uniform] ε=inf (baseline): clipped %d/%d rows to L1<=%g, no noise",
                    n_clipped, len(X), clip_norm)
        return Xc.astype(np.float32), 0.0

    scale = 2.0 * clip_norm / epsilon
    noise = rng.laplace(0.0, scale, size=Xc.shape)
    snr = np.abs(Xc).mean() / np.abs(noise).mean()
    logger.info("[uniform] ε=%g: clipped %d/%d rows to L1<=%g, Laplace b=%.4g, signal/noise≈%.4f",
                epsilon, n_clipped, len(X), clip_norm, scale, snr)
    return (Xc + noise).astype(np.float32), scale


def weighted_laplace_mechanism(
    X: np.ndarray, epsilon: float, clip_norm: float, weights: np.ndarray, feature_scale: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Importance-weighted per-record ε-DP (see module docstring).

    weights: budget share per feature (sum = 1). feature_scale: m_j = mean |x_j| on train.
    Returns (privatized X with only the released columns, as float32; Laplace scale in z space).
    """
    active = weights > 0
    m = np.maximum(feature_scale, 1e-12)
    Z = X[:, active].astype(np.float64) * (weights[active] / m[active])
    n_clipped = _n_clipped(Z, clip_norm)
    Zc = clip_l1(Z, clip_norm)

    if np.isinf(epsilon):
        scale = 0.0
        Zn = Zc
    else:
        scale = 2.0 * clip_norm / epsilon
        Zn = Zc + rng.laplace(0.0, scale, size=Zc.shape)

    # Only the released features are returned (the others carry no information).
    out = Zn * (m[active] / weights[active])

    if np.isinf(epsilon):
        logger.info("[importance] ε=inf (baseline): %d/%d features released, clipped %d/%d rows, no noise",
                    active.sum(), len(weights), n_clipped, len(X))
    else:
        # Per-feature signal/noise in z space: mean |z_j| / b. Compare with what uniform DP would give.
        snr = np.zeros(len(weights))
        snr[active] = np.abs(Zc).mean(axis=0) / scale
        snr_uniform = np.abs(clip_l1(X.astype(np.float64), clip_norm)).mean(axis=0) / scale
        top = np.argsort(weights)[::-1][:3]
        logger.info("[importance] ε=%g: %d/%d features released, clipped %d/%d rows, Laplace b=%.4g",
                    epsilon, active.sum(), len(weights), n_clipped, len(X), scale)
        logger.info("[importance]   signal/noise on top features %s: %s (uniform DP would give %s)",
                    top.tolist(), ", ".join(f"{snr[j]:.4f}" for j in top),
                    ", ".join(f"{snr_uniform[j]:.4f}" for j in top))
    return out.astype(np.float32), scale
