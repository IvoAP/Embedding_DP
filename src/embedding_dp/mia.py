"""Membership inference attack (MIA) on the final classifier.

Threshold attack of Yeom et al. (2018): the attacker sees the model output for a sample and
guesses "member" (was in the training set) when the model is unusually confident / has low loss
on it. No shadow models are needed, and it is the standard baseline for MIA evaluation.

Members are drawn from the training split and non-members from the test split, in equal numbers,
so 0.5 AUC / 0 advantage means the attacker cannot tell them apart.

Reported, for the loss score (−log p(true label)) and the confidence score (max class probability):
  * mia_auc_*           ROC-AUC of the attack (0.5 = no leakage, 1 = perfect attack)
  * mia_advantage_*     max over thresholds of TPR − FPR (membership advantage, Yeom et al.)
  * mia_tpr_at_1fpr_*   attack TPR at FPR = 1% (low-FPR regime, Carlini et al. 2022)
  * mia_accuracy_*      best balanced accuracy of the attack over thresholds

dp_advantage_bound = (e^ε − 1) / (e^ε + 1) is the largest advantage that an ε-DP release allows
against the membership of one node's embedding; it does not bound attacks that exploit the
labels, which are not privatized.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .data import Split


def _attack_metrics(is_member: np.ndarray, score: np.ndarray, suffix: str) -> dict:
    """score: higher = more likely to be a member."""
    fpr, tpr, _ = roc_curve(is_member, score)
    return {
        f"mia_auc_{suffix}": roc_auc_score(is_member, score),
        f"mia_advantage_{suffix}": float(np.max(tpr - fpr)),
        f"mia_tpr_at_1fpr_{suffix}": float(np.interp(0.01, fpr, tpr)),
        f"mia_accuracy_{suffix}": float(np.max((tpr + 1 - fpr) / 2)),
    }


def membership_inference(model, X: np.ndarray, y: np.ndarray, split: Split, epsilon: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = min(len(split.train), len(split.test))
    members = rng.choice(split.train, size=n, replace=False)
    non_members = rng.choice(split.test, size=n, replace=False)
    idx = np.concatenate([members, non_members])
    is_member = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    proba = model.predict_proba(X[idx])
    col = np.searchsorted(np.asarray(model.classes_), y[idx])
    p_true = np.clip(proba[np.arange(len(idx)), col], 1e-12, 1.0)

    out = {"mia_n_per_group": n}
    out |= _attack_metrics(is_member, np.log(p_true), "loss")         # low loss = member
    out |= _attack_metrics(is_member, proba.max(axis=1), "confidence")
    out["dp_advantage_bound"] = 1.0 if np.isinf(epsilon) else float(np.tanh(epsilon / 2))
    return out
