"""Classification metrics (same set as the GNN comparison)."""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def roc_points(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(fpr, tpr) of the ROC curve for binary tasks (positive class = classes[1]); None otherwise."""
    if len(classes) != 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, proba[:, 1], pos_label=classes[1])
    return fpr, tpr


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> dict:
    """Metrics from class probabilities; columns of `proba` follow `classes`."""
    y_pred = classes[proba.argmax(axis=1)]
    if len(classes) == 2:
        roc_auc = roc_auc_score(y_true, proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    per_class = {
        str(c): {"precision": p, "recall": r, "f1": f, "support": int(s)}
        for c, p, r, f, s in zip(classes, precision, recall, f1, support)
    }
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc,
        "per_class": json.dumps(per_class),
    }
