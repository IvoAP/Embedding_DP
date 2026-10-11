"""Hyperparameter search (Optuna), final training and evaluation.

Protocol (mirrors the GNN comparison):
  * Optuna maximizes macro-F1 on the validation set (models trained on train only).
  * The best configuration is retrained on train and evaluated once on test.
  * Training time covers only the final retraining; the Optuna cost is reported apart.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import optuna
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from .data import Split
from .log import fmt_seconds, logger
from .metrics import compute_metrics, roc_points
from .mia import membership_inference
from .models import build_model, suggest_params


@dataclass
class EvalResult:
    best_params: dict
    n_trials: int
    optuna_time_s: float
    best_trial_val_macro_f1: float
    final_val_macro_f1: float
    train_time_s: float
    inference_time_s: float
    test_metrics: dict
    roc: tuple[np.ndarray, np.ndarray] | None  # (fpr, tpr) on test, binary tasks only
    mia: dict                                  # membership inference attack metrics (see mia.py)


def fit(model, use_sample_weight: bool, X: np.ndarray, y: np.ndarray):
    if use_sample_weight:
        model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


class _ProgressCallback:
    """Logs each trial at DEBUG and a short progress line at INFO every ~25% of the trials."""

    def __init__(self, name: str, n_trials: int):
        self.name = name
        self.n_trials = n_trials
        self.t0 = time.perf_counter()
        self.report_every = max(1, n_trials // 4)

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        done = trial.number + 1
        elapsed = time.perf_counter() - self.t0
        trial_time = trial.duration.total_seconds() if trial.duration else float("nan")
        logger.debug("    %s trial %d/%d: val_macro_f1=%.4f (%.1fs) params=%s",
                     self.name, done, self.n_trials, trial.value, trial_time, trial.params)
        if done % self.report_every == 0 or done == self.n_trials:
            eta = elapsed / done * (self.n_trials - done)
            logger.info("    %s Optuna %d/%d | best val_macro_f1=%.4f | elapsed %s | ETA %s",
                        self.name, done, self.n_trials, study.best_value, fmt_seconds(elapsed), fmt_seconds(eta))


def tune(name: str, X: np.ndarray, y: np.ndarray, split: Split, n_classes: int, seed: int, args) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        model, sw = build_model(name, suggest_params(name, trial), n_classes, seed, args.n_jobs, args.xgb_device)
        fit(model, sw, X[split.train], y[split.train])
        return f1_score(y[split.val], model.predict(X[split.val]), average="macro")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   callbacks=[_ProgressCallback(name, args.n_trials)])
    return study


def tune_and_evaluate(name: str, X: np.ndarray, y: np.ndarray, split: Split, n_classes: int, seed: int,
                      epsilon: float, args) -> EvalResult:
    t0 = time.perf_counter()
    study = tune(name, X, y, split, n_classes, seed, args)
    optuna_time = time.perf_counter() - t0
    logger.info("    %s best params: %s", name, study.best_params)

    model, sw = build_model(name, study.best_params, n_classes, seed, args.n_jobs, args.xgb_device)
    logger.info("    %s retraining with the best params on train (%d samples)...", name, len(split.train))
    t0 = time.perf_counter()
    fit(model, sw, X[split.train], y[split.train])
    train_time = time.perf_counter() - t0

    final_val_f1 = f1_score(y[split.val], model.predict(X[split.val]), average="macro")

    t0 = time.perf_counter()
    proba = model.predict_proba(X[split.test])
    inference_time = time.perf_counter() - t0

    classes = np.asarray(model.classes_)
    mia = membership_inference(model, X, y, split, epsilon, seed)
    logger.info("    %s MIA (loss threshold): AUC=%.4f advantage=%.4f TPR@1%%FPR=%.4f",
                name, mia["mia_auc_loss"], mia["mia_advantage_loss"], mia["mia_tpr_at_1fpr_loss"])
    return EvalResult(
        best_params=study.best_params,
        n_trials=len(study.trials),
        optuna_time_s=optuna_time,
        best_trial_val_macro_f1=study.best_value,
        final_val_macro_f1=final_val_f1,
        train_time_s=train_time,
        inference_time_s=inference_time,
        test_metrics=compute_metrics(y[split.test], proba, classes),
        roc=roc_points(y[split.test], proba, classes),
        mia=mia,
    )
