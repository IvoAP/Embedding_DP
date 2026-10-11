"""Classifier construction and Optuna search spaces."""

from __future__ import annotations

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def suggest_params(name: str, trial: optuna.Trial) -> dict:
    if name == "LR":
        return {
            "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if name == "KNN":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 75),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }
    if name == "RF":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            # Bounded depth: unlimited trees were very slow on 26k samples and memorize training points.
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if name == "XGB":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "balanced": trial.suggest_categorical("balanced", [False, True]),
        }
    if name == "MLP":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", ["64", "128", "256", "128-64", "256-128"]
            ),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        }
    raise ValueError(f"Unknown model: {name}")


def build_model(name: str, params: dict, n_classes: int, seed: int, n_jobs: int, xgb_device: str):
    """Return (estimator, whether to fit with balanced sample weights)."""
    p = dict(params)
    if name == "LR":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=3000, random_state=seed),
        ), False
    if name == "KNN":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"], p=p["p"], n_jobs=n_jobs),
        ), False
    if name == "RF":
        return RandomForestClassifier(**p, n_jobs=n_jobs, random_state=seed), False
    if name == "XGB":
        balanced = p.pop("balanced")
        multiclass = n_classes > 2
        return XGBClassifier(
            **p,
            tree_method="hist",
            device=xgb_device,
            n_jobs=n_jobs,
            random_state=seed,
            objective="multi:softprob" if multiclass else "binary:logistic",
            eval_metric="mlogloss" if multiclass else "logloss",
        ), balanced
    if name == "MLP":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=tuple(int(h) for h in p["hidden_layer_sizes"].split("-")),
                alpha=p["alpha"],
                learning_rate_init=p["learning_rate_init"],
                batch_size=p["batch_size"],
                early_stopping=True,
                n_iter_no_change=10,
                max_iter=300,
                random_state=seed,
            ),
        ), False
    raise ValueError(f"Unknown model: {name}")
