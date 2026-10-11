"""Paths, task definitions, default values and command-line arguments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_DIR = ROOT / "embeddings"
CACHE_DIR = ROOT / ".cache"
RESULTS_DIR = ROOT / "results_ml"


@dataclass(frozen=True)
class TaskSpec:
    source: str                  # embedding folder prefix inside embeddings/
    level: str = "node"          # "node" = node classification, "graph" = one sample per thread
    language: str | None = None  # Twitch only: keep nodes whose name ends with _<language>


TWITCH_LANGUAGES = ["DE", "ENGB", "ES", "FR", "PTBR", "RU"]

# Same tasks as the GNN comparison: Git Web/ML, Reddit Threads and each Twitch language
# as an independent task. Only Git-Web-ML runs by default; the others stay selectable.
TASKS: dict[str, TaskSpec] = {
    "Git-Web-ML": TaskSpec("Musae-Github"),
    "Reddit-Threads": TaskSpec("Reddit-Lite", level="graph"),
    **{f"Twitch-{lang}": TaskSpec("Musae-Twitch", language=lang) for lang in TWITCH_LANGUAGES},
    "Facebook": TaskSpec("Musae-Facebook"),
}
DEFAULT_TASKS = ["Git-Web-ML"]

EPSILONS = [0.1, 0.5, 1.0, 5.0, 10.0]
MODELS = ["MLP", "KNN", "LR", "RF", "XGB"]

# DP mechanisms compared in a run. Both receive the same post-processing (neighbor aggregation).
DP_MODES = ["importance", "uniform"]
METHOD_LABELS = {
    "importance": "CE-weighted Laplace (ours)",
    "uniform": "Laplace (traditional)",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="embedding-dp",
        description="Laplace DP on VGAE embeddings + classifiers tuned with Optuna "
                    "(same protocol as the GNN comparison: 70/15/15 split, validation macro-F1).",
    )
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, choices=list(TASKS))
    ap.add_argument("--epsilons", nargs="+", type=float, default=EPSILONS)
    ap.add_argument("--baseline", action="store_true", help="also run without noise (epsilon=inf); off by default")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--clip-norm", type=float, default=1.0, help="maximum L1 norm C of each node embedding")
    ap.add_argument("--dp-modes", nargs="+", default=DP_MODES, choices=DP_MODES,
                    help="importance = cross-entropy weighted Laplace (ours, less noise on informative features); "
                         "uniform = traditional Laplace (same noise on every feature). Default: both, to compare")
    ap.add_argument("--importance-power", type=float, default=1.0,
                    help="importance mode: budget share w_j ∝ gain_j^power (0 = uniform over kept features)")
    ap.add_argument("--top-k", type=int, default=4,
                    help="importance mode: release only the top-k features by cross-entropy gain "
                         "(default 4, best on Git-Web-ML; 0 = all)")
    ap.add_argument("--n-bins", type=int, default=16, help="quantile bins used to estimate the cross-entropy gain")
    ap.add_argument("--hops", type=int, default=2,
                    help="neighbor aggregation after DP: features = [X, ÂX, ..., Â^hops X] (0 = off)")
    ap.add_argument("--bandwidths-mbps", nargs="+", type=float, default=[1.0, 10.0, 100.0],
                    help="uplink bandwidths used to estimate the edge → cloud transmission time")
    ap.add_argument("--n-trials", type=int, default=20, help="Optuna trials per (task, epsilon, seed, model)")
    ap.add_argument("--timeout", type=float, default=None, help="time limit in seconds per Optuna study")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42],
                    help="each seed gives a new split, new DP noise and new model initialization")
    ap.add_argument("--train-size", type=float, default=0.7,
                    help="train fraction; the rest is split equally into validation and test")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--xgb-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--checkpoint", type=Path, default=CACHE_DIR / "checkpoint_results.csv",
                    help="partial results file used to resume an interrupted run")
    ap.add_argument("--fresh", action="store_true", help="ignore the checkpoint and start from scratch")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="also log every Optuna trial")
    ap.add_argument("-q", "--quiet", action="store_true", help="only log warnings and the final summary")
    return ap.parse_args(argv)
