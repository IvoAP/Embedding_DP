"""Checkpointing during the run and writing the final results to results_ml/run_<timestamp>/."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .log import logger

# A row counts as "already done" when all of these columns match.
RUN_KEY = ["task", "epsilon", "seed", "model", "n_trials", "clip_norm_l1", "train_size",
           "dp_mode", "importance_power", "top_k", "hops"]
GROUP = ["task", "dp_mode", "top_k", "hops", "epsilon", "model"]
SUMMARY_METRICS = ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "cohen_kappa", "mcc", "roc_auc",
                   "train_time_s", "inference_time_s", "inference_ms_per_sample",
                   "mia_auc_loss", "mia_advantage_loss", "mia_tpr_at_1fpr_loss", "mia_accuracy_loss",
                   "mia_auc_confidence", "mia_advantage_confidence"]
ROC_KEY = ["task", "dp_mode", "top_k", "hops", "epsilon", "seed", "model"]


def _key(row) -> tuple:
    return tuple(str(row.get(k)) for k in RUN_KEY)  # .get: old checkpoints may lack newer columns


class ResultStore:
    """Accumulates rows (and ROC points) and appends them to checkpoints so a run can be resumed."""

    def __init__(self, checkpoint: Path, fresh: bool = False):
        self.checkpoint = checkpoint
        self.roc_checkpoint = checkpoint.with_name(checkpoint.stem + "_roc.csv")
        if fresh:
            for path in (self.checkpoint, self.roc_checkpoint):
                if path.exists():
                    path.unlink()
                    logger.info("--fresh: deleted %s", path)
        self.rows: list[dict] = []
        self.roc_frames: list[pd.DataFrame] = []
        if self.checkpoint.exists():
            self.rows = pd.read_csv(self.checkpoint).to_dict("records")
            logger.info("Resuming from checkpoint with %d finished results (%s)", len(self.rows), self.checkpoint)
        if self.roc_checkpoint.exists():
            self.roc_frames.append(pd.read_csv(self.roc_checkpoint))
        self._done = {_key(r) for r in self.rows}

    def is_done(self, **key) -> bool:
        return _key(key) in self._done

    def add(self, row: dict, roc: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        self.rows.append(row)
        self._done.add(_key(row))
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(self.checkpoint, mode="a", header=not self.checkpoint.exists(), index=False)
        if roc is not None:
            fpr, tpr = roc
            frame = pd.DataFrame({**{k: row[k] for k in ROC_KEY}, "fpr": fpr, "tpr": tpr})
            self.roc_frames.append(frame)
            frame.to_csv(self.roc_checkpoint, mode="a", header=not self.roc_checkpoint.exists(), index=False)

    def save_final(self, results_dir: Path) -> tuple[Path, pd.DataFrame, pd.DataFrame] | None:
        """Write, inside results_dir/run_<timestamp>/:
        results.csv (one row per run), summary.csv (mean/std over seeds),
        best_by_task.csv (model with the highest mean test macro-F1 per configuration and epsilon)
        and roc_curves.csv (test ROC points of every run, binary tasks)."""
        if not self.rows:
            logger.warning("No results to save")
            return None
        run_dir = results_dir / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.rows)
        for col in SUMMARY_METRICS:  # metrics added later are missing from old checkpoint rows
            if col not in df:
                df[col] = float("nan")
        df["top_k"] = df["top_k"].astype(str)  # mixes "all" and numbers; keep one type for groupby
        df["hops"] = df.get("hops", pd.Series(0, index=df.index)).fillna(0).astype(int)  # older rows: no hops
        df.to_csv(run_dir / "results.csv", index=False)

        summary = df.groupby(GROUP)[SUMMARY_METRICS].agg(["mean", "std"])
        summary.columns = [f"{m}_{s}" for m, s in summary.columns]
        summary = summary.reset_index()
        summary.insert(len(GROUP), "n_seeds", df.groupby(GROUP).size().values)
        summary.to_csv(run_dir / "summary.csv", index=False)

        config_cols = [c for c in GROUP if c != "model"]
        best = summary.loc[summary.groupby(config_cols)["macro_f1_mean"].idxmax()]
        best.to_csv(run_dir / "best_by_task.csv", index=False)

        roc_df = pd.concat(self.roc_frames, ignore_index=True) if self.roc_frames else pd.DataFrame()
        if not roc_df.empty:
            roc_df["top_k"] = roc_df["top_k"].astype(str)
            roc_df.to_csv(run_dir / "roc_curves.csv", index=False)

        self.checkpoint.unlink(missing_ok=True)
        self.roc_checkpoint.unlink(missing_ok=True)
        with pd.option_context("display.max_rows", None, "display.width", 250):
            print("\n=== Summary (mean over seeds) ===")
            print(df.groupby(GROUP)[["accuracy", "macro_f1", "weighted_f1", "cohen_kappa", "mcc", "roc_auc",
                                     "train_time_s"]].mean().round(4))
        return run_dir, df, roc_df
