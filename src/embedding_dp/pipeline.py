"""Orchestrates: tasks × seeds × epsilons × DP methods × models, then writes the comparison report."""

from __future__ import annotations

import json
import time
import zlib

import numpy as np

from .config import METHOD_LABELS
from .data import Split, Task, describe_labels, load_embedding_report, load_task, make_split
from .edge import measure, payload_info
from .feature_importance import cross_entropy_gain, importance_weights
from .log import fmt_seconds, logger
from .privacy import laplace_mechanism, weighted_laplace_mechanism
from .report import make_report
from .results import ResultStore
from .training import tune_and_evaluate


class Privatizer:
    """Applies one DP mechanism. In importance mode, the cross-entropy gains, budget weights and
    feature scales are computed once per (task, seed) from the training nodes."""

    def __init__(self, mode: str, task: Task, split: Split, seed: int, args):
        self.mode = mode
        self.args = args
        self.info: dict = {"n_features_released": task.X_nodes.shape[1]}
        # Edge cost of the one-off importance estimation (zero for traditional Laplace).
        self.importance_cost = {"time_s": 0.0, "peak_mib": 0.0}
        if mode != "importance":
            return

        with measure() as self.importance_cost:
            train_nodes = task.node_mask(split.train)
            X_tr, y_tr = task.X_nodes[train_nodes], task.node_labels[train_nodes]
            gain = cross_entropy_gain(X_tr, y_tr, n_bins=args.n_bins, seed=seed)
            self.weights = importance_weights(gain, args.importance_power, args.top_k or None)
            self.feature_scale = np.abs(X_tr).mean(axis=0)

        order = np.argsort(gain)[::-1]
        top = [{"feature": int(j), "ce_gain": round(float(gain[j]), 5), "weight": round(float(self.weights[j]), 5)}
               for j in order[:10]]
        self.info.update(n_features_released=int((self.weights > 0).sum()),
                         top_features=json.dumps(top),
                         ce_gain_total=float(gain.sum()))
        logger.info("Cross-entropy gain on %d train nodes (%s): total=%.4f nats, max=%.4f, "
                    "features with gain>0: %d/%d", len(y_tr), fmt_seconds(self.importance_cost["time_s"]),
                    gain.sum(), gain.max(), (gain > 0).sum(), len(gain))
        logger.info("Top features (feature: gain → budget share): %s",
                    ", ".join(f"{t['feature']}: {t['ce_gain']:.4f}→{100 * t['weight']:.1f}%" for t in top[:5]))

    def __call__(self, X: np.ndarray, epsilon: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        if self.mode == "importance":
            return weighted_laplace_mechanism(X, epsilon, self.args.clip_norm, self.weights,
                                              self.feature_scale, rng)
        return laplace_mechanism(X, epsilon, self.args.clip_norm, rng)


def run(args) -> None:
    t_start = time.perf_counter()
    store = ResultStore(args.checkpoint, fresh=args.fresh)
    epsilons = list(args.epsilons) + ([float("inf")] if args.baseline else [])
    top_k = args.top_k if args.top_k else "all"

    total = len(args.tasks) * len(args.seeds) * len(epsilons) * len(args.dp_modes) * len(args.models)
    logger.info("Plan: %d tasks × %d seeds %s × %d epsilons %s × %d DP methods × %d models %s = %d runs "
                "(%d Optuna trials each)", len(args.tasks), len(args.seeds), args.seeds, len(epsilons), epsilons,
                len(args.dp_modes), len(args.models), args.models, total, args.n_trials)
    logger.info("Tasks: %s", ", ".join(args.tasks))
    logger.info("DP methods: %s | clip_norm=%g | ours: importance_power=%g, top_k=%s, n_bins=%d",
                ", ".join(METHOD_LABELS[m] for m in args.dp_modes), args.clip_norm, args.importance_power,
                top_k, args.n_bins)
    logger.info("Neighbor aggregation after DP (same for both methods): hops=%d%s",
                args.hops, " (off)" if args.hops == 0 else "")
    logger.info("Split=%.0f/%.0f/%.0f, xgb_device=%s", 100 * args.train_size, 50 * (1 - args.train_size),
                50 * (1 - args.train_size), args.xgb_device)

    step = 0
    run_times: list[float] = []

    for task_name in args.tasks:
        logger.info("=" * 80)
        task = load_task(task_name)
        unit = "graphs" if task.level == "graph" else "nodes"
        logger.info("%s (%s-level, source=%s): %d %s, %d dims, %d classes [%s]",
                    task.name, task.level, task.source, task.n_samples, unit, task.X_nodes.shape[1],
                    task.n_classes, describe_labels(task.y))
        logger.info("%s graph: %d nodes, %d directed edges, mean degree %.1f", task.name, len(task.X_nodes),
                    task.edge_index.shape[1], task.mean_degree)
        vgae_cost = load_embedding_report(task.source)
        if vgae_cost:
            logger.info("Embedding generation (VGAE, from run_report.json, %s): %.3fs, peak RAM %.0f MiB",
                        vgae_cost["vgae_device"], vgae_cost["vgae_inference_time_s"], vgae_cost["vgae_peak_ram_mib"])
        graph_info = {"n_nodes": len(task.X_nodes), "n_edges_directed": task.edge_index.shape[1],
                      "embedding_dim": task.X_nodes.shape[1]}

        for seed in args.seeds:
            split = make_split(task.y, args.train_size, seed)
            privatizers: dict[str, Privatizer] = {}  # built lazily: skipped when everything is in the checkpoint

            for e_i, epsilon in enumerate(epsilons):
                for mode in args.dp_modes:
                    base_key = dict(task=task.name, epsilon=epsilon, seed=seed, n_trials=args.n_trials,
                                    clip_norm_l1=args.clip_norm, train_size=args.train_size, dp_mode=mode,
                                    importance_power=args.importance_power,
                                    top_k=top_k if mode == "importance" else "all", hops=args.hops)
                    todo = [m for m in args.models if not store.is_done(model=m, **base_key)]
                    skipped = len(args.models) - len(todo)
                    if skipped:
                        step += skipped
                        logger.info("[%s | ε=%g | %s | seed=%d] skipping %d model(s) already in checkpoint",
                                    task.name, epsilon, mode, seed, skipped)
                    if not todo:
                        continue

                    logger.info("-" * 80)
                    logger.info(">>> %s | ε=%g | seed=%d | %s", task.name, epsilon, seed, METHOD_LABELS[mode])
                    if mode not in privatizers:
                        privatizers[mode] = Privatizer(mode, task, split, seed, args)
                    privatizer = privatizers[mode]
                    # Noise seed depends on (seed, task, epsilon), the same for both methods.
                    rng = np.random.default_rng([seed, zlib.crc32(task.name.encode()), e_i])
                    # Edge stages, measured separately: DP noise, then aggregation (post-processing,
                    # plus pooling for graph-level tasks). X is exactly what is sent to the cloud.
                    with measure() as noise_cost:
                        X_nodes_dp, scale = privatizer(task.X_nodes, epsilon, rng)
                    with measure() as agg_cost:
                        X = task.to_samples(task.propagate(X_nodes_dp, args.hops))
                    logger.info("Model input: %d samples × %d features (%d released dims × %d hop blocks)",
                                X.shape[0], X.shape[1], X_nodes_dp.shape[1], args.hops + 1)

                    imp = privatizer.importance_cost
                    edge_info = {
                        **graph_info,
                        **vgae_cost,
                        "edge_importance_time_s": imp["time_s"],
                        "edge_noise_time_s": noise_cost["time_s"],
                        "edge_aggregation_time_s": agg_cost["time_s"],
                        "edge_dp_time_s": imp["time_s"] + noise_cost["time_s"] + agg_cost["time_s"],
                        "edge_total_time_s": (vgae_cost.get("vgae_inference_time_s") or 0.0) + imp["time_s"]
                                             + noise_cost["time_s"] + agg_cost["time_s"],
                        "edge_dp_peak_mem_mib": max(imp["peak_mib"], noise_cost["peak_mib"], agg_cost["peak_mib"]),
                        **payload_info(X, args.bandwidths_mbps),
                    }
                    logger.info("Edge cost: importance=%.3fs noise=%.3fs aggregation=%.3fs (peak %.1f MiB) | "
                                "payload to cloud: %d values/sample = %d B/sample, %.2f MiB total",
                                imp["time_s"], noise_cost["time_s"], agg_cost["time_s"],
                                edge_info["edge_dp_peak_mem_mib"], edge_info["payload_dims"],
                                edge_info["payload_bytes_per_sample"], edge_info["payload_mib_total"])

                    for name in todo:
                        step += 1
                        logger.info("[%d/%d] %s | ε=%g | %s | seed=%d | %s: starting Optuna (%d trials)",
                                    step, total, task.name, epsilon, mode, seed, name, args.n_trials)
                        t_run = time.perf_counter()
                        res = tune_and_evaluate(name, X, task.y, split, task.n_classes, seed, epsilon, args)
                        m = res.test_metrics
                        store.add({
                            **base_key,
                            "model": name,
                            "method": METHOD_LABELS[mode],
                            "source": task.source,
                            "level": task.level,
                            "laplace_scale": scale,
                            **privatizer.info,
                            **edge_info,
                            "n_train": len(split.train),
                            "n_val": len(split.val),
                            "n_test": len(split.test),
                            "n_classes": task.n_classes,
                            "optuna_trials_run": res.n_trials,
                            "optuna_time_s": res.optuna_time_s,
                            "best_trial_val_macro_f1": res.best_trial_val_macro_f1,
                            "final_val_macro_f1": res.final_val_macro_f1,
                            "train_time_s": res.train_time_s,
                            "inference_time_s": res.inference_time_s,
                            "inference_ms_per_sample": 1000 * res.inference_time_s / len(split.test),
                            **m,
                            **res.mia,
                            "best_params": json.dumps(res.best_params),
                        }, roc=res.roc)

                        run_times.append(time.perf_counter() - t_run)
                        remaining = total - step
                        logger.info("[%d/%d] %s done in %s | test acc=%.4f macro_f1=%.4f weighted_f1=%.4f "
                                    "kappa=%.4f mcc=%.4f auc=%.4f | train=%.2fs | remaining runs=%d, ETA≈%s",
                                    step, total, name, fmt_seconds(run_times[-1]), m["accuracy"], m["macro_f1"],
                                    m["weighted_f1"], m["cohen_kappa"], m["mcc"], m["roc_auc"], res.train_time_s,
                                    remaining, fmt_seconds(np.mean(run_times) * remaining))

    saved = store.save_final(args.results_dir)
    if saved:
        run_dir, df, roc = saved
        make_report(df, roc, run_dir)
        logger.info("All outputs in %s", run_dir)
    logger.info("Finished in %s", fmt_seconds(time.perf_counter() - t_start))
