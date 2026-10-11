"""Comparison between our method (CE-weighted Laplace) and traditional Laplace: CSV tables + figures.

For each task, DP method and epsilon, the "best" model is chosen by mean validation macro-F1
(never by test), and its test metrics / ROC curve are the ones shown in the figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import METHOD_LABELS  # noqa: E402
from .edge import BYTES_PER_VALUE, raw_edges_bytes  # noqa: E402
from .log import logger  # noqa: E402

COMPARE_METRICS = ["accuracy", "macro_f1", "weighted_f1", "cohen_kappa", "mcc", "roc_auc", "train_time_s",
                   "mia_auc_loss", "mia_advantage_loss", "mia_tpr_at_1fpr_loss", "mia_auc_confidence"]
PLOT_METRICS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
    "cohen_kappa": "Cohen's kappa",
    "mcc": "MCC",
    "roc_auc": "ROC-AUC",
}
# Membership inference attack: lower = less leakage. (metric, title, reference line for "no leakage")
MIA_METRICS = {
    "mia_auc_loss": ("MIA AUC (loss attack)", 0.5),
    "mia_advantage_loss": ("MIA advantage (TPR − FPR)", 0.0),
    "mia_tpr_at_1fpr_loss": ("MIA TPR at 1% FPR", 0.01),
}
SHORT = {"uniform": "laplace", "importance": "ce_laplace"}

# Categorical slots 1-2 of the reference palette, plus line style / marker so identity is not color-only.
STYLE = {
    "uniform": dict(color="#2a78d6", linestyle="--", marker="s"),
    "importance": dict(color="#eb6834", linestyle="-", marker="o"),
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#a3a29c"  # bars that belong to neither method (raw graph, VGAE without DP)
CONFIG = ["task", "dp_mode", "epsilon"]

# Edge pipeline stages, in execution order (categorical slots 1-4 of the reference palette).
EDGE_STAGES = {
    "vgae_inference_time_s": ("VGAE embedding generation", "#2a78d6"),
    "edge_importance_time_s": ("Importance estimation (CE gain)", "#eb6834"),
    "edge_noise_time_s": ("Laplace noise (clip + perturb)", "#1baf7a"),
    "edge_aggregation_time_s": ("Neighbor aggregation", "#eda100"),
}
EDGE_COLS = list(EDGE_STAGES) + ["vgae_peak_ram_mib", "vgae_peak_vram_mib", "edge_dp_time_s", "edge_total_time_s",
                                 "edge_dp_peak_mem_mib", "payload_dims", "payload_bytes_per_sample",
                                 "payload_mib_total"]


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _eps_label(eps: float) -> str:
    return "∞\n(no noise)" if np.isinf(eps) else f"{eps:g}"


def best_models(df: pd.DataFrame) -> pd.DataFrame:
    """Per (task, method, epsilon): the model with the highest mean validation macro-F1, with mean test metrics."""
    metrics = [m for m in COMPARE_METRICS if m in df]
    means = df.groupby(CONFIG + ["model"])[["final_val_macro_f1"] + metrics].mean().reset_index()
    return means.loc[means.groupby(CONFIG)["final_val_macro_f1"].idxmax()].reset_index(drop=True)


def comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Side by side per (task, epsilon, model) plus a 'best (val)' row: traditional, ours and delta = ours − traditional."""
    metrics = [m for m in COMPARE_METRICS if m in df]
    per_model = df.groupby(CONFIG + ["model"])[metrics].mean().reset_index()
    best = best_models(df).rename(columns={"model": "selected_model"}).assign(model="best (val)")
    rows = pd.concat([per_model, best.drop(columns="final_val_macro_f1")], ignore_index=True)

    wide = rows.pivot_table(index=["task", "epsilon", "model"], columns="dp_mode",
                            values=metrics, aggfunc="first")
    out = pd.DataFrame(index=wide.index)
    if "selected_model" in rows:
        sel = rows.pivot_table(index=["task", "epsilon", "model"], columns="dp_mode",
                               values="selected_model", aggfunc="first")
        for mode in sel.columns:
            out[f"selected_model_{SHORT[mode]}"] = sel[mode]
    for metric in metrics:
        for mode in ("uniform", "importance"):
            if (metric, mode) in wide:
                out[f"{metric}_{SHORT[mode]}"] = wide[(metric, mode)]
        if (metric, "uniform") in wide and (metric, "importance") in wide:
            out[f"{metric}_delta"] = wide[(metric, "importance")] - wide[(metric, "uniform")]
    return out.reset_index().sort_values(["task", "model", "epsilon"])


def plot_metrics_vs_epsilon(best: pd.DataFrame, task: str, path: Path, metrics: dict | None = None,
                            title: str | None = None) -> None:
    """One panel per metric. metrics: {column: title} or {column: (title, reference_value)}."""
    metrics = metrics or PLOT_METRICS
    data = best[best["task"] == task]
    eps_values = sorted(data["epsilon"].unique(), key=lambda e: (np.isinf(e), e))
    x_of = {e: i for i, e in enumerate(eps_values)}

    n_rows = int(np.ceil(len(metrics) / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.75 * n_rows + 0.75), facecolor=SURFACE, squeeze=False)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    for ax, (metric, spec) in zip(axes.flat, metrics.items()):
        title_m, ref = spec if isinstance(spec, tuple) else (spec, None)
        _style_axes(ax)
        if ref is not None:
            ax.axhline(ref, color=INK_2, linestyle=":", linewidth=1)
            ax.annotate("no leakage", (len(eps_values) - 1, ref), textcoords="offset points", xytext=(0, 4),
                        ha="right", fontsize=8, color=INK_2)
        for mode in ("uniform", "importance"):
            d = data[data["dp_mode"] == mode].sort_values("epsilon", key=lambda s: s.map(x_of))
            if d.empty:
                continue
            st = STYLE[mode]
            finite = d[~np.isinf(d["epsilon"])]
            ax.plot(finite["epsilon"].map(x_of), finite[metric], color=st["color"], linestyle=st["linestyle"],
                    linewidth=2, marker=st["marker"], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=METHOD_LABELS[mode])
            no_noise = d[np.isinf(d["epsilon"])]  # reference point, not joined to the noisy curve
            ax.plot(no_noise["epsilon"].map(x_of), no_noise[metric], linestyle="none", marker=st["marker"],
                    markersize=7, markerfacecolor="none", markeredgecolor=st["color"], markeredgewidth=2)
        ax.set_title(title_m, color=INK, fontsize=11, loc="left")
        ax.set_xticks(range(len(eps_values)), [_eps_label(e) for e in eps_values])
        ax.set_xlabel("ε (privacy budget)", color=INK_2, fontsize=9)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles, labels = zip(*[(h, l) for h, l in zip(handles, labels) if not l.startswith("_")])
    note = "; hollow markers = no noise" if any(np.isinf(e) for e in eps_values) else ""
    title = title or "test metrics vs ε"
    top = 1 - 0.9 / fig.get_figheight()  # leave ~0.9 inch for title + legend
    fig.suptitle(f"{task}: {title} (best model per point, selected on validation macro-F1{note})",
                 color=INK, fontsize=12, y=1 - 0.15 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.45 / fig.get_figheight()),
               ncol=2, frameon=False, fontsize=10, labelcolor=INK)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def mia_by_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, model, epsilon): MIA AUC / advantage and macro-F1 for both methods + delta (ours − trad.)."""
    cols = ["mia_auc_loss", "mia_advantage_loss", "macro_f1"]
    means = df.groupby(["task", "model", "epsilon", "dp_mode"])[cols].mean().unstack("dp_mode")
    out = pd.DataFrame(index=means.index)
    for metric in cols:
        for mode in ("uniform", "importance"):
            if (metric, mode) in means:
                out[f"{metric}_{SHORT[mode]}"] = means[(metric, mode)]
        if (metric, "uniform") in means and (metric, "importance") in means:
            out[f"{metric}_delta"] = means[(metric, "importance")] - means[(metric, "uniform")]
    return out.reset_index()


def plot_mia_by_model(df: pd.DataFrame, task: str, path: Path) -> None:
    """Columns = models. Top row: MIA AUC vs ε (privacy, lower is better). Bottom row: macro-F1 (utility)."""
    data = df[df["task"] == task].groupby(["model", "dp_mode", "epsilon"])[["mia_auc_loss", "macro_f1"]] \
        .mean().reset_index()
    models = [m for m in ["MLP", "KNN", "LR", "RF", "XGB"] if m in set(data["model"])]
    eps_values = sorted(data["epsilon"].unique(), key=lambda e: (np.isinf(e), e))
    x_of = {e: i for i, e in enumerate(eps_values)}
    rows = [("mia_auc_loss", "MIA AUC (lower = more private)", 0.5), ("macro_f1", "Test macro-F1", None)]

    fig, axes = plt.subplots(2, len(models), figsize=(3.2 * len(models) + 0.6, 7.2), facecolor=SURFACE,
                             squeeze=False, sharey="row")
    for r, (metric, ylabel, ref) in enumerate(rows):
        for c, model in enumerate(models):
            ax = axes[r, c]
            _style_axes(ax)
            if ref is not None:
                ax.axhline(ref, color=INK_2, linestyle=":", linewidth=1)
            for mode in ("uniform", "importance"):
                d = data[(data["model"] == model) & (data["dp_mode"] == mode)].sort_values(
                    "epsilon", key=lambda s: s.map(x_of))
                if d.empty:
                    continue
                st = STYLE[mode]
                ax.plot(d["epsilon"].map(x_of), d[metric], color=st["color"], linestyle=st["linestyle"],
                        linewidth=2, marker=st["marker"], markersize=6, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, label=METHOD_LABELS[mode])
            ax.set_xticks(range(len(eps_values)), [_eps_label(e) for e in eps_values])
            if r == 0:
                ax.set_title(model, color=INK, fontsize=11, loc="left")
            else:
                ax.set_xlabel("ε (privacy budget)", color=INK_2, fontsize=9)
            if c == 0:
                ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        if ref is not None:
            axes[r, -1].annotate("no leakage", (len(eps_values) - 1, ref), textcoords="offset points",
                                 xytext=(0, 4), ha="right", fontsize=8, color=INK_2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"{task}: membership inference (loss attack) and utility per model", color=INK, fontsize=12,
                 y=1 - 0.15 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / fig.get_figheight()))
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.45 / fig.get_figheight()),
               ncol=2, frameon=False, fontsize=10, labelcolor=INK)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _tx_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("tx_time_s_")]


def edge_cost_table(df: pd.DataFrame) -> pd.DataFrame:
    """Edge-side cost per (task, method, epsilon), averaged over seeds (models share the same edge stage)."""
    cols = [c for c in EDGE_COLS + _tx_cols(df) if c in df]
    per_run = df.groupby(CONFIG + ["seed"])[cols].first().reset_index()
    out = per_run.groupby(CONFIG)[cols].mean().reset_index()
    out.insert(2, "method", out["dp_mode"].map(METHOD_LABELS))
    return out


def communication_table(df: pd.DataFrame) -> pd.DataFrame:
    """Bytes leaving the edge per scenario, with estimated transmission time per bandwidth."""
    tx = _tx_cols(df)
    bandwidths = {c: float(c.removeprefix("tx_time_s_").removesuffix("mbps")) for c in tx}
    rows = []
    for task, d in df.groupby("task"):
        first = d.iloc[0]
        n_nodes, dim = int(first["n_nodes"]), int(first["embedding_dim"])
        scenarios = [
            ("Raw graph edge list (node features not included)", "raw", np.nan,
             raw_edges_bytes(int(first["n_edges_directed"]))),
            (f"VGAE embeddings without DP ({dim} dims)", "embedding", dim, n_nodes * dim * BYTES_PER_VALUE),
        ]
        for mode in ("uniform", "importance"):
            m = d[d["dp_mode"] == mode]
            if not m.empty:
                scenarios.append((f"{METHOD_LABELS[mode]}, after aggregation", mode,
                                  m["payload_dims"].mean(), m["payload_mib_total"].mean() * 2**20))
        for name, kind, values, total in scenarios:
            row = {"task": task, "scenario": name, "kind": kind, "values_per_sample": values,
                   "bytes_per_node": total / n_nodes, "mib_total": total / 2**20}
            for col, bw in bandwidths.items():
                row[col] = total * 8 / (bw * 1e6)
            rows.append(row)
    out = pd.DataFrame(rows)
    trad = out[out["kind"] == "uniform"].set_index("task")["mib_total"]
    out["reduction_vs_traditional"] = out["task"].map(trad) / out["mib_total"]
    return out


def _bar_label(ax, y, x, text) -> None:
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(6, 0), va="center", fontsize=8.5, color=INK)


def plot_edge_cost(edge: pd.DataFrame, task: str, path: Path) -> None:
    """Left: stacked edge time per method. Right: peak memory of VGAE inference vs DP stage per method."""
    d = edge[edge["task"] == task].groupby("dp_mode").mean(numeric_only=True)
    modes = [m for m in ("uniform", "importance") if m in d.index]
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(14, 3.6), facecolor=SURFACE,
                                     gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.55})

    _style_axes(ax_t)
    ax_t.grid(axis="y", visible=False)
    for i, mode in enumerate(modes):
        left = 0.0
        for col, (label, color) in EDGE_STAGES.items():
            value = float(d.loc[mode, col]) if col in d and pd.notna(d.loc[mode, col]) else 0.0
            ax_t.barh(i, value, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=2,
                      label=label if i == 0 else None)
            left += value
        _bar_label(ax_t, i, left, f"{left:.3f} s")
    ax_t.set_yticks(range(len(modes)), [METHOD_LABELS[m] for m in modes], color=INK, fontsize=9.5)
    ax_t.set_xlabel("Edge processing time for the whole graph (s)", color=INK_2, fontsize=9)
    ax_t.set_xlim(0, ax_t.get_xlim()[1] * 1.18)
    ax_t.set_title("Time per edge stage (mean over ε)", color=INK, fontsize=11, loc="left")
    ax_t.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False, fontsize=8.5,
                labelcolor=INK)

    _style_axes(ax_m)
    ax_m.grid(axis="y", visible=False)
    # Neutral bars: in this figure orange/blue already mean edge stages on the left panel.
    bars = [("VGAE inference (peak RAM, GPU run)", d["vgae_peak_ram_mib"].iloc[0] if "vgae_peak_ram_mib" in d
             else np.nan)]
    bars += [(f"DP + aggregation · {METHOD_LABELS[m]}", d.loc[m, "edge_dp_peak_mem_mib"]) for m in modes]
    for i, (label, value) in enumerate(bars):
        if pd.notna(value):
            ax_m.barh(i, value, height=0.5, color=NEUTRAL)
            _bar_label(ax_m, i, value, f"{value:,.1f} MiB")
    ax_m.set_yticks(range(len(bars)), [b[0] for b in bars], color=INK, fontsize=9)
    ax_m.invert_yaxis()
    ax_m.set_xlim(0, ax_m.get_xlim()[1] * 1.3)
    ax_m.set_xlabel("Peak memory (MiB)", color=INK_2, fontsize=9)
    ax_m.set_title("Peak memory", color=INK, fontsize=11, loc="left")

    fig.suptitle(f"{task}: edge-side cost (VGAE time and RAM from run_report.json; DP stages measured on CPU)",
                 color=INK, fontsize=11.5, y=1.02)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_communication(comm: pd.DataFrame, task: str, path: Path) -> None:
    d = comm[comm["task"] == task].reset_index(drop=True)
    tx = _tx_cols(d)
    ref_bw = tx[len(tx) // 2] if tx else None  # middle bandwidth for the labels
    colors = {"raw": NEUTRAL, "embedding": NEUTRAL, "uniform": STYLE["uniform"]["color"],
              "importance": STYLE["importance"]["color"]}

    fig, ax = plt.subplots(figsize=(12, 0.75 * len(d) + 1.4), facecolor=SURFACE)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    for i, r in d.iterrows():
        ax.barh(i, r["mib_total"], height=0.55, color=colors[r["kind"]])
        label = f"{r['bytes_per_node']:,.0f} B/node · {r['mib_total']:,.2f} MiB"
        if ref_bw:
            bw = ref_bw.removeprefix("tx_time_s_").removesuffix("mbps")
            label += f" · {r[ref_bw]:,.1f} s @ {bw} Mbps"
        if r["kind"] == "importance" and pd.notna(r["reduction_vs_traditional"]):
            label += f" · {r['reduction_vs_traditional']:.0f}× less than traditional"
        _bar_label(ax, i, r["mib_total"], label)
    ax.set_yticks(range(len(d)), d["scenario"], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, d["mib_total"].max() * 2.1)  # room for the value labels
    ax.set_xlabel("Data sent from edge to cloud for the whole graph (MiB)", color=INK_2, fontsize=9)
    ax.set_title(f"{task}: communication cost edge → cloud", color=INK, fontsize=11.5, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _mean_roc(curves: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Average TPR over seeds on a common FPR grid."""
    tprs = []
    for _, c in curves.groupby("seed"):
        c = c.sort_values(["fpr", "tpr"])
        tprs.append(np.interp(grid, c["fpr"], c["tpr"]))
    return np.mean(tprs, axis=0)


def plot_roc_curves(best: pd.DataFrame, roc: pd.DataFrame, task: str, path: Path) -> None:
    data = best[best["task"] == task]
    eps_values = sorted(data["epsilon"].unique(), key=lambda e: (np.isinf(e), e))
    n_cols = 3
    n_rows = int(np.ceil(len(eps_values) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 4.3 * n_rows), facecolor=SURFACE, squeeze=False)
    grid = np.linspace(0, 1, 201)

    for ax, eps in zip(axes.flat, eps_values):
        _style_axes(ax)
        ax.plot([0, 1], [0, 1], color=INK_2, linestyle=":", linewidth=1, label="Chance")
        for mode in ("uniform", "importance"):
            pick = data[(data["dp_mode"] == mode) & (data["epsilon"] == eps)]
            if pick.empty:
                continue
            model = pick["model"].iloc[0]
            curves = roc[(roc["task"] == task) & (roc["dp_mode"] == mode) & (roc["epsilon"] == eps)
                         & (roc["model"] == model)]
            if curves.empty:
                continue
            st = STYLE[mode]
            ax.plot(grid, _mean_roc(curves, grid), color=st["color"], linestyle=st["linestyle"], linewidth=2,
                    label=f"{METHOD_LABELS[mode]} · {model} · AUC {pick['roc_auc'].iloc[0]:.3f}")
        ax.set_title(f"ε = {_eps_label(eps).replace(chr(10), ' ')}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("False positive rate", color=INK_2, fontsize=9)
        ax.set_ylabel("True positive rate", color=INK_2, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    for ax in list(axes.flat)[len(eps_values):]:
        ax.set_visible(False)

    fig.suptitle(f"{task}: test ROC curves (best model per method, selected on validation macro-F1)",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_report(df: pd.DataFrame, roc: pd.DataFrame, run_dir: Path) -> list[Path]:
    paths = []
    table = comparison_table(df)
    paths.append(run_dir / "comparison.csv")
    table.to_csv(paths[-1], index=False)

    has_mia = "mia_auc_loss" in df and df["mia_auc_loss"].notna().any()
    if has_mia:
        paths.append(run_dir / "mia_by_model.csv")
        mia_by_model_table(df).to_csv(paths[-1], index=False)

    has_edge = "payload_mib_total" in df and df["payload_mib_total"].notna().any()
    if has_edge:
        edge = edge_cost_table(df)
        comm = communication_table(df)
        paths.append(run_dir / "edge_cost.csv")
        edge.to_csv(paths[-1], index=False)
        paths.append(run_dir / "communication_cost.csv")
        comm.to_csv(paths[-1], index=False)

    best = best_models(df)
    for task in sorted(df["task"].unique()):
        slug = task.lower()
        paths.append(run_dir / f"metrics_vs_epsilon_{slug}.png")
        plot_metrics_vs_epsilon(best, task, paths[-1])
        if has_mia:
            paths.append(run_dir / f"mia_vs_epsilon_{slug}.png")
            plot_metrics_vs_epsilon(best, task, paths[-1], MIA_METRICS,
                                    "membership inference attack vs ε (lower = more private)")
            paths.append(run_dir / f"mia_by_model_{slug}.png")
            plot_mia_by_model(df, task, paths[-1])
        if has_edge:
            paths.append(run_dir / f"edge_cost_{slug}.png")
            plot_edge_cost(edge, task, paths[-1])
            paths.append(run_dir / f"communication_{slug}.png")
            plot_communication(comm, task, paths[-1])
        if not roc.empty and (roc["task"] == task).any():
            paths.append(run_dir / f"roc_curves_{slug}.png")
            plot_roc_curves(best, roc, task, paths[-1])

    show = ["task", "epsilon", "selected_model_laplace", "selected_model_ce_laplace"] + [
        f"{m}_{s}" for m in ("macro_f1", "cohen_kappa", "roc_auc", "mia_auc_loss")
        for s in ("laplace", "ce_laplace", "delta")]
    best_rows = table[table["model"] == "best (val)"]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("\n=== Ours (CE-weighted Laplace) vs traditional Laplace — best model per ε (selected on validation) ===")
        print(best_rows[[c for c in show if c in best_rows]].round(4).to_string(index=False))
    if has_edge:
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print("\n=== Edge cost (mean over ε and seeds) ===")
            print(edge.groupby(["task", "method"])[[c for c in EDGE_COLS if c in edge]].mean().round(4).T)
            print("\n=== Communication cost edge → cloud ===")
            print(comm.drop(columns="kind").round(3).to_string(index=False))
    logger.info("Report written: %s", ", ".join(p.name for p in paths))
    return paths


def main() -> None:
    """Regenerate tables and figures of an existing run: embedding-dp-report results_ml/run_<timestamp>"""
    import argparse

    from .log import setup_logging

    ap = argparse.ArgumentParser(prog="embedding-dp-report", description=main.__doc__)
    ap.add_argument("run_dir", type=Path, help="folder with results.csv (and optionally roc_curves.csv)")
    args = ap.parse_args()
    setup_logging(0)
    roc_path = args.run_dir / "roc_curves.csv"
    make_report(pd.read_csv(args.run_dir / "results.csv"),
                pd.read_csv(roc_path) if roc_path.exists() else pd.DataFrame(), args.run_dir)


if __name__ == "__main__":
    main()
