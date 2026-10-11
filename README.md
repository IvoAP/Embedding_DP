# Embedding_DP

Node-level differential privacy (Laplace mechanism) on VGAE graph embeddings, evaluated with
classifiers tuned by Optuna.

The setting is an edge → cloud architecture. The edge generates node embeddings with a VGAE,
privatizes them and aggregates them over graph neighborhoods. Only the resulting matrix is sent
to the cloud, where a classifier is trained. The project compares two DP mechanisms:

| Mode         | Name                         | Idea                                                                                   |
|--------------|------------------------------|----------------------------------------------------------------------------------------|
| `uniform`    | Laplace (traditional)        | Clip each embedding to L1 ≤ C and add Laplace(2C/ε) noise to every dimension.           |
| `importance` | CE-weighted Laplace (ours)   | Split the privacy budget across dimensions in proportion to their cross-entropy gain, so informative features get less noise; only the top-k features are released. |

Each run reports utility (classification metrics), privacy (membership inference attack) and
edge cost (time, memory and payload size).

## Pipeline

For every task × seed × ε × DP mode × model:

1. **Load the task.** Read the VGAE embeddings (`.wsg.json`) and build a stratified 70/15/15 train/val/test split.
2. **Estimate feature importance** (`importance` mode only). Compute the cross-entropy gain of each
   dimension, an estimate of I(x_j; y), on the training nodes with quantile binning and 5-fold cross-fitting.
3. **Apply DP.** Clip and add Laplace noise (see [Privacy](#privacy)).
4. **Aggregate neighbors.** Build features `[X, ÂX, …, Â^hops X]` (SGC-style, no learned weights).
   This is post-processing, so the DP guarantee is kept. For graph-level tasks, node features are then mean-pooled per graph.
5. **Tune and train.** Optuna maximizes validation macro-F1. The best configuration is retrained on
   the training split and evaluated once on the test split.
6. **Evaluate.** Compute classification metrics, a membership inference attack and edge/communication costs.

## Tasks

| Task             | Source (`embeddings/` prefix) | Level | Notes                                     |
|------------------|-------------------------------|-------|-------------------------------------------|
| `Git-Web-ML`     | `Musae-Github`                | node  | Default task                              |
| `Twitch-<LANG>`  | `Musae-Twitch`                | node  | One task per language: DE, ENGB, ES, FR, PTBR, RU |
| `Facebook`       | `Musae-Facebook`              | node  |                                           |
| `Reddit-Threads` | `Reddit-Lite`                 | graph | One sample per thread (mean pooling)      |

## Installation

Requires Python ≥ 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Data

Embeddings are not versioned. Place each VGAE output folder in `embeddings/`:

```
embeddings/
└── Musae-Github__<suffix>/
    ├── <name>.wsg.json      # node features, labels, node names and edge_index
    └── run_report.json      # optional: VGAE inference time and memory (used in the edge cost)
```

On first load, each `.wsg.json` is parsed and cached as `.npz` in `.cache/embeddings/`.

## Usage

```bash
# Default: Git-Web-ML, ε ∈ {0.1, 0.5, 1, 5, 10}, both DP modes, all models, seed 42
uv run embedding-dp

# Several tasks and seeds, with a no-noise baseline (ε = ∞)
uv run embedding-dp --tasks Git-Web-ML Twitch-PTBR Facebook --seeds 42 43 44 --baseline

# Quick run
uv run embedding-dp --models LR XGB --epsilons 1 10 --n-trials 5

# Regenerate tables and figures of an existing run
uv run embedding-dp-report results_ml/run_<timestamp>
```

Main options (run `uv run embedding-dp --help` for the full list):

| Option                | Default                 | Description                                                         |
|-----------------------|-------------------------|---------------------------------------------------------------------|
| `--tasks`             | `Git-Web-ML`            | Tasks to run                                                        |
| `--epsilons`          | `0.1 0.5 1 5 10`        | Privacy budgets                                                     |
| `--baseline`          | off                     | Also run without noise (ε = ∞)                                      |
| `--dp-modes`          | `importance uniform`    | DP mechanisms to compare                                            |
| `--models`            | `MLP KNN LR RF XGB`     | Classifiers                                                         |
| `--clip-norm`         | `1.0`                   | Maximum L1 norm C of each embedding                                 |
| `--top-k`             | `4`                     | `importance` mode: number of released features (0 = all)           |
| `--importance-power`  | `1.0`                   | `importance` mode: budget share w_j ∝ gain_j^power                  |
| `--hops`              | `2`                     | Neighbor aggregation hops after DP (0 = off)                        |
| `--n-trials`          | `20`                    | Optuna trials per (task, ε, seed, model)                            |
| `--seeds`             | `42`                    | Each seed gives a new split, DP noise and model initialization      |
| `--bandwidths-mbps`   | `1 10 100`              | Uplink bandwidths used to estimate transmission time                |
| `--xgb-device`        | `cpu`                   | `cpu` or `cuda`                                                     |
| `--fresh`             | off                     | Ignore the checkpoint and start from scratch                        |
| `-v` / `-q`           |                         | Log every Optuna trial / only warnings and the final summary        |

### Resuming

Every finished run is appended to `.cache/checkpoint_results.csv`. If a run is interrupted, running
the same command again skips what is already done. The checkpoint is deleted after the final results are written.

## Outputs

Each run writes to `results_ml/run_<timestamp>/`:

| File                             | Content                                                                 |
|----------------------------------|-------------------------------------------------------------------------|
| `results.csv`                    | One row per (task, ε, seed, DP mode, model), with all metrics and costs  |
| `summary.csv`                    | Mean and std over seeds                                                 |
| `best_by_task.csv`               | Best model per configuration and ε                                      |
| `comparison.csv`                 | Ours vs traditional side by side, with deltas                           |
| `mia_by_model.csv`               | Membership inference results per model                                  |
| `edge_cost.csv`                  | Edge time and peak memory per stage                                     |
| `communication_cost.csv`         | Bytes sent edge → cloud and transmission time per bandwidth             |
| `roc_curves.csv`                 | Test ROC points (binary tasks)                                          |
| `metrics_vs_epsilon_<task>.png`  | Test metrics vs ε (best model per point, selected on validation)        |
| `mia_vs_epsilon_<task>.png`, `mia_by_model_<task>.png` | Membership inference vs ε                         |
| `edge_cost_<task>.png`, `communication_<task>.png`     | Edge-side and communication cost                  |
| `roc_curves_<task>.png`          | Test ROC curves per ε                                                   |

### Metrics

- **Classification:** accuracy, macro-F1, weighted-F1, balanced accuracy, Cohen's kappa, MCC, ROC-AUC, per-class precision/recall/F1.
- **Membership inference:** threshold attack of Yeom et al. (2018) on loss and confidence scores, with members drawn from train and non-members from test in equal numbers. The attack reports AUC, advantage (max TPR − FPR), TPR at 1% FPR and attack accuracy. For reference, it also reports the ε-DP advantage bound (e^ε − 1)/(e^ε + 1).
- **Edge cost:** time and peak memory of importance estimation, noise and aggregation; VGAE inference cost from `run_report.json`; payload size and transmission time.

## Privacy

**Uniform.** Each row is clipped to ‖x‖₁ ≤ C. Replacing one node changes the released matrix by
at most 2C in L1, so adding Laplace(0, 2C/ε) noise to each coordinate is ε-DP per node.

**Importance-weighted.** Each feature j gets a budget share w_j (Σ w_j = 1) from its cross-entropy
gain. The mechanism works in three steps:

1. Map z_j = x_j · w_j / m_j, where m_j is the mean |x_j| on training nodes.
2. Clip ‖z‖₁ ≤ C and add Laplace(0, 2C/ε) noise.
3. Map back with x̂_j = z̃_j · m_j / w_j.

The noise on feature j in the original scale is 2C·m_j / (ε·w_j), so features with a larger
share get less noise. Features with w_j = 0 are not released.

**Assumptions and limitations:**

- The guarantee covers each node's embedding vector. Graph edges are treated as public.
- In `importance` mode, the weights w and scales m are computed from the raw training embeddings and labels without privacy.
- Labels are not privatized, so the DP bound does not limit attacks that exploit them.
- The VGAE embedding of a node already depends on its neighbors. The per-node sensitivity therefore does not cover one node's influence on the embeddings of other nodes.

## Project structure

```
src/embedding_dp/
├── __init__.py            # entry point (embedding-dp)
├── config.py              # paths, tasks, defaults and CLI arguments
├── data.py                # embedding loading, tasks, neighbor aggregation, split
├── feature_importance.py  # cross-entropy gain and budget weights
├── privacy.py             # uniform and importance-weighted Laplace mechanisms
├── models.py              # classifiers and Optuna search spaces
├── training.py            # Optuna tuning, final training and evaluation
├── metrics.py             # classification metrics and ROC points
├── mia.py                 # membership inference attack
├── edge.py                # edge time/memory measurement and payload size
├── pipeline.py            # orchestration of the full experiment
├── results.py             # checkpointing and final CSVs
├── report.py              # comparison tables and figures (embedding-dp-report)
└── log.py                 # logging setup
```
