"""Loading the embeddings (.wsg.json), building each task and the train/val/test split."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import orjson
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import CACHE_DIR, EMBEDDINGS_DIR, TASKS
from .log import fmt_seconds, logger


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class Task:
    name: str
    source: str
    level: str                 # "node" or "graph"
    X_nodes: np.ndarray        # node embeddings of this task (DP is applied here)
    y: np.ndarray              # one label per sample (node or graph), classes 0..k-1
    groups: np.ndarray | None  # graph level only: sample index of each node
    edge_index: np.ndarray     # [2, n_edges] between the nodes of this task (both directions)

    def __post_init__(self):
        # Row-normalized adjacency with self loops: (A_hat @ X)_i = mean of x over i and its neighbors.
        n = len(self.X_nodes)
        A = sp.coo_matrix((np.ones(self.edge_index.shape[1]), (self.edge_index[0], self.edge_index[1])),
                          shape=(n, n)).tocsr()
        A.data[:] = 1.0  # drop duplicate edges
        A = A + sp.eye(n, format="csr")
        self.A_hat = sp.diags(1.0 / np.asarray(A.sum(axis=1)).ravel()) @ A

    @property
    def mean_degree(self) -> float:
        return self.edge_index.shape[1] / len(self.X_nodes)

    def propagate(self, X_nodes: np.ndarray, hops: int) -> np.ndarray:
        """Neighbor aggregation [X, ÂX, Â²X, ...] (SGC-style, no learned weights).

        Applied to the already privatized embeddings, so it is post-processing and keeps the
        DP guarantee of the node features. DP noise is independent across nodes, so averaging
        over neighbors cancels part of it. Edges are treated as public.
        """
        if hops <= 0:
            return X_nodes
        feats, cur = [X_nodes], X_nodes.astype(np.float64)
        for _ in range(hops):
            cur = self.A_hat @ cur
            feats.append(cur.astype(np.float32))
        return np.hstack(feats)

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def n_classes(self) -> int:
        return len(np.unique(self.y))

    @property
    def node_labels(self) -> np.ndarray:
        """Label of every node (graph level: the label of its graph)."""
        return self.y if self.groups is None else self.y[self.groups]

    def node_mask(self, sample_idx: np.ndarray) -> np.ndarray:
        """Boolean mask of the nodes that belong to the given samples (nodes or graphs)."""
        mask = np.zeros(len(self.X_nodes), dtype=bool)
        if self.groups is None:
            mask[sample_idx] = True
        else:
            mask[np.isin(self.groups, sample_idx)] = True
        return mask

    def to_samples(self, X_nodes: np.ndarray) -> np.ndarray:
        """Node level: identity. Graph level: mean pooling of the node embeddings of each graph.

        Pooling happens after DP, so it is post-processing and keeps the privacy guarantee.
        """
        if self.groups is None:
            return X_nodes
        sums = np.zeros((self.n_samples, X_nodes.shape[1]), dtype=np.float64)
        np.add.at(sums, self.groups, X_nodes)
        counts = np.bincount(self.groups, minlength=self.n_samples)[:, None]
        return (sums / counts).astype(np.float32)


def find_embedding_file(source: str, embeddings_dir: Path = EMBEDDINGS_DIR) -> Path:
    matches = sorted(embeddings_dir.glob(f"{source}__*/*.wsg.json"))
    if not matches:
        raise FileNotFoundError(f"No .wsg.json found for {source} in {embeddings_dir}")
    if len(matches) > 1:
        logger.warning("%d files found for %s; using %s", len(matches), source, matches[-1].name)
    return matches[-1]


@lru_cache(maxsize=None)
def load_embedding_report(source: str) -> dict:
    """Cost of generating the embeddings, from the run_report.json saved next to them by the VGAE training.

    Values refer to one inference pass over the whole graph on the device used at training time.
    """
    path = find_embedding_file(source).parent / "run_report.json"
    if not path.exists():
        logger.warning("%s: run_report.json not found; embedding generation cost will be missing", source)
        return {}
    report = orjson.loads(path.read_bytes())
    perf = report.get("Performance_Metrics", {})
    return {
        "vgae_inference_time_s": perf.get("Inference_Duration_Seconds"),
        "vgae_peak_ram_mib": perf.get("Inference_Peak_RAM_MiB"),
        "vgae_peak_vram_mib": perf.get("Memory_Peak_VRAM_MiB"),
        "vgae_device": report.get("Hyperparameters", {}).get("Device"),
    }


@lru_cache(maxsize=None)
def load_source(source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X [n_nodes, dim] float32, raw labels y, node names, edge_index [2, n_edges]).

    Cached in memory (Twitch is reused by six tasks) and on disk as .npz.
    """
    path = find_embedding_file(source)
    cache = CACHE_DIR / "embeddings" / f"{path.stem}.v3.npz"
    t0 = time.perf_counter()

    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        data = np.load(cache)
        logger.info("Loaded %s from cache (%s)", source, fmt_seconds(time.perf_counter() - t0))
        return data["X"], data["y"], data["names"], data["edge_index"]

    logger.info("Reading %s (%.0f MiB), this may take a while...", path.name, path.stat().st_size / 2**20)
    raw = orjson.loads(path.read_bytes())
    meta = raw["metadata"]
    n, dim = meta["num_nodes"], meta["num_total_features"]

    y = np.asarray(raw["graph_structure"]["y"])
    names = np.asarray(raw["graph_structure"]["node_names"], dtype=str)
    if len(y) != n or len(names) != n:
        raise ValueError(f"{source}: len(y)={len(y)}, len(names)={len(names)}, num_nodes={n}")

    X = np.zeros((n, dim), dtype=np.float32)
    filled = np.zeros(n, dtype=bool)
    for node_id, feat in raw["node_features"].items():
        i = int(node_id)
        X[i, feat["indices"]] = feat["weights"]
        filled[i] = True
    if not filled.all():
        raise ValueError(f"{source}: {(~filled).sum()} nodes without an embedding")

    edge_index = np.asarray(raw["graph_structure"]["edge_index"], dtype=np.int64)
    if edge_index.shape[0] != 2 or edge_index.max() >= n:
        raise ValueError(f"{source}: unexpected edge_index shape {edge_index.shape}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, names=names, edge_index=edge_index)
    logger.info("Parsed %s in %s and cached it to %s", source, fmt_seconds(time.perf_counter() - t0), cache)
    return X, y, names, edge_index


def _subgraph(edge_index: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep edges with both ends in mask and renumber the nodes to 0..mask.sum()-1."""
    new_id = np.full(len(mask), -1, dtype=np.int64)
    new_id[mask] = np.arange(mask.sum())
    keep = mask[edge_index[0]] & mask[edge_index[1]]
    return new_id[edge_index[:, keep]]


def load_task(name: str) -> Task:
    spec = TASKS[name]
    X, y, names, edge_index = load_source(spec.source)

    if spec.language is not None:
        langs = np.array([n.rsplit("_", 1)[-1] for n in names])
        mask = langs == spec.language
        if not mask.any():
            raise ValueError(f"{name}: no nodes with suffix _{spec.language}")
        X, y, names = X[mask], y[mask], names[mask]
        n_before = edge_index.shape[1]
        edge_index = _subgraph(edge_index, mask)
        logger.info("%s: kept %d nodes with suffix _%s and %d/%d edges", name, mask.sum(), spec.language,
                    edge_index.shape[1], n_before)

    if spec.level == "node":
        return Task(name, spec.source, "node", X, LabelEncoder().fit_transform(y), None, edge_index)

    # Graph level (Reddit threads): node "thread_<id>_user_<k>" belongs to graph "thread_<id>".
    graph_ids = np.array([n.split("_user_")[0] for n in names])
    _, groups = np.unique(graph_ids, return_inverse=True)
    n_graphs = groups.max() + 1
    y_enc = LabelEncoder().fit_transform(y)
    y_graph = np.full(n_graphs, -1)
    y_graph[groups] = y_enc
    n_mixed = np.unique(np.stack([groups, y_enc], axis=1), axis=0).shape[0] - n_graphs
    if n_mixed:
        logger.warning("%s: %d graphs have nodes with different labels; using the last one", name, n_mixed)
    logger.info("%s: %d nodes grouped into %d graphs (mean pooling)", name, len(names), n_graphs)
    return Task(name, spec.source, "graph", X, y_graph, groups, edge_index)


def make_split(y: np.ndarray, train_size: float, seed: int) -> Split:
    """Stratified train / val / test, with val and test splitting the remainder equally (70/15/15)."""
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, train_size=train_size, stratify=y, random_state=seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    logger.info("Split (seed=%d): train=%d | val=%d | test=%d", seed, len(train), len(val), len(test))
    return Split(train=train, val=val, test=test)


def describe_labels(y: np.ndarray) -> str:
    """E.g. '0: 27961 (74.2%), 1: 9739 (25.8%)'."""
    classes, counts = np.unique(y, return_counts=True)
    return ", ".join(f"{c}: {n} ({100 * n / len(y):.1f}%)" for c, n in zip(classes, counts))
