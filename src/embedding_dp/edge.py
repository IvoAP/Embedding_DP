"""Edge-side cost measurement (time and memory) and communication cost of what leaves the edge.

In the proposed architecture the edge generates the embeddings (VGAE), privatizes them and aggregates
them over graph neighborhoods; only the resulting matrix is sent to the cloud. Each stage is timed
and its peak memory measured with tracemalloc (NumPy and SciPy report their buffers to it).
"""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager

import numpy as np

BYTES_PER_VALUE = 4  # float32 for embeddings
BYTES_PER_INDEX = 4  # int32 node ids for edge lists


@contextmanager
def measure():
    """Yields a dict that receives time_s and peak_mib of the enclosed block."""
    out: dict = {}
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        out["time_s"] = time.perf_counter() - t0
        out["peak_mib"] = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()


def payload_info(X: np.ndarray, bandwidths_mbps: list[float]) -> dict:
    """Size of the matrix sent to the cloud and its estimated transmission time per bandwidth."""
    total_bytes = X.shape[0] * X.shape[1] * BYTES_PER_VALUE
    info = {
        "payload_dims": X.shape[1],
        "payload_bytes_per_sample": X.shape[1] * BYTES_PER_VALUE,
        "payload_mib_total": total_bytes / 2**20,
    }
    for bw in bandwidths_mbps:
        info[f"tx_time_s_{bw:g}mbps"] = total_bytes * 8 / (bw * 1e6)
    return info


def raw_edges_bytes(n_edges_directed: int) -> int:
    """Edge list of the raw graph (each undirected edge once, two int32 ids); node features not included."""
    return (n_edges_directed // 2) * 2 * BYTES_PER_INDEX
