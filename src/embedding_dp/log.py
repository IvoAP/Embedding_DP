"""Logging setup shared by all modules."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("embedding_dp")


def setup_logging(verbosity: int) -> None:
    """verbosity: -1 = warnings only, 0 = progress (INFO), 1+ = every Optuna trial (DEBUG)."""
    level = logging.WARNING if verbosity < 0 else logging.INFO if verbosity == 0 else logging.DEBUG
    # Windows consoles default to cp1252, which cannot print symbols such as ε or ≈.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", "%H:%M:%S"))
    logger.handlers[:] = [handler]
    logger.setLevel(level)
    logger.propagate = False


def fmt_seconds(s: float) -> str:
    """Human-readable duration, e.g. 3725 -> '1h02m05s'."""
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"
