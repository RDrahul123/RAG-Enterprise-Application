"""Structured logging, stage timers and tiny Prometheus-style metrics.

Deliberately dependency-free: a portfolio project that exposes /metrics without
pulling in prometheus_client still demonstrates the important part (that you
instrumented the pipeline at all).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator

_LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=_LOG_LEVEL,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
)

# Third-party libraries log one line per HTTP request during model download.
for _noisy in (
    "httpx",
    "httpx2",
    "httpcore",
    "urllib3",
    "filelock",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger("rag")


def log_event(event: str, **fields: Any) -> None:
    """Emit one JSON log line. Keeps logs greppable and machine-parseable."""
    logger.info(json.dumps({"event": event, **fields}, default=str))


@contextmanager
def timed() -> Iterator[dict[str, float]]:
    """``with timed() as t: ...`` then read ``t['ms']``."""
    box: dict[str, float] = {"ms": 0.0}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = round((time.perf_counter() - start) * 1000, 2)


class Metrics:
    """In-process counters. Good enough for a single-container demo."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._latency_sum_ms: dict[str, float] = defaultdict(float)
        self._latency_count: dict[str, int] = defaultdict(int)

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] += value

    def observe_ms(self, name: str, ms: float, **labels: str) -> None:
        self.inc(name + "_sum_ms", ms, **labels)
        self.inc(name + "_count", 1, **labels)

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                if labels:
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels)
                else:
                    label_str = ""
                metric = name if not label_str else f"{name}{{{label_str}}}"
                lines.append(f"rag_{metric} {value:g}")
        return "\n".join(lines) + "\n"


metrics = Metrics()
