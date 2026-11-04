from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from src.phase2.config import LOG_DIR, ensure_dirs


def setup_logging(name: str = "pcos_phase2", level: int = logging.INFO) -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


@dataclass
class PerformanceRecord:
    event: str
    duration_seconds: float
    metadata: dict[str, Any]

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class PerformanceTracker:
    def __init__(self, log_path: Path | None = None) -> None:
        ensure_dirs()
        self._path = log_path or (LOG_DIR / "performance.jsonl")
        self._logger = setup_logging("performance")

    def record(self, event: str, duration_seconds: float, **metadata: Any) -> None:
        rec = PerformanceRecord(event=event, duration_seconds=duration_seconds, metadata=metadata)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(rec.to_json_line() + "\n")
        self._logger.info("%s em %.3fs | %s", event, duration_seconds, metadata)

    @contextmanager
    def timed(self, event: str, **metadata: Any) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(event, time.perf_counter() - start, **metadata)
