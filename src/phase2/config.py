from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", cls.base_url).rstrip("/"),
            model=os.getenv("OLLAMA_MODEL", cls.model),
            timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT", "120")),
        )


def fitness_workers() -> int:
    """Workers para avaliação paralela de fitness (escala com núcleos de CPU)."""
    raw = os.getenv("GA_FITNESS_WORKERS", "4")
    try:
        return max(1, int(raw))
    except ValueError:
        return 4


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
