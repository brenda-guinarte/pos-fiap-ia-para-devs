from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

MAX_FEATURES_CHOICES = ("sqrt", "log2", "1.0")


@dataclass(frozen=True)
class RandomForestGenome:
    """Genes = hiperparâmetros do Random Forest (Fase 1, modelo não linear)."""

    n_estimators: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    max_features_idx: int

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.n_estimators,
                self.max_depth,
                self.min_samples_split,
                self.min_samples_leaf,
                self.max_features_idx,
            ],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "RandomForestGenome":
        return cls(
            n_estimators=int(np.clip(round(v[0]), 50, 600)),
            max_depth=int(np.clip(round(v[1]), 3, 20)),
            min_samples_split=int(np.clip(round(v[2]), 2, 20)),
            min_samples_leaf=int(np.clip(round(v[3]), 1, 10)),
            max_features_idx=int(np.clip(round(v[4]), 0, len(MAX_FEATURES_CHOICES) - 1)),
        )

    @classmethod
    def from_baseline(cls) -> "RandomForestGenome":
        return cls(
            n_estimators=400,
            max_depth=6,
            min_samples_split=2,
            min_samples_leaf=3,
            max_features_idx=0,
        )

    @classmethod
    def random(cls, rng: np.random.Generator) -> "RandomForestGenome":
        return cls(
            n_estimators=int(rng.integers(80, 601)),
            max_depth=int(rng.integers(3, 21)),
            min_samples_split=int(rng.integers(2, 21)),
            min_samples_leaf=int(rng.integers(1, 11)),
            max_features_idx=int(rng.integers(0, len(MAX_FEATURES_CHOICES))),
        )


def genome_to_model_params(genome: RandomForestGenome) -> dict[str, Any]:
    mf = MAX_FEATURES_CHOICES[genome.max_features_idx]
    max_features: Any = mf if mf != "1.0" else 1.0
    return {
        "n_estimators": genome.n_estimators,
        "max_depth": genome.max_depth,
        "min_samples_split": genome.min_samples_split,
        "min_samples_leaf": genome.min_samples_leaf,
        "max_features": max_features,
    }
