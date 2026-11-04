"""Compara tempo de 1 geração do AG com 1 vs N workers (demo de escalabilidade)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase2.genetic.algorithm import GAConfig, run_genetic_algorithm
from src.train import get_train_test_split


def _run_once(workers: int) -> float:
    os.environ["GA_FITNESS_WORKERS"] = str(workers)
    X_train, _, y_train, _ = get_train_test_split()
    cfg = GAConfig(
        experiment_name=f"bench_w{workers}",
        population_size=12,
        generations=1,
        random_seed=99,
    )
    start = time.perf_counter()
    run_genetic_algorithm(X_train, y_train, cfg, parallel_fitness=workers > 1)
    return time.perf_counter() - start


def main() -> None:
    t1 = _run_once(1)
    t4 = _run_once(4)
    print(f"1 worker:  {t1:.2f}s")
    print(f"4 workers: {t4:.2f}s")
    if t4 > 0:
        print(f"Speedup aproximado: {t1 / t4:.2f}x")


if __name__ == "__main__":
    main()
