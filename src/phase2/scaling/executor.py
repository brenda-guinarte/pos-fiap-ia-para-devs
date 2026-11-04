from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_parallel(
    items: list[T],
    fn: Callable[[T], R],
    max_workers: int,
) -> list[R]:
    """
    Executa `fn` em paralelo sobre `items`.

    Usado no AG para escalar a avaliação de fitness quando a demanda (tamanho da
    população × gerações) cresce — equivalente local a múltiplos workers consumindo
    uma fila de indivíduos a avaliar.
    """
    if max_workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]

    results: list[R | None] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(fn, item): idx for idx, item in enumerate(items)}
        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()
    return [r for r in results if r is not None]
