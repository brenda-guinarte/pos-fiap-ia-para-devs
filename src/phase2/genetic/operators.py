from __future__ import annotations

import numpy as np

from src.phase2.genetic.encoding import RandomForestGenome


def tournament_selection(
    population: list[RandomForestGenome],
    fitnesses: list[float],
    k: int,
    rng: np.random.Generator,
) -> RandomForestGenome:
    indices = rng.choice(len(population), size=k, replace=False)
    best = max(indices, key=lambda i: fitnesses[i])
    return population[best]


def uniform_crossover(
    parent_a: RandomForestGenome,
    parent_b: RandomForestGenome,
    rng: np.random.Generator,
) -> tuple[RandomForestGenome, RandomForestGenome]:
    va, vb = parent_a.to_vector(), parent_b.to_vector()
    mask = rng.random(len(va)) < 0.5
    child1 = RandomForestGenome.from_vector(np.where(mask, va, vb))
    child2 = RandomForestGenome.from_vector(np.where(mask, vb, va))
    return child1, child2


def mutate(
    genome: RandomForestGenome, mutation_rate: float, rng: np.random.Generator
) -> RandomForestGenome:
    v = genome.to_vector()
    for i in range(len(v)):
        if rng.random() < mutation_rate:
            if i == 4:
                v[i] = rng.integers(0, 3)
            elif i == 0:
                v[i] += rng.integers(-40, 41)
            else:
                v[i] += rng.integers(-2, 3)
    return RandomForestGenome.from_vector(v)
