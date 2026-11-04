import numpy as np

from src.phase2.genetic.encoding import RandomForestGenome, genome_to_model_params
from src.phase2.genetic.operators import mutate, tournament_selection, uniform_crossover


def test_genome_roundtrip():
    rng = np.random.default_rng(0)
    g = RandomForestGenome.random(rng)
    g2 = RandomForestGenome.from_vector(g.to_vector())
    assert g.n_estimators == g2.n_estimators


def test_baseline_genome_maps_to_fase1():
    g = RandomForestGenome.from_baseline()
    params = genome_to_model_params(g)
    assert params["n_estimators"] == 400
    assert params["max_depth"] == 6


def test_operators():
    rng = np.random.default_rng(1)
    pop = [RandomForestGenome.random(rng) for _ in range(8)]
    fitness = list(range(8))
    assert tournament_selection(pop, fitness, 3, rng) in pop
    c1, c2 = uniform_crossover(pop[0], pop[1], rng)
    assert mutate(c1, 1.0, rng)
