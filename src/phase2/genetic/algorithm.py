from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.phase2.config import OUTPUT_DIR, ensure_dirs, fitness_workers
from src.phase2.scaling.executor import map_parallel
from src.phase2.genetic.encoding import RandomForestGenome, genome_to_model_params
from src.phase2.genetic.fitness import evaluate_genome_cv
from src.phase2.genetic.operators import mutate, tournament_selection, uniform_crossover
from src.phase2.monitoring.logging_config import PerformanceTracker, setup_logging


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 24
    generations: int = 15
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    tournament_size: int = 3
    elitism: int = 2
    random_seed: int = 42
    experiment_name: str = "default"


@dataclass
class GAResult:
    config: GAConfig
    best_genome: RandomForestGenome
    best_fitness: float
    history: list[dict]
    best_params: dict


def run_genetic_algorithm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: GAConfig,
    fitness_fn: Callable[
        [RandomForestGenome, pd.DataFrame, pd.Series], tuple[float, dict]
    ] = evaluate_genome_cv,
    parallel_fitness: bool = True,
) -> GAResult:
    logger = setup_logging()
    tracker = PerformanceTracker()
    rng = np.random.default_rng(config.random_seed)
    history: list[dict] = []

    population = [RandomForestGenome.random(rng) for _ in range(config.population_size)]
    best_genome = population[0]
    best_fitness = -1.0

    for gen in range(config.generations):
        fitnesses: list[float] = []
        details_list: list[dict] = []
        workers = fitness_workers() if parallel_fitness else 1
        with tracker.timed(
            "ga_generation",
            generation=gen,
            experiment=config.experiment_name,
            fitness_workers=workers,
            population_size=len(population),
        ):
            if workers <= 1:
                for individual in population:
                    fit, details = fitness_fn(individual, X_train, y_train)
                    fitnesses.append(fit)
                    details_list.append(details)
            else:

                def _eval_one(individual: RandomForestGenome) -> tuple[float, dict]:
                    return fitness_fn(individual, X_train, y_train)

                evaluated = map_parallel(population, _eval_one, workers)
                for fit, details in evaluated:
                    fitnesses.append(fit)
                    details_list.append(details)

        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fit = fitnesses[gen_best_idx]
        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_genome = population[gen_best_idx]

        history.append(
            {
                "generation": gen,
                "best_fitness": gen_best_fit,
                "mean_fitness": float(np.mean(fitnesses)),
                "best_details": details_list[gen_best_idx],
                "best_params": genome_to_model_params(population[gen_best_idx]),
            }
        )
        logger.info(
            "GA [%s] gen=%d fitness=%.4f recall_cv=%.4f",
            config.experiment_name,
            gen,
            gen_best_fit,
            details_list[gen_best_idx]["recall_cv"],
        )

        sorted_idx = np.argsort(fitnesses)[::-1]
        new_pop: list[RandomForestGenome] = [
            population[i] for i in sorted_idx[: config.elitism]
        ]

        while len(new_pop) < config.population_size:
            if rng.random() < config.crossover_rate:
                p1 = tournament_selection(population, fitnesses, config.tournament_size, rng)
                p2 = tournament_selection(population, fitnesses, config.tournament_size, rng)
                c1, c2 = uniform_crossover(p1, p2, rng)
                new_pop.append(mutate(c1, config.mutation_rate, rng))
                if len(new_pop) < config.population_size:
                    new_pop.append(mutate(c2, config.mutation_rate, rng))
            else:
                parent = tournament_selection(
                    population, fitnesses, config.tournament_size, rng
                )
                new_pop.append(mutate(parent, config.mutation_rate, rng))

        population = new_pop[: config.population_size]

    result = GAResult(
        config=config,
        best_genome=best_genome,
        best_fitness=best_fitness,
        history=history,
        best_params=genome_to_model_params(best_genome),
    )
    _persist_result(result)
    return result


def _persist_result(result: GAResult) -> None:
    ensure_dirs()
    path = OUTPUT_DIR / f"ga_{result.config.experiment_name}.json"
    payload = {
        "experiment_name": result.config.experiment_name,
        "ga_config": result.config.__dict__,
        "best_fitness": result.best_fitness,
        "best_params": result.best_params,
        "history": result.history,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
