from __future__ import annotations

from src.phase2.experiments.compare import compare_baseline_vs_optimized, plot_ga_history
from src.phase2.genetic.algorithm import GAConfig, run_genetic_algorithm
from src.phase2.llm.interpreter import SopDiagnosisInterpreter
from src.phase2.monitoring.logging_config import setup_logging
from src.train import get_train_test_split


EXPERIMENT_CONFIGS: list[GAConfig] = [
    GAConfig(
        experiment_name="exp1_pop16_mut10",
        population_size=16,
        generations=10,
        mutation_rate=0.10,
        random_seed=11,
    ),
    GAConfig(
        experiment_name="exp2_pop24_mut15",
        population_size=24,
        generations=10,
        mutation_rate=0.15,
        random_seed=22,
    ),
    GAConfig(
        experiment_name="exp3_pop20_mut25",
        population_size=20,
        generations=10,
        mutation_rate=0.25,
        random_seed=33,
    ),
]


def run_all() -> None:
    logger = setup_logging()
    X_train, X_test, y_train, y_test = get_train_test_split()
    ga_results = []

    for cfg in EXPERIMENT_CONFIGS:
        logger.info("Experimento %s", cfg.experiment_name)
        ga_results.append(run_genetic_algorithm(X_train, y_train, cfg))

    plot_ga_history(ga_results)

    best = max(ga_results, key=lambda r: r.best_fitness)
    cmp = compare_baseline_vs_optimized(X_train, X_test, y_train, y_test, best)

    interpreter = SopDiagnosisInterpreter()
    sample = X_test.iloc[0]
    metrics = cmp["optimized_metrics"]
    llm_case = interpreter.explain_case(cmp["optimized_pipeline"], sample, metrics)
    llm_report = interpreter.report_optimization(
        cmp["baseline_metrics"],
        cmp["optimized_metrics"],
        best.best_params,
    )

    from src.phase2.config import OUTPUT_DIR

    (OUTPUT_DIR / "llm_sample_case.txt").write_text(llm_case.text, encoding="utf-8")
    (OUTPUT_DIR / "llm_optimization_report.txt").write_text(llm_report.text, encoding="utf-8")
    logger.info("Melhor experimento AG: %s (fitness CV=%.4f)", best.config.experiment_name, best.best_fitness)


if __name__ == "__main__":
    run_all()
