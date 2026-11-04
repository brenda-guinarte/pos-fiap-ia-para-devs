from __future__ import annotations

import argparse

from src.phase2.experiments.compare import compare_baseline_vs_optimized
from src.phase2.genetic.algorithm import GAConfig, run_genetic_algorithm
from src.phase2.llm.interpreter import SopDiagnosisInterpreter
from src.phase2.monitoring.logging_config import setup_logging
from src.train import get_train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tech Challenge Fase 2 — AG + LLM (continuação PCOS Fase 1)"
    )
    parser.add_argument("--quick", action="store_true", help="Menos gerações/população")
    args = parser.parse_args()
    setup_logging()

    X_train, X_test, y_train, y_test = get_train_test_split()
    cfg = GAConfig(
        experiment_name="demo_quick" if args.quick else "demo_full",
        population_size=10 if args.quick else 20,
        generations=4 if args.quick else 12,
        mutation_rate=0.15,
    )
    ga = run_genetic_algorithm(X_train, y_train, cfg)
    cmp = compare_baseline_vs_optimized(X_train, X_test, y_train, y_test, ga)

    interpreter = SopDiagnosisInterpreter()
    explanation = interpreter.explain_case(
        cmp["optimized_pipeline"],
        X_test.iloc[0],
        cmp["optimized_metrics"],
    )

    print("\n=== Random Forest: baseline Fase 1 vs AG (teste) ===")
    print("Recall baseline:", round(cmp["comparison"]["baseline_fase1"]["recall"], 4))
    print("Recall otimizado:", round(cmp["comparison"]["otimizado_ag"]["recall"], 4))
    print("\n=== Interpretação LLM (caso de teste) ===\n")
    print(explanation.text)


if __name__ == "__main__":
    main()
