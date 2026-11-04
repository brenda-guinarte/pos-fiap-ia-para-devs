from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline

from src.phase2.config import OUTPUT_DIR, ensure_dirs
from src.phase2.genetic.algorithm import GAResult
from src.train import (
    BASELINE_RF_HYPERPARAMS,
    build_random_forest_pipeline,
    calculate_metrics,
)


def metrics_to_dict(metrics: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in metrics.items()}


def compare_baseline_vs_optimized(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    ga_result: GAResult,
) -> dict:
    baseline = build_random_forest_pipeline(X_train)
    optimized = build_random_forest_pipeline(X_train, **ga_result.best_params)

    baseline.fit(X_train, y_train)
    optimized.fit(X_train, y_train)

    baseline_metrics = calculate_metrics(baseline, X_test, y_test)
    optimized_metrics = calculate_metrics(optimized, X_test, y_test)

    comparison = {
        "modelo": "Random Forest",
        "baseline_fase1": {
            **metrics_to_dict(baseline_metrics),
            "hyperparams": {k: v for k, v in BASELINE_RF_HYPERPARAMS.items() if k != "n_jobs"},
        },
        "otimizado_ag": {
            **metrics_to_dict(optimized_metrics),
            "hyperparams": ga_result.best_params,
            "ga_fitness_cv": ga_result.best_fitness,
        },
        "delta": {
            k: optimized_metrics[k] - baseline_metrics[k]
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc")
        },
        "referencia_fase1": {
            "melhor_modelo_clinico": "Regressão Logística",
            "criterio": "recall no teste, depois F1 e ROC-AUC",
            "nota": "O AG desta fase otimiza o Random Forest; compare também com LR em resumo_resultados.json.",
        },
    }
    ensure_dirs()
    out = OUTPUT_DIR / f"comparison_{ga_result.config.experiment_name}.json"
    out.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "comparison": comparison,
        "baseline_metrics": baseline_metrics,
        "optimized_metrics": optimized_metrics,
        "optimized_pipeline": optimized,
        "baseline_pipeline": baseline,
    }


def load_phase1_model() -> Pipeline | None:
    path = Path(__file__).resolve().parents[3] / "models" / "modelo_pcos.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def plot_ga_history(results: list[GAResult], output_name: str = "ga_convergence.png") -> Path:
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(8, 5))
    for res in results:
        gens = [h["generation"] for h in res.history]
        best = [h["best_fitness"] for h in res.history]
        ax.plot(gens, best, label=res.config.experiment_name)
    ax.set_xlabel("Geração")
    ax.set_ylabel("Fitness (CV treino)")
    ax.set_title("Convergência do AG — Random Forest PCOS")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = OUTPUT_DIR / output_name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
