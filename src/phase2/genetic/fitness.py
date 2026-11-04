from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.phase2.genetic.encoding import RandomForestGenome, genome_to_model_params
from src.train import build_random_forest_pipeline


def _composite_score(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
    """Alinhado à Fase 1: recall da classe SOP é prioridade."""
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc = 0.0
    return 0.5 * recall + 0.3 * f1 + 0.2 * roc


def evaluate_genome_cv(
    genome: RandomForestGenome,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 3,
    random_state: int = 42,
) -> tuple[float, dict[str, float]]:
    """Fitness por validação cruzada no treino (evita vazamento do holdout de teste)."""
    params = genome_to_model_params(genome)
    pipeline = build_random_forest_pipeline(X_train, n_jobs=1, **params)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict")
    y_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    y_array = y_train.to_numpy()
    score = _composite_score(y_array, y_pred, y_proba)
    details = {
        "recall_cv": float(recall_score(y_array, y_pred, zero_division=0)),
        "f1_cv": float(f1_score(y_array, y_pred, zero_division=0)),
        "roc_auc_cv": float(roc_auc_score(y_array, y_proba)),
        "fitness": float(score),
    }
    return score, details
