"""Train and evaluate PCOS classifiers and generate report artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/raw/PCOS_data_without_infertility.xlsx"
FIGURES_DIR = ROOT / "reports/figures"
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data/processed"
TARGET = "SOP (S/N)"
RANDOM_STATE = 42

# Nomes originais do dataset (após limpeza) -> português para análise e gráficos.
COLUMN_MAP = {
    "Sl. No": "Nº sequencial",
    "Patient File No.": "Nº do prontuário",
    "PCOS (Y/N)": "SOP (S/N)",
    "Age (yrs)": "Idade (anos)",
    "Weight (Kg)": "Peso (kg)",
    "Height(Cm)": "Altura (cm)",
    "BMI": "IMC",
    "Blood Group": "Grupo sanguíneo",
    "Pulse rate(bpm)": "Frequência cardíaca (bpm)",
    "RR (breaths/min)": "Frequência respiratória (rpm)",
    "Hb(g/dl)": "Hemoglobina (g/dL)",
    "Cycle(R/I)": "Ciclo (R/I)",
    "Cycle length(days)": "Duração do ciclo (dias)",
    "Marraige Status (Yrs)": "Tempo de casamento (anos)",
    "Pregnant(Y/N)": "Gestante (S/N)",
    "No. of aborptions": "Nº de abortos",
    "I beta-HCG(mIU/mL)": "Beta-HCG I (mIU/mL)",
    "II beta-HCG(mIU/mL)": "Beta-HCG II (mIU/mL)",
    "FSH(mIU/mL)": "FSH (mIU/mL)",
    "LH(mIU/mL)": "LH (mIU/mL)",
    "FSH/LH": "FSH/LH",
    "Hip(inch)": "Quadril (pol)",
    "Waist(inch)": "Cintura (pol)",
    "Waist:Hip Ratio": "Razão cintura/quadril",
    "TSH (mIU/L)": "TSH (mIU/L)",
    "AMH(ng/mL)": "AMH (ng/mL)",
    "PRL(ng/mL)": "Prolactina (ng/mL)",
    "Vit D3 (ng/mL)": "Vitamina D3 (ng/mL)",
    "PRG(ng/mL)": "Progesterona (ng/mL)",
    "RBS(mg/dl)": "Glicemia aleatória (mg/dL)",
    "Weight gain(Y/N)": "Ganho de peso (S/N)",
    "hair growth(Y/N)": "Crescimento de pelos (S/N)",
    "Skin darkening (Y/N)": "Escurecimento da pele (S/N)",
    "Hair loss(Y/N)": "Queda de cabelo (S/N)",
    "Pimples(Y/N)": "Espinhas (S/N)",
    "Fast food (Y/N)": "Fast food (S/N)",
    "Reg.Exercise(Y/N)": "Exercício regular (S/N)",
    "BP _Systolic (mmHg)": "PA sistólica (mmHg)",
    "BP _Diastolic (mmHg)": "PA diastólica (mmHg)",
    "Follicle No. (L)": "Nº de folículos (E)",
    "Follicle No. (R)": "Nº de folículos (D)",
    "Avg. F size (L) (mm)": "Tamanho médio do folículo (E) (mm)",
    "Avg. F size (R) (mm)": "Tamanho médio do folículo (D) (mm)",
    "Endometrium (mm)": "Endométrio (mm)",
}


def clean_column_name(name: object) -> str:
    """Normalize repeated whitespace without changing clinical labels."""
    return " ".join(str(name).strip().split())


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the clinical sheet and return clean predictors and target."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {path}. Execute `python scripts/download_data.py`."
        )

    data = pd.read_excel(path, sheet_name="Full_new")
    data.columns = [clean_column_name(column) for column in data.columns]
    data = data.rename(columns=COLUMN_MAP)
    data = data.dropna(axis=1, how="all")
    data = data.drop(
        columns=[column for column in data if column.lower().startswith("unnamed")]
    )
    data = data.drop_duplicates()

    removable = {"Nº sequencial", "Nº do prontuário"}
    data = data.drop(columns=[column for column in removable if column in data])

    if TARGET not in data:
        raise ValueError(f"Coluna alvo `{TARGET}` não encontrada.")

    y = pd.to_numeric(data.pop(TARGET), errors="raise").astype(int)
    X = data.apply(pd.to_numeric, errors="coerce")
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Impute data, one-hot encode coded categories, and scale numerics."""
    categorical = [
        column
        for column in ("Grupo sanguíneo", "Ciclo (R/I)")
        if column in X.columns
    ]
    numerical = [column for column in X.columns if column not in categorical]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numerical),
            ("cat", categorical_pipeline, categorical),
        ],
        verbose_feature_names_out=False,
    )


BASELINE_LR_HYPERPARAMS: dict[str, object] = {
    "class_weight": "balanced",
    "max_iter": 2_000,
    "random_state": RANDOM_STATE,
}

BASELINE_RF_HYPERPARAMS: dict[str, object] = {
    "class_weight": "balanced",
    "max_depth": 6,
    "min_samples_leaf": 3,
    "n_estimators": 400,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


def build_random_forest_pipeline(
    X: pd.DataFrame, **model_kwargs: object
) -> Pipeline:
    """Pipeline RF com o mesmo pré-processamento da Fase 1."""
    params = {**BASELINE_RF_HYPERPARAMS, **model_kwargs}
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X)),
            ("model", RandomForestClassifier(**params)),
        ]
    )


def build_logistic_pipeline(X: pd.DataFrame, **model_kwargs: object) -> Pipeline:
    """Pipeline de regressão logística com o mesmo pré-processamento da Fase 1."""
    params = {**BASELINE_LR_HYPERPARAMS, **model_kwargs}
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(X)),
            ("model", LogisticRegression(**params)),
        ]
    )


def get_train_test_split(
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Mesmo split estratificado usado em `main()` (reprodutibilidade Fase 1 → 2)."""
    X, y = load_data()
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def build_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Create interpretable baseline and nonlinear ensemble pipelines."""
    return {
        "Regressão Logística": build_logistic_pipeline(X),
        "Random Forest": build_random_forest_pipeline(X),
    }


def save_eda(X: pd.DataFrame, y: pd.Series) -> None:
    """Generate compact EDA artifacts used by the notebook and report."""
    sns.set_theme(style="whitegrid")

    labels = y.map({0: "Sem SOP", 1: "Com SOP"})
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=labels, hue=labels, legend=False, palette="Set2")
    ax.set(title="Distribuição da variável alvo", xlabel="", ylabel="Pacientes")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "distribuicao_alvo.png", dpi=160)
    plt.close()

    correlations = pd.concat([X, y.rename(TARGET)], axis=1).corr(numeric_only=True)
    target_corr = (
        correlations[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(12)
    )
    plt.figure(figsize=(8, 6))
    sns.barplot(x=target_corr.values, y=target_corr.index, color="#4C72B0")
    plt.title("Variáveis com maior correlação absoluta com SOP")
    plt.xlabel("|Correlação de Pearson|")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlacao_alvo.png", dpi=160)
    plt.close()

    top_features = target_corr.head(6).index.tolist()
    plot_data = pd.concat([X[top_features], labels.rename("Diagnóstico")], axis=1)
    melted = plot_data.melt(
        id_vars="Diagnóstico", var_name="Variável", value_name="Valor"
    )
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=melted,
        x="Variável",
        y="Valor",
        hue="Diagnóstico",
        showfliers=False,
    )
    plt.xticks(rotation=35, ha="right")
    plt.title("Distribuições das variáveis mais relacionadas ao diagnóstico")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "distribuicoes_relevantes.png", dpi=160)
    plt.close()

    pd.DataFrame(
        {
            "tipo": X.dtypes.astype(str),
            "ausentes": X.isna().sum(),
            "percentual_ausente": (X.isna().mean() * 100).round(2),
            "valores_unicos": X.nunique(),
        }
    ).to_csv(PROCESSED_DIR / "qualidade_dados.csv")
    correlations.to_csv(PROCESSED_DIR / "matriz_correlacao.csv")


def calculate_metrics(
    model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Calculate test metrics emphasizing missed positive cases."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def _display_feature_names(feature_names: np.ndarray) -> list[str]:
    """Remove ColumnTransformer prefixes for readable plots."""
    cleaned: list[str] = []
    for name in feature_names:
        text = str(name)
        for prefix in ("num__", "cat__", "remainder__"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break
        cleaned.append(text)
    return cleaned


def _build_shap_explanation(
    estimator, transformed: np.ndarray, feature_names: list[str]
):
    """Compute SHAP values for the fitted estimator on transformed data."""
    if isinstance(estimator, RandomForestClassifier):
        explanation = shap.TreeExplainer(estimator)(transformed)
        values = explanation.values
        if values.ndim == 3:
            values = values[:, :, 1]
        base_values = explanation.base_values
        if np.ndim(base_values) == 2:
            base_values = base_values[:, 1]
        return shap.Explanation(
            values=values,
            base_values=base_values,
            data=transformed,
            feature_names=feature_names,
        )
    explanation = shap.LinearExplainer(estimator, transformed)(transformed)
    return shap.Explanation(
        values=explanation.values,
        base_values=explanation.base_values,
        data=transformed,
        feature_names=feature_names,
    )


def _pick_case_indices(
    y_true: pd.Series, y_pred: np.ndarray, probabilities: np.ndarray
) -> dict[str, int]:
    """Pick one correct positive case and one clinically relevant error."""
    y_array = y_true.to_numpy()
    true_positive = np.where((y_array == 1) & (y_pred == 1))[0]
    false_negative = np.where((y_array == 1) & (y_pred == 0))[0]
    false_positive = np.where((y_array == 0) & (y_pred == 1))[0]
    true_negative = np.where((y_array == 0) & (y_pred == 0))[0]

    if len(true_positive):
        correct_idx = int(true_positive[np.argmax(probabilities[true_positive])])
        correct_tipo = "verdadeiro_positivo"
    elif len(true_negative):
        correct_idx = int(true_negative[np.argmin(probabilities[true_negative])])
        correct_tipo = "verdadeiro_negativo"
    else:
        raise RuntimeError("Não foi possível selecionar um caso correto no teste.")

    if len(false_negative):
        error_idx = int(false_negative[np.argmin(probabilities[false_negative])])
        error_tipo = "falso_negativo"
    elif len(false_positive):
        error_idx = int(false_positive[np.argmax(probabilities[false_positive])])
        error_tipo = "falso_positivo"
    else:
        raise RuntimeError("Não foi possível selecionar um erro de classificação.")

    return {
        "acerto_idx": correct_idx,
        "acerto_tipo": correct_tipo,
        "erro_idx": error_idx,
        "erro_tipo": error_tipo,
    }


def save_case_shap(
    best_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> None:
    """Save SHAP waterfall plots for one correct case and one error."""
    preprocessor = best_model.named_steps["preprocessor"]
    estimator = best_model.named_steps["model"]
    transformed = preprocessor.transform(X_test)
    feature_names = _display_feature_names(preprocessor.get_feature_names_out())
    explanation = _build_shap_explanation(estimator, transformed, feature_names)

    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    cases = _pick_case_indices(y_test, predictions, probabilities)

    records: list[dict[str, object]] = []
    plot_specs = (
        ("acerto", "shap_caso_acerto.png", cases["acerto_idx"], cases["acerto_tipo"]),
        ("erro", "shap_caso_erro.png", cases["erro_idx"], cases["erro_tipo"]),
    )
    tipo_labels = {
        "verdadeiro_positivo": "Verdadeiro positivo (tem SOP e o modelo acertou)",
        "verdadeiro_negativo": "Verdadeiro negativo (sem SOP e o modelo acertou)",
        "falso_negativo": "Falso negativo (tem SOP e o modelo errou)",
        "falso_positivo": "Falso positivo (sem SOP e o modelo errou)",
    }

    for alias, filename, idx, tipo in plot_specs:
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(explanation[idx], max_display=12, show=False)
        plt.title(f"SHAP individual — {alias} ({model_name})")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename, dpi=160, bbox_inches="tight")
        plt.close()

        records.append(
            {
                "caso": alias,
                "rotulo_anonimo": "Paciente A" if alias == "acerto" else "Paciente B",
                "tipo": tipo,
                "descricao": tipo_labels[tipo],
                "y_real": int(y_test.iloc[idx]),
                "y_predito": int(predictions[idx]),
                "probabilidade_sop": float(probabilities[idx]),
                "figura": filename,
            }
        )

    pd.DataFrame(records).to_csv(PROCESSED_DIR / "shap_casos.csv", index=False)


def save_explainability(
    best_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> None:
    """Save global feature importance and SHAP explanations."""
    preprocessor = best_model.named_steps["preprocessor"]
    estimator = best_model.named_steps["model"]
    transformed = preprocessor.transform(X_test)
    feature_names = _display_feature_names(preprocessor.get_feature_names_out())

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    else:
        importance = np.abs(estimator.coef_[0])

    feature_importance = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(
        PROCESSED_DIR / "feature_importance.csv", index=False
    )

    top = feature_importance.head(15).sort_values("importance")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"], color="#55A868")
    plt.title(f"Importância global — {model_name}")
    plt.xlabel("Importância")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=160)
    plt.close()

    sample_size = min(100, len(X_test))
    sample = transformed[:sample_size]
    explanation = _build_shap_explanation(estimator, transformed, feature_names)
    shap.summary_plot(
        explanation.values[:sample_size],
        sample,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.title(f"Impacto das variáveis (SHAP) — {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    save_case_shap(best_model, X_test, y_test, model_name)


def main() -> None:
    """Run the complete, reproducible modeling workflow."""
    for directory in (FIGURES_DIR, MODELS_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    save_eda(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    split_summary = {
        "total": len(X),
        "treino": len(X_train),
        "teste": len(X_test),
        "positivos_total": int(y.sum()),
        "prevalencia_total": float(y.mean()),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}
    for name, model in build_models(X).items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=("accuracy", "precision", "recall", "f1", "roc_auc"),
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        fitted_models[name] = model
        test_metrics = calculate_metrics(model, X_test, y_test)
        row: dict[str, object] = {"modelo": name}
        row.update({f"teste_{key}": value for key, value in test_metrics.items()})
        row.update(
            {
                f"cv_{metric}_media": float(scores[f"test_{metric}"].mean())
                for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")
            }
        )
        results.append(row)

        predictions = model.predict(X_test)
        report = classification_report(
            y_test,
            predictions,
            target_names=["Sem SOP", "Com SOP"],
            output_dict=True,
            zero_division=0,
        )
        safe_name = name.lower().replace(" ", "_").replace("ã", "a")
        pd.DataFrame(report).transpose().to_csv(
            PROCESSED_DIR / f"classification_report_{safe_name}.csv"
        )
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            display_labels=["Sem SOP", "Com SOP"],
            cmap="Blues",
        )
        plt.title(f"Matriz de confusão — {name}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"matriz_confusao_{safe_name}.png", dpi=160)
        plt.close()

    results_df = pd.DataFrame(results).sort_values(
        ["teste_recall", "teste_f1", "teste_roc_auc"], ascending=False
    )
    results_df.to_csv(PROCESSED_DIR / "comparacao_modelos.csv", index=False)
    best_name = str(results_df.iloc[0]["modelo"])
    best_model = fitted_models[best_name]
    joblib.dump(best_model, MODELS_DIR / "modelo_pcos.joblib")
    save_explainability(best_model, X_test, y_test, best_name)

    summary = {
        "melhor_modelo": best_name,
        "criterio_selecao": "recall, seguido de F1 e ROC-AUC no teste",
        "split": split_summary,
        "metricas": results_df.to_dict(orient="records"),
    }
    (PROCESSED_DIR / "resumo_resultados.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(results_df.round(4).to_string(index=False))
    print(f"\nMelhor modelo: {best_name}")
    print("Artefatos salvos em data/processed, reports/figures e models.")


if __name__ == "__main__":
    # Backend não interativo apenas na execução via terminal (salvar figuras).
    import matplotlib

    matplotlib.use("Agg")
    main()
