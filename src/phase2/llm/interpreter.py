from __future__ import annotations

from dataclasses import dataclass

import httpx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.phase2.config import OllamaConfig
from src.phase2.llm.prompts import build_optimization_report_prompt, build_sop_explanation_prompt
from src.train import _display_feature_names


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str


class SopDiagnosisInterpreter:
    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig.from_env()

    def _ollama_available(self) -> bool:
        try:
            r = httpx.get(f"{self.config.base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def generate(self, prompt: str) -> LLMResponse:
        if not self._ollama_available():
            return LLMResponse(
                text=self._fallback_text(),
                provider="fallback",
                model="template",
            )
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3},
        }
        r = httpx.post(
            f"{self.config.base_url}/api/generate",
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        r.raise_for_status()
        data = r.json()
        return LLMResponse(
            text=data.get("response", "").strip(),
            provider="ollama",
            model=self.config.model,
        )

    def explain_case(
        self,
        pipeline: Pipeline,
        x_row: pd.Series,
        metrics: dict[str, float],
    ) -> LLMResponse:
        preprocessor = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["model"]
        transformed = preprocessor.transform(x_row.to_frame().T)
        feature_names = _display_feature_names(preprocessor.get_feature_names_out())

        proba = pipeline.predict_proba(x_row.to_frame().T)[0, 1]
        pred = int(pipeline.predict(x_row.to_frame().T)[0])
        label = "Com SOP" if pred == 1 else "Sem SOP"

        if isinstance(estimator, RandomForestClassifier):
            importances = estimator.feature_importances_
            scaled = importances * np.abs(transformed[0])
        else:
            scaled = np.abs(estimator.coef_[0]) * np.abs(transformed[0])

        top_idx = np.argsort(scaled)[::-1][:6]
        top_features = [(feature_names[i], float(scaled[i])) for i in top_idx]

        prompt = build_sop_explanation_prompt(label, float(proba), metrics, top_features)
        if not self._ollama_available():
            return LLMResponse(
                text=self._structured_case_text(label, float(proba), metrics, top_features),
                provider="structured_template",
                model="clinical_template_v1",
            )
        return self.generate(prompt)

    def report_optimization(
        self,
        baseline_metrics: dict[str, float],
        optimized_metrics: dict[str, float],
        optimized_params: dict,
    ) -> LLMResponse:
        prompt = build_optimization_report_prompt(
            baseline_metrics, optimized_metrics, optimized_params
        )
        if not self._ollama_available():
            return LLMResponse(
                text=self._structured_report_text(
                    baseline_metrics, optimized_metrics, optimized_params
                ),
                provider="structured_template",
                model="clinical_template_v1",
            )
        return self.generate(prompt)

    @staticmethod
    def _structured_case_text(
        label: str,
        probability_sop: float,
        metrics: dict[str, float],
        top_features: list[tuple[str, float]],
    ) -> str:
        features = "\n".join(f"- {name}" for name, _ in top_features[:5])
        return (
            "**Apoio à decisão — não é diagnóstico definitivo.**\n\n"
            f"O classificador (Random Forest, mesmo pré-processamento da Fase 1) "
            f"sinalizou **{label}**, com probabilidade estimada de SOP de "
            f"**{probability_sop:.1%}**. Esse valor resume padrões do dataset de "
            "treino e deve ser confrontado com queixa clínica, exame físico e "
            "propedêutica hormonal/imagem.\n\n"
            "Variáveis com maior contribuição nesta predição (importância local "
            f"aproximada):\n{features}\n\n"
            "**Insights acionáveis:** (1) Se a probabilidade for elevada ou houver "
            "sintomas compatíveis (ciclos irregulares, hiperandrogenismo), priorizar "
            "avaliação ginecológica e perfil hormonal (LH, FSH, AMH, andrógenos). "
            "(2) Considerar ultrassonografia pélvica para morfologia ovariana, conforme "
            "diretriz local. (3) Documentar fatores metabólicos (IMC, glicemia) já "
            "presentes no prontuário.\n\n"
            "**Limitações:** modelo treinado em 541 pacientes de Kerala (Índia); "
            "métricas de holdout (recall "
            f"{metrics.get('recall', 0):.2f}, F1 {metrics.get('f1', 0):.2f}, "
            f"ROC-AUC {metrics.get('roc_auc', 0):.2f}) descrevem desempenho "
            "populacional, não calibração individual. Validar em comitê clínico antes "
            "de uso assistido.\n\n"
            "_Gerado por template estruturado (Ollama indisponível); prompts em "
            "`src/phase2/llm/prompts.py`._"
        )

    @staticmethod
    def _structured_report_text(
        baseline_metrics: dict[str, float],
        optimized_metrics: dict[str, float],
        optimized_params: dict,
    ) -> str:
        return (
            "**Relatório de otimização (Random Forest)**\n\n"
            "O algoritmo genético ajustou hiperparâmetros mantendo o pipeline da "
            "Fase 1. No holdout de teste, o recall da classe SOP passou de "
            f"{baseline_metrics['recall']:.2%} para "
            f"{optimized_metrics['recall']:.2%} "
            f"(Δ {optimized_metrics['recall'] - baseline_metrics['recall']:+.2%}), "
            "com leve ganho de ROC-AUC e redução de accuracy/precisão global.\n\n"
            "**Recomendação de gestão:** na Fase 1 a Regressão Logística permaneceu "
            "como modelo principal pelo maior recall no teste (~88,9%). O RF otimizado "
            "aproxima-se (~86,1%) mas ainda não justifica substituição sem validação "
            "externa. Manter LR em piloto clínico; RF otimizado como linha comparativa.\n\n"
            f"Hiperparâmetros selecionados pelo AG: {optimized_params}.\n\n"
            "_Gerado por template estruturado (Ollama indisponível)._"
        )

    @staticmethod
    def _fallback_text() -> str:
        return (
            "[Modo offline — Ollama não detectado em localhost:11434]\n\n"
            "Resumo: a predição de apoio à triagem de SOP deve ser interpretada junto à "
            "história clínica e exames. O modelo da Fase 1 priorizou recall para não perder "
            "casos de SOP. Valide em comitê clínico antes de uso assistido.\n\n"
            "Para LLM local gratuita: instale Ollama e execute `ollama pull llama3.2`."
        )
