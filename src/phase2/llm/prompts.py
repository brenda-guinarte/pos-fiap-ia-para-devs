from __future__ import annotations

SYSTEM_SOP = """Você apoia equipes de saúde na triagem de Síndrome dos Ovários Policísticos (SOP/PCOS).
Regras:
- Não substitui avaliação médica; deixe isso explícito.
- Português do Brasil, linguagem clara.
- Use apenas os dados fornecidos; não invente exames.
- Priorize segurança: falso negativo em SOP pode atrasar investigação.
"""


def build_sop_explanation_prompt(
    predicted_label: str,
    probability_sop: float,
    metrics: dict[str, float],
    top_features: list[tuple[str, float]],
) -> str:
    features_text = "\n".join(
        f"- {name}: peso relativo {contrib:.3f}" for name, contrib in top_features[:6]
    )
    return f"""{SYSTEM_SOP}

Caso (modelo de apoio — dados clínicos estruturados, Fase 1/Fase 2):
- Predição: {predicted_label}
- Probabilidade estimada de SOP: {probability_sop:.1%}

Métricas do modelo no holdout de teste (população, não individual):
- recall (classe SOP): {metrics.get('recall', 0):.3f}
- f1: {metrics.get('f1', 0):.3f}
- roc_auc: {metrics.get('roc_auc', 0):.3f}

Variáveis que mais influenciaram esta predição:
{features_text}

Tarefa:
1) Explique o resultado para a equipe de atenção primária em 2 parágrafos.
2) Liste 3 ações clínicas razoáveis (ex.: investigação hormonal, ultrassom).
3) Destaque limitações (amostra regional, possível viés, necessidade de confirmação).
"""


def build_optimization_report_prompt(
    baseline_metrics: dict[str, float],
    optimized_metrics: dict[str, float],
    optimized_params: dict,
) -> str:
    return f"""{SYSTEM_SOP}

Comparativo Random Forest — hiperparâmetros Fase 1 vs otimizados por algoritmo genético (mesmo pré-processamento e split 80/20):

Baseline Fase 1:
- recall: {baseline_metrics['recall']:.4f}
- f1: {baseline_metrics['f1']:.4f}
- roc_auc: {baseline_metrics['roc_auc']:.4f}

Otimizado (AG):
- recall: {optimized_metrics['recall']:.4f}
- f1: {optimized_metrics['f1']:.4f}
- roc_auc: {optimized_metrics['roc_auc']:.4f}

Hiperparâmetros AG: {optimized_params}

Contexto: na Fase 1 a Regressão Logística foi escolhida clinicamente pelo maior recall no teste.

Tarefa: relatório curto para coordenação do hospital universitário — ganhos, riscos, e se o RF otimizado justifica revisão do modelo em produção piloto.
"""
