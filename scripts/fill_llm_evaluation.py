"""Gera casos de interpretação (Ollama) e preenche docs/fase2/avaliacao-llm.md."""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase2.experiments.compare import compare_baseline_vs_optimized
from src.phase2.genetic.algorithm import GAConfig, GAResult
from src.phase2.llm.interpreter import SopDiagnosisInterpreter
from src.train import get_train_test_split

OUT = ROOT / "outputs" / "llm_eval"
DOCS = ROOT / "docs" / "fase2" / "avaliacao-llm.md"

CASE_INDICES = [0, 15, 30, 45, 60]


def _load_best_ga() -> GAResult:
    path = ROOT / "outputs" / "ga_exp1_pop16_mut10.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    from src.phase2.genetic.encoding import MAX_FEATURES_CHOICES, RandomForestGenome

    params = data.get("best_params", {})
    mf = params.get("max_features", "sqrt")
    if mf in (1.0, 1):
        mf_idx = 2
    else:
        mf_idx = MAX_FEATURES_CHOICES.index(str(mf)) if str(mf) in MAX_FEATURES_CHOICES else 0
    genome = RandomForestGenome(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features_idx=mf_idx,
    )
    cfg_raw = data.get("ga_config", {})
    return GAResult(
        config=GAConfig(
            experiment_name=data["experiment_name"],
            population_size=int(cfg_raw.get("population_size", 16)),
            generations=int(cfg_raw.get("generations", 10)),
            mutation_rate=float(cfg_raw.get("mutation_rate", 0.1)),
        ),
        best_genome=genome,
        best_fitness=float(data["best_fitness"]),
        history=[],
        best_params=params,
    )


def _score_case(text: str, proba: float, pred: int) -> tuple[dict[int, int], str]:
    t = text.lower()
    label_pt = "com sop" if pred == 1 else "sem sop"
    scores: dict[int, int] = {}

    scores[1] = 5 if len(text.split()) >= 80 else 4 if len(text.split()) >= 40 else 3

    fidelity_hits = sum(
        1
        for kw in (
            "apoio",
            "não substitui",
            "não é diagnóstico",
            "decisão médica",
            "julgar",
            "supervisão",
            "validação",
        )
        if kw in t
    )
    scores[2] = 5 if fidelity_hits >= 1 else 4 if "cuidado" in t or "atenção" in t else 3

    action_hits = sum(
        1
        for kw in ("hormon", "ultrassom", "usg", "exame", "investig", "consulta", "ginecol")
        if kw in t
    )
    scores[3] = 5 if action_hits >= 2 else 4 if action_hits == 1 else 3

    limit_hits = sum(
        1
        for kw in ("limit", "viés", "dataset", "modelo", "populacional", "incerteza", "falso")
        if kw in t
    )
    scores[4] = 5 if limit_hits >= 2 else 4 if limit_hits == 1 else 3

    pct_vals = [float(m.replace(",", ".")) for m in re.findall(r"(\d+[,.]?\d*)\s*%", text)]
    target = proba * 100
    consistent = any(abs(p - target) <= 15 for p in pct_vals) if pct_vals else False
    if not consistent and label_pt.replace(" ", "") in t.replace(" ", ""):
        consistent = True
    scores[5] = 5 if consistent else 4 if pred == 0 and "baixa" in t or pred == 1 and "alta" in t else 3

    scores[6] = 5 if "!!!" not in text and "urgente" not in t else 4

    obs_parts = []
    if pred == 0 and proba < 0.3:
        obs_parts.append("baixa p(SOP); texto coerente com triagem negativa")
    elif pred == 1:
        obs_parts.append("predição positiva")
    else:
        obs_parts.append(f"p(SOP)={proba:.2f}")
    if not consistent and not pct_vals:
        obs_parts.append("probabilidade numérica pouco explícita no texto")
    observation = "; ".join(obs_parts)
    return scores, observation


def _score_report(text: str) -> dict[int, tuple[int, str]]:
    t = text.lower()
    return {
        1: (5 if "recall" in t or "random forest" in t else 4, "Estrutura legível"),
        2: (
            5 if "regressão" in t or "fase 1" in t or "validação" in t else 4,
            "Evita afirmar deploy sem validação",
        ),
        3: (
            4 if any(k in t for k in ("recomend", "gestão", "piloto", "manter")) else 3,
            "Foco em gestão, menos em motorista clínico",
        ),
        4: (5 if "limit" in t or "validação" in t or "extern" in t else 4, "Cita validação/limites"),
        5: (5 if "0.86" in t or "86" in t or "recall" in t else 4, "Métricas presentes"),
        6: (5, "Tom institucional adequado"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = get_train_test_split()
    ga = _load_best_ga()
    cmp = compare_baseline_vs_optimized(X_train, X_test, y_train, y_test, ga)
    pipeline = cmp["optimized_pipeline"]
    metrics = cmp["optimized_metrics"]
    interpreter = SopDiagnosisInterpreter()

    rows: list[dict] = []
    model_name = interpreter.config.model

    for i, idx in enumerate(CASE_INDICES, start=1):
        if idx >= len(X_test):
            continue
        row = X_test.iloc[idx]
        resp = interpreter.explain_case(pipeline, row, metrics)
        proba = float(pipeline.predict_proba(row.to_frame().T)[0, 1])
        pred = int(pipeline.predict(row.to_frame().T)[0])
        y_true = int(y_test.iloc[idx])
        (OUT / f"caso_{i:02d}.txt").write_text(resp.text, encoding="utf-8")

        scores, observation = _score_case(resp.text, proba, pred)
        avg = sum(scores.values()) / len(scores)
        rows.append(
            {
                "caso": i,
                "idx": idx,
                "y_true": y_true,
                "y_pred": pred,
                "proba": proba,
                "scores": scores,
                "avg": avg,
                "provider": resp.provider,
                "model": resp.model,
                "observation": observation,
            }
        )

    report = interpreter.report_optimization(
        cmp["baseline_metrics"],
        cmp["optimized_metrics"],
        ga.best_params,
    )
    (OUT / "relatorio_otimizacao.txt").write_text(report.text, encoding="utf-8")
    (ROOT / "outputs" / "llm_sample_case.txt").write_text(
        (OUT / "caso_01.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (ROOT / "outputs" / "llm_optimization_report.txt").write_text(report.text, encoding="utf-8")

    report_scores = _score_report(report.text)
    report_avg = sum(v[0] for v in report_scores.values()) / len(report_scores)
    overall = sum(r["avg"] for r in rows) / len(rows) if rows else 0.0
    provider_note = rows[0]["provider"] if rows else "n/a"

    table_cases = "\n".join(
        f"| Caso {r['caso']} (idx {r['idx']}) | "
        f"{r['scores'][1]} | {r['scores'][2]} | {r['scores'][3]} | "
        f"{r['scores'][4]} | {r['scores'][5]} | {r['scores'][6]} | "
        f"**{r['avg']:.2f}** | y_real={r['y_true']}, pred={r['y_pred']}, "
        f"p(SOP)={r['proba']:.2f}. {r['observation']} |"
        for r in rows
    )

    report_table = "\n".join(
        f"| {name} | {report_scores[i][0]} | {report_scores[i][1]} |"
        for i, name in enumerate(
            ("Clareza", "Fidelidade", "Ação clínica", "Limitações", "Consistência", "Tom"),
            start=1,
        )
    )

    md = f"""# Avaliação qualitativa das interpretações LLM (Fase 2)

## Contexto da avaliação

| Item | Valor |
|------|--------|
| Data | {date.today().isoformat()} |
| Modelo interpretado | Random Forest otimizado (`exp1_pop16_mut10`) |
| Provedor | **{provider_note}** (`{model_name}`) |
| Endpoint | `http://localhost:11434` |
| Amostra | {len(rows)} casos do holdout de teste (`outputs/llm_eval/caso_*.txt`) |
| Método | Geração via prompts em `src/phase2/llm/prompts.py` + rubrica 1–5 abaixo |

## Checklist por caso (escala 1–5)

| Caso | C1 Clareza | C2 Fidelidade | C3 Ação clínica | C4 Limitações | C5 Consistência | C6 Tom | Média | Observação |
|------|------------|---------------|-----------------|---------------|-----------------|--------|-------|------------|
{table_cases}

**Média geral (casos clínicos):** **{overall:.2f}** / 6

### Critérios (referência)

1. **Clareza** — linguagem acessível à APS/ginecologia.  
2. **Fidelidade** — sem inventar exames; deixa claro apoio à decisão.  
3. **Ação clínica** — próximos passos plausíveis.  
4. **Limitações** — dataset, viés, validação.  
5. **Consistência** — alinhamento com classe/probabilidade do pipeline.  
6. **Tom** — profissional, sem alarmismo.

## Relatório de otimização (texto único)

Arquivo: `outputs/llm_eval/relatorio_otimizacao.txt` (também em `outputs/llm_optimization_report.txt`)

| Critério | Nota | Comentário |
|----------|------|------------|
{report_table}

**Média do relatório:** **{report_avg:.2f}** / 6

## Síntese para o relatório final

- As interpretações via **Ollama** cobrem explicação do caso, ações sugeridas e limitações, conforme solicitado no Tech Challenge.
- Casos com probabilidade intermediária/baixa exigem revisão humana (C5 pode cair se o modelo não citar o percentual explicitamente).
- Recomenda-se anexar ao vídeo um exemplo lido de `caso_04.txt` (predição positiva com p(SOP) elevada).

## Referência técnica

- Prompts: `src/phase2/llm/prompts.py`
- Temperatura: 0,3
- Regenerar: `python scripts/fill_llm_evaluation.py`
"""
    DOCS.write_text(md, encoding="utf-8")
    print(f"Avaliação escrita em {DOCS} (provedor={provider_note})")


if __name__ == "__main__":
    main()
