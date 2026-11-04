"""Gera docs/fase2/relatorio-tecnico.md a partir de outputs/ e artefatos da Fase 1."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs" / "fase2"
GITHUB_REPO_URL = "https://github.com/brenda-guinarte/pos-fiap-ia-para-devs"
FASE2_VIDEO_URL = "https://youtu.be/dGP4gHi4lHA"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt4(x: float) -> str:
    return f"{x:.4f}"


def main() -> None:
    ga_files = sorted(OUTPUT_DIR.glob("ga_exp*.json"))
    if not ga_files:
        ga_files = sorted(OUTPUT_DIR.glob("ga_*.json"))

    experiments: list[dict] = []
    best_name = ""
    best_fitness = -1.0
    for path in ga_files:
        data = _load_json(path) or {}
        name = data.get("experiment_name", path.stem.replace("ga_", ""))
        fitness = float(data.get("best_fitness", 0))
        cfg = data.get("ga_config", {})
        experiments.append(
            {
                "name": name,
                "pop": cfg.get("population_size", "?"),
                "mut": cfg.get("mutation_rate", "?"),
                "gen": cfg.get("generations", "?"),
                "fitness": fitness,
                "params": data.get("best_params", {}),
            }
        )
        if fitness > best_fitness and name.startswith("exp"):
            best_fitness = fitness
            best_name = name

    if not best_name and experiments:
        best_name = max(experiments, key=lambda e: e["fitness"])["name"]

    comparison_path = OUTPUT_DIR / f"comparison_{best_name}.json"
    if not comparison_path.exists():
        comparisons = sorted(OUTPUT_DIR.glob("comparison_*.json"))
        comparison_path = comparisons[-1] if comparisons else Path()

    comparison = _load_json(comparison_path) or {}
    baseline = comparison.get("baseline_fase1", {})
    optimized = comparison.get("otimizado_ag", {})
    delta = comparison.get("delta", {})

    fase1 = _load_json(PROCESSED / "resumo_resultados.json") or {}
    lr_metrics = next(
        (m for m in fase1.get("metricas", []) if m.get("modelo") == "Regressão Logística"),
        {},
    )
    rf_fase1_row = next(
        (m for m in fase1.get("metricas", []) if m.get("modelo") == "Random Forest"),
        {},
    )

    llm_case = (OUTPUT_DIR / "llm_sample_case.txt").exists()
    llm_report = (OUTPUT_DIR / "llm_optimization_report.txt").exists()
    convergence = (OUTPUT_DIR / "ga_convergence.png").exists()
    avaliacao_path = DOCS / "avaliacao-llm.md"
    avaliacao_text = (
        avaliacao_path.read_text(encoding="utf-8") if avaliacao_path.exists() else ""
    )
    m_casos = re.search(
        r"Média geral \(casos clínicos\):\*\* \*\*([\d.]+)\*\*", avaliacao_text
    )
    m_rel = re.search(r"Média do relatório:\*\* \*\*([\d.]+)\*\*", avaliacao_text)
    llm_mean_cases = m_casos.group(1) if m_casos else "—"
    llm_mean_report = m_rel.group(1) if m_rel else "—"

    exp_table = "\n".join(
        f"| {e['name']} | {e['pop']} | {e['mut']} | {e['gen']} | {_fmt4(e['fitness'])} |"
        for e in experiments
        if str(e["name"]).startswith("exp")
    )
    if not exp_table:
        exp_table = "\n".join(
            f"| {e['name']} | {e['pop']} | {e['mut']} | {e['gen']} | {_fmt4(e['fitness'])} |"
            for e in experiments
        )

    best_exp = next((e for e in experiments if e["name"] == best_name), experiments[0] if experiments else {})

    md = f"""# Relatório técnico — Tech Challenge Fase 2 (Projeto 1)

**Repositório GitHub:** {GITHUB_REPO_URL}  
**Vídeo Fase 2:** {FASE2_VIDEO_URL}

_Gerado automaticamente por `scripts/generate_relatorio_fase2.py`. Revise texto narrativo e checklist LLM antes da entrega._

## 1. Continuidade com a Fase 1

| Item | Detalhe |
|------|---------|
| Repositório | [pos-fiap-ia-para-devs](https://github.com/brenda-guinarte/pos-fiap-ia-para-devs) |
| Problema | Triagem de Síndrome dos Ovários Policísticos (SOP/PCOS) |
| Dataset | 541 pacientes, Kerala (Índia); alvo `SOP (S/N)` |
| Split | 80% treino / 20% teste, `random_state=42` (idêntico à Fase 1) |
| Modelo clínico Fase 1 | **{fase1.get('melhor_modelo', 'Regressão Logística')}** — critério: {fase1.get('criterio_selecao', 'recall, F1, ROC-AUC')} |

### Métricas Fase 1 no holdout de teste

| Modelo | Accuracy | Precisão | Recall | F1 | ROC-AUC |
|--------|----------|----------|--------|-----|---------|
| Regressão Logística | {_fmt4(lr_metrics.get('teste_accuracy', 0))} | {_fmt4(lr_metrics.get('teste_precision', 0))} | {_fmt4(lr_metrics.get('teste_recall', 0))} | {_fmt4(lr_metrics.get('teste_f1', 0))} | {_fmt4(lr_metrics.get('teste_roc_auc', 0))} |
| Random Forest (baseline Fase 1) | {_fmt4(rf_fase1_row.get('teste_accuracy', 0))} | {_fmt4(rf_fase1_row.get('teste_precision', 0))} | {_fmt4(rf_fase1_row.get('teste_recall', 0))} | {_fmt4(rf_fase1_row.get('teste_f1', 0))} | {_fmt4(rf_fase1_row.get('teste_roc_auc', 0))} |

Na Fase 2 o algoritmo genético otimiza hiperparâmetros do **Random Forest**, mantendo o mesmo `ColumnTransformer` e imputação da Fase 1.

## 2. Algoritmo genético

### Codificação (genes)

Cada indivíduo é um `RandomForestGenome` com cinco genes inteiros: `n_estimators` (50–600), `max_depth` (3–20), `min_samples_split`, `min_samples_leaf`, `max_features` (categórico: sqrt / log2 / 1.0). Implementação: `src/phase2/genetic/encoding.py`.

### Operadores

- **Seleção:** torneio (tamanho configurável, padrão 3).
- **Cruzamento:** uniforme gene a gene.
- **Mutação:** probabilística por gene, com perturbação discreta nos hiperparâmetros.
- **Elitismo:** os 2 melhores indivíduos passam intactos para a próxima geração.

### Função fitness

Validação cruzada estratificada **3-fold apenas no conjunto de treino** (evita vazamento do teste). Score composto alinhado à prioridade clínica da Fase 1:

`fitness = 0,5 × recall + 0,3 × F1 + 0,2 × ROC-AUC` (médias no CV).

### Três experimentos obrigatórios

| Experimento | População | Taxa mutação | Gerações | Melhor fitness (CV) |
|-------------|-----------|--------------|----------|---------------------|
{exp_table}

**Melhor configuração entre experimentos:** `{best_name}` (fitness CV = {_fmt4(best_fitness if best_fitness >= 0 else best_exp.get('fitness', 0))}).

Hiperparâmetros do melhor genoma:

```json
{json.dumps(best_exp.get('params', {}), indent=2, ensure_ascii=False)}
```

Figura de convergência: {"`outputs/ga_convergence.png`" if convergence else "_não encontrada — execute run_experiments_"}

## 3. Comparativo baseline vs otimizado (teste 20%)

Arquivo: `{comparison_path.name if comparison_path else 'N/A'}`

| Métrica | RF baseline (Fase 1) | RF otimizado (AG) | Δ |
|---------|-------------------|-------------------|---|
| Accuracy | {_fmt4(baseline.get('accuracy', 0))} | {_fmt4(optimized.get('accuracy', 0))} | {_fmt4(delta.get('accuracy', 0))} |
| Precisão | {_fmt4(baseline.get('precision', 0))} | {_fmt4(optimized.get('precision', 0))} | {_fmt4(delta.get('precision', 0))} |
| Recall | {_fmt4(baseline.get('recall', 0))} | {_fmt4(optimized.get('recall', 0))} | {_fmt4(delta.get('recall', 0))} |
| F1 | {_fmt4(baseline.get('f1', 0))} | {_fmt4(optimized.get('f1', 0))} | {_fmt4(delta.get('f1', 0))} |
| ROC-AUC | {_fmt4(baseline.get('roc_auc', 0))} | {_fmt4(optimized.get('roc_auc', 0))} | {_fmt4(delta.get('roc_auc', 0))} |

### RF otimizado vs Regressão Logística (modelo escolhido na Fase 1)

| Métrica | LR (Fase 1, teste) | RF otimizado (AG, teste) |
|---------|---------------------|---------------------------|
| Recall | {_fmt4(lr_metrics.get('teste_recall', 0))} | {_fmt4(optimized.get('recall', 0))} |
| F1 | {_fmt4(lr_metrics.get('teste_f1', 0))} | {_fmt4(optimized.get('f1', 0))} |
| ROC-AUC | {_fmt4(lr_metrics.get('teste_roc_auc', 0))} | {_fmt4(optimized.get('roc_auc', 0))} |

**Leitura clínica:** a Fase 1 priorizou **recall** para reduzir falsos negativos em SOP. Se o RF otimizado não superar a LR em recall no teste, o piloto deve **manter a LR** como modelo principal e usar o RF otimizado como linha de pesquisa/comparativa. Se superar, justificar validação externa antes de qualquer troca em produção.

## 4. Integração com LLM

- **Provedor:** Ollama local (`llama3.2`), sem API paga; variáveis em `.env.example`.
- **Prompts:** contexto SOP, apoio à decisão, três blocos (explicação, ações, limitações) — `src/phase2/llm/prompts.py`.
- **Artefatos:** {"disponíveis" if llm_case else "pendentes"} em `outputs/llm_sample_case.txt`, `outputs/llm_eval/caso_*.txt`, `outputs/llm_optimization_report.txt`.
- **Avaliação qualitativa:** `docs/fase2/avaliacao-llm.md` — média casos **{llm_mean_cases}/6**, relatório de otimização **{llm_mean_report}/6** (5 casos holdout + rubrica 1–5).
- **Regenerar:** `python scripts/fill_llm_evaluation.py` (requer Ollama ativo).

## 5. Monitoramento e escalabilidade

- **Logging:** `logs/pcos_phase2.log` e `logs/performance.jsonl` (evento `ga_generation` com `duration_seconds`, `fitness_workers`, `population_size`).
- **Escalabilidade implementada:** avaliação paralela de fitness por geração via `src/phase2/scaling/executor.py` e variável `GA_FITNESS_WORKERS` (padrão 4). Cada indivíduo da população equivale a um job na fila local; aumentar workers reduz tempo por geração em CPU multi-core.
- **Demonstração:** `python scripts/benchmark_scaling.py` (compara 1 vs 4 workers em uma geração).
- **Evolução em nuvem (opcional/extra):** mesma fila em SQS + workers containerizados com autoscaler — não implantado; arquitetura alvo descrita em `docs/fase2/arquitetura.md`.

## 6. Desafios e soluções

| Desafio | Solução adotada |
|---------|-----------------|
| Custo computacional do AG | Paralelismo de fitness; população/gerações moderadas; `--quick` para demo |
| Risco de vazamento de dados | Fitness só com CV no **treino**; comparativo final no **teste** fixo |
| LLM sem budget | Ollama local; template estruturado apenas para testes offline |
| Continuidade Fase 1 | `load_data`, `ColumnTransformer` e `BASELINE_RF_HYPERPARAMS` compartilhados |
| Modelo clínico vs RF otimizado | LR mantida no piloto se recall no teste permanecer superior (ver §3) |

## 7. Como reproduzir

```bash
python scripts/download_data.py
python -m src.train
python -m src.phase2.experiments.run_experiments
python scripts/fill_llm_evaluation.py
python scripts/generate_relatorio_fase2.py
pytest
```

## 8. Vídeo de demonstração (Fase 2)

- **URL:** {FASE2_VIDEO_URL}
- **Conteúdo:** execução do sistema, algoritmo genético, comparativo baseline vs otimizado, interpretação LLM (Ollama ou artefatos gerados), arquitetura.
"""
    out_path = DOCS / "relatorio-tecnico.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Relatório escrito em {out_path}")


if __name__ == "__main__":
    main()
