# Relatório técnico — Tech Challenge Fase 2 (Projeto 1)

**Repositório GitHub:** https://github.com/brenda-guinarte/pos-fiap-ia-para-devs  
**Vídeo Fase 2:** https://youtu.be/dGP4gHi4lHA

_Gerado automaticamente por `scripts/generate_relatorio_fase2.py`. Revise texto narrativo e checklist LLM antes da entrega._

## 1. Continuidade com a Fase 1

| Item | Detalhe |
|------|---------|
| Repositório | [pos-fiap-ia-para-devs](https://github.com/brenda-guinarte/pos-fiap-ia-para-devs) |
| Problema | Triagem de Síndrome dos Ovários Policísticos (SOP/PCOS) |
| Dataset | 541 pacientes, Kerala (Índia); alvo `SOP (S/N)` |
| Split | 80% treino / 20% teste, `random_state=42` (idêntico à Fase 1) |
| Modelo clínico Fase 1 | **Regressão Logística** — critério: recall, seguido de F1 e ROC-AUC no teste |

### Métricas Fase 1 no holdout de teste

| Modelo | Accuracy | Precisão | Recall | F1 | ROC-AUC |
|--------|----------|----------|--------|-----|---------|
| Regressão Logística | 0.8716 | 0.7619 | 0.8889 | 0.8205 | 0.9505 |
| Random Forest (baseline Fase 1) | 0.9083 | 0.8824 | 0.8333 | 0.8571 | 0.9444 |

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
| exp1_pop16_mut10 | 16 | 0.1 | 10 | 0.8733 |
| exp2_pop24_mut15 | 24 | 0.15 | 10 | 0.8723 |
| exp3_pop20_mut25 | 20 | 0.25 | 10 | 0.8724 |

**Melhor configuração entre experimentos:** `exp1_pop16_mut10` (fitness CV = 0.8733).

Hiperparâmetros do melhor genoma:

```json
{
  "n_estimators": 530,
  "max_depth": 18,
  "min_samples_split": 18,
  "min_samples_leaf": 8,
  "max_features": "sqrt"
}
```

Figura de convergência: `outputs/ga_convergence.png`

## 3. Comparativo baseline vs otimizado (teste 20%)

Arquivo: `comparison_exp1_pop16_mut10.json`

| Métrica | RF baseline (Fase 1) | RF otimizado (AG) | Δ |
|---------|-------------------|-------------------|---|
| Accuracy | 0.9083 | 0.8899 | -0.0183 |
| Precisão | 0.8824 | 0.8158 | -0.0666 |
| Recall | 0.8333 | 0.8611 | 0.0278 |
| F1 | 0.8571 | 0.8378 | -0.0193 |
| ROC-AUC | 0.9444 | 0.9486 | 0.0042 |

### RF otimizado vs Regressão Logística (modelo escolhido na Fase 1)

| Métrica | LR (Fase 1, teste) | RF otimizado (AG, teste) |
|---------|---------------------|---------------------------|
| Recall | 0.8889 | 0.8611 |
| F1 | 0.8205 | 0.8378 |
| ROC-AUC | 0.9505 | 0.9486 |

**Leitura clínica:** a Fase 1 priorizou **recall** para reduzir falsos negativos em SOP. Se o RF otimizado não superar a LR em recall no teste, o piloto deve **manter a LR** como modelo principal e usar o RF otimizado como linha de pesquisa/comparativa. Se superar, justificar validação externa antes de qualquer troca em produção.

## 4. Integração com LLM

- **Provedor:** Ollama local (`llama3.2`), sem API paga; variáveis em `.env.example`.
- **Prompts:** contexto SOP, apoio à decisão, três blocos (explicação, ações, limitações) — `src/phase2/llm/prompts.py`.
- **Artefatos:** disponíveis em `outputs/llm_sample_case.txt`, `outputs/llm_eval/caso_*.txt`, `outputs/llm_optimization_report.txt`.
- **Avaliação qualitativa:** `docs/fase2/avaliacao-llm.md` — média casos **4.93/6**, relatório de otimização **4.67/6** (5 casos holdout + rubrica 1–5).
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

- **URL:** https://youtu.be/dGP4gHi4lHA
- **Conteúdo:** execução do sistema, algoritmo genético, comparativo baseline vs otimizado, interpretação LLM (Ollama ou artefatos gerados), arquitetura.