# Avaliação qualitativa das interpretações LLM (Fase 2)

## Contexto da avaliação

| Item | Valor |
|------|--------|
| Data | 2026-10-22 |
| Modelo interpretado | Random Forest otimizado (`exp1_pop16_mut10`) |
| Provedor | **ollama** (`llama3.2`) |
| Endpoint | `http://localhost:11434` |
| Amostra | 5 casos do holdout de teste (`outputs/llm_eval/caso_*.txt`) |
| Método | Geração via prompts em `src/phase2/llm/prompts.py` + rubrica 1–5 abaixo |

## Checklist por caso (escala 1–5)

| Caso | C1 Clareza | C2 Fidelidade | C3 Ação clínica | C4 Limitações | C5 Consistência | C6 Tom | Média | Observação |
|------|------------|---------------|-----------------|---------------|-----------------|--------|-------|------------|
| Caso 1 (idx 0) | 5 | 5 | 5 | 5 | 5 | 5 | **5.00** | y_real=1, pred=0, p(SOP)=0.13. baixa p(SOP); texto coerente com triagem negativa |
| Caso 2 (idx 15) | 5 | 5 | 5 | 5 | 5 | 5 | **5.00** | y_real=0, pred=0, p(SOP)=0.15. baixa p(SOP); texto coerente com triagem negativa |
| Caso 3 (idx 30) | 5 | 5 | 5 | 5 | 5 | 5 | **5.00** | y_real=0, pred=0, p(SOP)=0.21. baixa p(SOP); texto coerente com triagem negativa |
| Caso 4 (idx 45) | 5 | 5 | 5 | 5 | 5 | 5 | **5.00** | y_real=1, pred=1, p(SOP)=0.76. predição positiva |
| Caso 5 (idx 60) | 5 | 5 | 5 | 5 | 3 | 5 | **4.67** | y_real=0, pred=0, p(SOP)=0.13. baixa p(SOP); texto coerente com triagem negativa; probabilidade numérica pouco explícita no texto |

**Média geral (casos clínicos):** **4.93** / 6

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
| Clareza | 5 | Estrutura legível |
| Fidelidade | 5 | Evita afirmar deploy sem validação |
| Ação clínica | 4 | Foco em gestão, menos em motorista clínico |
| Limitações | 4 | Cita validação/limites |
| Consistência | 5 | Métricas presentes |
| Tom | 5 | Tom institucional adequado |

**Média do relatório:** **4.67** / 6

## Síntese para o relatório final

- As interpretações via **Ollama** cobrem explicação do caso, ações sugeridas e limitações, conforme solicitado no Tech Challenge.
- Casos com probabilidade intermediária/baixa exigem revisão humana (C5 pode cair se o modelo não citar o percentual explicitamente).
- Recomenda-se anexar ao vídeo um exemplo lido de `caso_04.txt` (predição positiva com p(SOP) elevada).

## Referência técnica

- Prompts: `src/phase2/llm/prompts.py`
- Temperatura: 0,3
- Regenerar: `python scripts/fill_llm_evaluation.py`
