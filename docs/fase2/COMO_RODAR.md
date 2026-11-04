# Fase 2 — Tech Challenge (Projeto 1)

## Pré-requisitos

Mesmos da Fase 1: Python 3.10+, dataset em `data/raw/` (`python scripts/download_data.py`).

Opcional: [Ollama](https://ollama.com/) + `ollama pull llama3.2` (LLM gratuita local).

## Comandos

```bash
pip install -r requirements.txt
python -m src.train          # Fase 1 (se ainda não rodou)
python -m src.phase2.cli --quick
python -m src.phase2.experiments.run_experiments   # 3 experimentos AG
python scripts/fill_llm_evaluation.py              # Ollama + avaliacao-llm.md
python scripts/generate_relatorio_fase2.py
python scripts/benchmark_scaling.py              # demo escalabilidade (opcional)
pytest
```

Artefatos: `outputs/`, `logs/`, figura `outputs/ga_convergence.png`.

Relatório: `docs/fase2/relatorio-tecnico.md` (gerado pelo script acima). Checklist LLM: `docs/fase2/avaliacao-llm.md`.
