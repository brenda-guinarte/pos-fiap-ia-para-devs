# Como rodar e testar o projeto

Guia rápido de execução do projeto de detecção de SOP (PCOS). Todos os comandos
são executados na raiz do repositório.

```bash
cd /home/brendaguinarte/Documentos/tech-challenge-pcos
```

## 1. Ativar o ambiente

Se o `.venv` já existe:

```bash
source .venv/bin/activate
```

Se ainda não existir, crie e instale as dependências:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

> Alternativa sem `uv` (precisa de `python3-venv`):
>
> ```bash
> python3 -m venv .venv
> source .venv/bin/activate
> python -m pip install -r requirements.txt
> ```

## 2. Baixar o dataset (se ainda não tiver)

```bash
python scripts/download_data.py
```

## 3. Rodar o treino completo

Gera métricas, gráficos, SHAP e o modelo serializado:

```bash
python -m src.train
```

Saída esperada (aproximada):

- Regressão Logística com recall ~0,89
- Random Forest com accuracy ~0,90
- Mensagem de artefatos salvos em `data/processed`, `reports/figures` e `models`

## 4. Rodar os testes automatizados

```bash
pytest
```

Esperado: `2 passed`.

## 5. Abrir o notebook

```bash
jupyter notebook notebooks/01_analise_pcos.ipynb
```

Rode as células na ordem. A célula que chama `main()` executa o fluxo completo.

## O que conferir

| Artefato | Onde |
|---|---|
| Comparação dos modelos | `data/processed/comparacao_modelos.csv` |
| Resumo JSON | `data/processed/resumo_resultados.json` |
| Gráficos / SHAP | `reports/figures/` |
| Modelo salvo | `models/modelo_pcos.joblib` |
