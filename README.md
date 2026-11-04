# Tech Challenge — Detecção de Síndrome dos Ovários Policísticos

**Repositório GitHub:** https://github.com/brenda-guinarte/pos-fiap-ia-para-devs  
**Vídeo Fase 2 (Tech Challenge):** https://youtu.be/dGP4gHi4lHA

Projeto da Fase 1 para desenvolver uma solução inicial de apoio à triagem de
Síndrome dos Ovários Policísticos (PCOS/SOP) a partir de dados clínicos
estruturados.

> Este é um projeto acadêmico. O modelo não realiza diagnóstico e não deve ser
> usado em decisões clínicas sem validação externa e supervisão médica.

## Problema

A SOP é uma condição hormonal que pode afetar ciclo menstrual, fertilidade e
saúde metabólica. O projeto compara modelos de classificação para identificar
pacientes com maior probabilidade de SOP e apoiar a priorização da avaliação
médica. Como um falso negativo pode atrasar investigação e tratamento, o
**recall da classe positiva** é a métrica prioritária, acompanhado de F1,
ROC-AUC, precisão e accuracy.

## Dataset

- Fonte: [Polycystic ovary syndrome (PCOS) — Kaggle](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos)
- Autor: Prasoon Kottarathil (2020)
- Base principal: `PCOS_data_without_infertility.xlsx`
- Registros: 541 pacientes
- Alvo: `SOP (S/N)` — 0 sem SOP; 1 com SOP
- As variáveis são traduzidas para português no carregamento (`src/train.py`)
- Contexto: dados físicos e clínicos coletados em dez hospitais de Kerala, Índia

Os dados não são versionados no Git. O script de download usa a API pública do
Kaggle e não exige token:

```bash
python scripts/download_data.py
```



## Metodologia

1. Limpeza de identificadores, coluna residual e valores não numéricos.
2. Imputação pela mediana para variáveis numéricas.
3. Imputação pela moda e One-Hot Encoding para grupo sanguíneo e ciclo.
4. Padronização das variáveis numéricas.
5. Separação estratificada: 80% treino e 20% teste, `random_state=42`.
6. Validação cruzada estratificada com cinco folds no conjunto de treino.
7. Comparação entre Regressão Logística e Random Forest.
8. Seleção por recall, seguida de F1 e ROC-AUC no teste.
9. Explicabilidade com importância global e SHAP.

Todo pré-processamento é ajustado somente nos dados de treino dentro de
`Pipeline`/`ColumnTransformer`, evitando vazamento de dados.

## Resultados atuais


| Modelo              | Accuracy  | Precisão  | Recall    | F1        | ROC-AUC   |
| ------------------- | --------- | --------- | --------- | --------- | --------- |
| Regressão Logística | 0,872     | 0,762     | **0,889** | 0,821     | **0,951** |
| Random Forest       | **0,899** | **0,838** | 0,861     | **0,849** | 0,941     |


A Regressão Logística foi selecionada pelo critério clínico definido: maior
recall da classe com SOP. Os números são resultados acadêmicos nesta divisão do
dataset e não representam desempenho clínico generalizável.

## Como executar



### Pré-requisitos

- Python 3.10 ou superior
- Git



### Instalação tradicional

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Se o sistema não tiver `python3-venv`, também é possível usar
[uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```



### Execução completa

```bash
python scripts/download_data.py
python -m src.train
pytest
jupyter notebook notebooks/01_analise_pcos.ipynb
```

O dataset já pode estar presente após o download; rodá-lo novamente apenas
substitui os arquivos locais.

## Artefatos gerados

- `data/processed/comparacao_modelos.csv`: métricas de teste e validação
- `data/processed/resumo_resultados.json`: resumo legível por programa
- `data/processed/qualidade_dados.csv`: ausências e cardinalidade
- `data/processed/matriz_correlacao.csv`: correlações
- `data/processed/feature_importance.csv`: importância das variáveis
- `reports/figures/`: EDA, matrizes de confusão, importância e SHAP
- `models/modelo_pcos.joblib`: pipeline treinado



## Estrutura

```text
.
├── data/
│   ├── raw/                  # arquivos originais (ignorados pelo Git)
│   └── processed/            # métricas e tabelas geradas
├── models/                   # modelo serializado
├── notebooks/
│   └── 01_analise_pcos.ipynb
├── reports/
│   ├── figures/
│   └── relatorio_tecnico.md
├── scripts/
│   └── download_data.py
├── src/
│   └── train.py
├── tests/
├── README.md
└── requirements.txt
```



## Limitações e uso responsável

- A amostra é pequena e de uma região específica da Índia.
- O desempenho pode mudar em populações com características diferentes.
- Alguns atributos são componentes usados na investigação clínica da SOP, o
que exige cuidado ao definir em que etapa da triagem o modelo seria aplicado.
- Há valores ausentes e inconsistências de preenchimento.
- É necessária validação externa, calibração, avaliação de viés e monitoramento
antes de qualquer uso prático.
- Dados médicos são sensíveis e devem seguir LGPD, controle de acesso e
anonimização.
- A decisão final deve permanecer com a equipe médica.



## Fase 2 — Tech Challenge (Projeto 1)

Continuação direta deste repositório: **algoritmo genético** para hiperparâmetros do
Random Forest (mesmo pré-processamento e split da Fase 1), **comparativo** com o baseline,
**logging** de desempenho e **LLM local gratuita** (Ollama) para interpretação em
linguagem natural.

```bash
python -m src.phase2.cli --quick
python -m src.phase2.experiments.run_experiments
```

Documentação: [`docs/fase2/COMO_RODAR.md`](docs/fase2/COMO_RODAR.md),
[`docs/fase2/arquitetura.md`](docs/fase2/arquitetura.md),
[`docs/fase2/relatorio-tecnico.md`](docs/fase2/relatorio-tecnico.md) e
[`docs/fase2/avaliacao-llm.md`](docs/fase2/avaliacao-llm.md).

Relatório e vídeo de entrega: links no [topo deste README](#tech-challenge--detecção-de-síndrome-dos-ovários-policísticos).

## Próximos passos

- Avaliar calibração das probabilidades e ajuste de limiar para recall.
- Realizar validação externa em dados brasileiros.
- Avaliar estabilidade das explicações e possíveis vieses por subgrupo.
- Criar uma interface de demonstração para triagem assistida.



## Autores

- Brenda Guinarte — RM376294

## Vídeos

| Fase | Conteúdo | Link |
|------|----------|------|
| **1** | Triagem PCOS/SOP — modelos, métricas e EDA | https://youtu.be/rRfgysj36dU |
| **2** | Tech Challenge — algoritmo genético, comparativo e LLM (Ollama) | https://youtu.be/dGP4gHi4lHA |

Relatório técnico Fase 2 (inclui o link do vídeo §8): [`docs/fase2/relatorio-tecnico.md`](docs/fase2/relatorio-tecnico.md).
