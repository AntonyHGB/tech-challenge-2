# Sistema de Recomendação para E-commerce

Sistema de recomendação de produtos para e-commerce baseado no comportamento de
navegação dos usuários. O modelo central é uma **rede neural (MLP com
embeddings)** treinada com **PyTorch**, comparada contra **baselines do
Scikit-Learn**, com pipeline reprodutível versionado em **DVC**, experimentos
rastreados no **MLflow** (incluindo Model Registry) e execução containerizada em
**Docker**.

Projeto desenvolvido para o **Tech Challenge — Fase 02**.

> **Status:** as quatro etapas estão concluídas. O pipeline roda de ponta a ponta
> com `dvc repro`, gera runs no MLflow e promove o melhor modelo a `Production`.

## Contexto do problema

Uma empresa de e-commerce precisa recomendar produtos a partir das interações
usuário–item. O objetivo é treinar um modelo de recomendação e compará-lo, com
pelo menos quatro métricas, contra baselines clássicos — tudo de forma
reprodutível e seguindo boas práticas de engenharia.

O dataset é o **MovieLens `ml-latest-small`** (GroupLens), usado como proxy
público de interações usuário–item: **90.274 interações**, 610 usuários e 3.650
itens após a limpeza (acima do mínimo de 10.000 exigido). A interação é tratada
como **feedback implícito**: nota maior ou igual a 4,0 é um sinal positivo.

## Resultados

Métricas no split de teste (8.767 interações, 561 usuários avaliados no ranking):

| Modelo | ROC AUC | Acurácia | F1 | Log loss | Precision@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| **mlp** (PyTorch, promovido) | **0,7593** | **0,6948** | **0,6941** | **0,5899** | 0,6552 | 0,8163 |
| logistic (Scikit-Learn) | 0,7468 | 0,6859 | 0,6794 | 0,6049 | 0,6575 | 0,8160 |
| popularity (baseline) | 0,6762 | 0,6261 | 0,6258 | 0,6452 | 0,6561 | 0,8222 |

A rede neural vence na métrica primária (ROC AUC) e é promovida a `Production`
no Model Registry. Os números completos ficam em `reports/metrics.json` e
`reports/comparison.md`, regenerados a cada `dvc repro`. A análise de
performance, limitações e vieses está no **[Model Card](docs/model_card.md)**.

## Tecnologias

| Camada | Ferramenta |
| --- | --- |
| Modelo neural | PyTorch |
| Baselines, pré-processamento e métricas | Scikit-Learn |
| Tracking de experimentos e Model Registry | MLflow |
| Versionamento de dados e pipeline | DVC |
| Gerenciamento de dependências | Poetry |
| Configuração tipada | Pydantic Settings |
| Lint e formatação | ruff |
| Testes | pytest |
| Containerização | Docker (multi-stage) + Docker Compose |

## Estrutura do projeto

```
src/recsys/
  config.py             # Settings (Pydantic) — configuração tipada lida do .env
  seeding.py            # set_global_seed — reprodutibilidade em random/numpy/torch
  metrics.py            # Métricas de classificação e de ranking
  ranker.py             # TopKRanker — geração de recomendações top-k
  data/                 # Ingestão: download, limpeza, encoders, split, arrays
  features/             # Rótulo implícito, escala e o feature store persistido
  preprocessing/        # Padrão Strategy + Template Method (scalers do Scikit-Learn)
  models/               # Interface comum, Factory, MLP (PyTorch), baselines
  tracking/             # Fachada do MLflow, wrapper pyfunc e promoção no Registry
  pipelines/            # Os stages do DVC + contexto, caminhos e parâmetros
scripts/
  validate_env.py       # Validação de ambiente (versão, pacotes, settings)
  download_dataset.py   # Baixa o dataset bruto para ser versionado no DVC
configs/params.yaml     # Hiperparâmetros do pipeline (rastreados pelo DVC)
dvc.yaml / dvc.lock     # Pipeline reprodutível (6 stages)
Dockerfile              # Build multi-stage (builder de deps + runtime slim)
docker-compose.yml      # Serviço de treino + servidor MLflow
docs/model_card.md      # Model Card (performance, limitações, vieses)
reports/                # Métricas, comparação e recomendações de exemplo
tests/                  # Suíte de testes (pytest)
data/ models/           # Artefatos versionados via DVC, fora do git
```

## Padrões de projeto aplicados

- **Factory** — `ModelFactory` constrói os recomendadores a partir de uma chave de
  texto. Novos modelos entram por registro (`models/factory.py`), sem ramificar
  a lógica com `if/else`. O mapeamento de configuração para argumentos também é
  por registro (`pipelines/params.py`).
- **Strategy** — `PreprocessingStrategy` permite trocar o algoritmo de
  pré-processamento de cada coluna sem alterar o `PreprocessingPipeline` que o
  consome, e a estratégia usada vem do `params.yaml`.
- **Template Method** — `SklearnScalerStrategy` implementa o esqueleto
  fit/transform (validação, reshape, guarda de estado) e delega apenas a criação
  do scaler às subclasses.

A interface comum `RecommenderModel` mantém pipeline, avaliação e tracking
desacoplados de qualquer framework específico (Inversão de Dependência), e o
ranqueamento vive no `TopKRanker`, fora dos modelos (Responsabilidade Única).

## Como executar

Requer **Python 3.12**. O ambiente é reproduzido a partir do `poetry.lock`
commitado.

### 1. Instalação do zero

```bash
poetry install                 # cria o virtualenv e instala prod + dev a partir do lock
cp .env.example .env           # configurações locais (o .env é git-ignored)
poetry run python scripts/validate_env.py   # checa Python, pacotes e settings
```

### 2. Pipeline completo

```bash
poetry run python scripts/download_dataset.py   # baixa data/raw/ratings.csv
poetry run dvc repro                            # executa os 6 stages
poetry run dvc push                             # envia dados/modelos ao remote
```

O `dvc repro` executa, na ordem: limpeza dos dados, engenharia de features,
treino dos três modelos e avaliação comparativa com promoção no Model Registry.
Cada stage loga parâmetros, métricas e artefatos no MLflow.

Para inspecionar os experimentos:

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

### 3. Execução em containers

```bash
docker compose up --build      # sobe o MLflow server e roda o pipeline de treino
```

O serviço `mlflow` expõe a UI e o Model Registry em `http://localhost:5000`
(backend SQLite em volume nomeado). O serviço `trainer` aguarda o healthcheck do
MLflow e executa `dvc repro` apontando o tracking para `http://mlflow:5000`. Os
diretórios `data/`, `models/`, `reports/`, o cache do DVC e o repositório git
(de onde o DVC resolve o estado do pipeline) são montados do host, de modo que os
artefatos gerados dentro do container ficam disponíveis fora dele.

O `Dockerfile` é multi-stage: o estágio *builder* resolve as dependências de
produção com Poetry a partir do lock file e o estágio *runtime* copia apenas o
virtualenv e o código, rodando como usuário não-root.

### 4. Qualidade

```bash
poetry run ruff check .        # lint (isort, pydocstyle/google, pep8-naming, annotations)
poetry run ruff format .       # formatação
poetry run pytest              # suíte de testes (58 testes)
poetry run pytest tests/test_models.py::test_mlp_learns_a_separable_signal   # um teste
poetry run pre-commit install  # ativa os hooks de pre-commit
```

## Pipeline DVC

| Stage | Entrada | Saída |
| --- | --- | --- |
| `preprocess` | `data/raw/ratings.csv` (versionado no DVC) | `interactions.parquet` |
| `feature_eng` | `interactions.parquet` | splits `train`/`validation`/`test`, `feature_store.json`, `preprocessor.joblib` |
| `train@mlp` | splits + `feature_store.json` | `models/mlp/model.joblib`, `reports/train/mlp.json` |
| `train@popularity` | idem | `models/popularity/model.joblib`, relatório |
| `train@logistic` | idem | `models/logistic/model.joblib`, relatório |
| `evaluate` | split de teste + os três modelos | `metrics.json`, `comparison.md`, `model_registry.json`, `sample_recommendations.json` |

O dataset bruto é versionado com `dvc add` (`data/raw/ratings.csv.dvc`) e o
remote padrão é o diretório local `dvcstore/` — trocável por S3 alterando
`.dvc/config`.

## Modelo e reprodutibilidade

- **Rede neural:** embeddings de usuário e item (32d) concatenados a quatro
  features comportamentais, torre densa com dropout, perda
  `BCEWithLogitsLoss`, otimizador Adam e **early stopping** (paciência 3) com
  restauração dos melhores pesos.
- **Baselines:** popularidade suavizada e regressão logística do Scikit-Learn
  sobre as mesmas features.
- **Divisão dos dados:** cronológica dentro do histórico de cada usuário, o que
  evita vazamento temporal mantendo todos os usuários no treino.
- **Features:** calculadas somente no split de treino e padronizadas com o
  `StandardScaler` do Scikit-Learn, aplicado aos splits de validação e teste sem
  refit.
- **Seeds:** `RANDOM_SEED` do `.env` é aplicado a `random`, NumPy e PyTorch por
  `set_global_seed`; há teste garantindo que dois treinos produzem as mesmas
  previsões.

## MLflow e Model Registry

Cada `dvc repro` registra quatro runs (um por modelo mais a avaliação) com
parâmetros, curvas de aprendizado por época, métricas de validação e teste, e o
modelo empacotado como `pyfunc` — o que torna a rede PyTorch e os baselines do
Scikit-Learn intercambiáveis no Registry.

No stage `evaluate`, o melhor modelo pela métrica primária (`roc_auc`) é
registrado como nova versão de `ecommerce-recsys-recommender`, passa por
`Staging` e é promovido a `Production`, com as versões anteriores arquivadas e o
alias `champion` reapontado. O resultado fica em `reports/model_registry.json`.

## Etapas do desafio

1. **Clean Code e Estrutura** — estrutura do projeto, SOLID, Factory/Strategy/
   Template Method, type hints, docstrings Google, ruff e pre-commit. ✅
2. **Ambiente e Dependências** — Poetry com deps prod/dev separadas, lock file
   commitado, `.env` via Pydantic Settings, `scripts/validate_env.py`. ✅
3. **Containerização e Versionamento** — Dockerfile multi-stage, compose com
   servidor MLflow, DVC inicializado com dataset versionado e remote, pipeline
   `preprocess → feature_eng → train → evaluate` e tracking no MLflow. ✅
4. **Rede Neural, Registry e Entrega** — MLP em PyTorch com early stopping,
   comparação com baselines em nove métricas, promoção no Model Registry, Model
   Card e documentação final. ✅
