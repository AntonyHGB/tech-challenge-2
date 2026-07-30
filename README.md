# ML Tech Challenge — Fase 2 (Sistema de Recomendação para E-commerce)

---

## 1) Estrutura do projeto

```text
.
├── configs/
│   └── params.yaml            ← Hiperparâmetros do pipeline e modelos (rastreados pelo DVC)
├── data/
│   ├── raw/
│   │   └── ratings.csv.dvc    ← Dataset bruto versionado no DVC (MovieLens ml-latest-small)
│   └── processed/             ← Parquets limpos, splits e feature store persistido
├── docs/
│   └── model_card.md          ← Model Card com performance, limitações e análise de vieses
├── dvcstore/                  ← Storage local padrão do remote DVC
├── mlartifacts/ / mlruns/     ← Banco de dados (SQLite) e artefatos de experimentos do MLflow
├── models/                    ← Artefatos dos modelos treinados (.joblib) por estágio do pipeline
├── reports/
│   ├── metrics.json           ← Relatório consolidador de métricas de teste em JSON
│   ├── comparison.md          ← Tabela comparativa de performance em Markdown
│   └── model_registry.json    ← Status de promoção do modelo vencedor no Model Registry
├── scripts/
│   ├── download_dataset.py    ← Baixa e extrai o dataset bruto em data/raw/
│   └── validate_env.py        ← Valida versão do Python, pacotes e configurações tipadas
├── src/
│   └── recsys/
│       ├── config.py          ← Configuração tipada lida do .env via Pydantic Settings
│       ├── seeding.py         ← Garantia de reprodutibilidade (seed fixado em random/numpy/torch)
│       ├── metrics.py         ← Implementação das métricas de classificação e ranking (@10)
│       ├── data/              ← Módulos de ingestão, limpeza, encoders e split temporal por usuário
│       ├── features/          ← Rótulo implícito, normalização e feature store
│       ├── preprocessing/     ← Padrões Strategy + Template Method (Scalers do Scikit-Learn)
│       ├── models/            ← Padrão Factory, modelo neural MLP (PyTorch) e baselines
│       ├── tracking/          ← Integrador do MLflow, wrapper pyfunc e Model Registry
│       └── pipelines/         ← Implementação dos 6 estágios do DVC (preprocess, feature_eng, train, evaluate)
├── tests/                     ← Suíte de 59 testes automatizados (pytest)
├── .dockerignore
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile                 ← Build multi-stage (builder de dependências + runtime slim)
├── docker-compose.yml         ← Orquestração do servidor MLflow + serviço de treino
├── dvc.yaml / dvc.lock        ← Pipeline de dados e treino reprodutível no DVC
├── pyproject.toml / poetry.lock
└── README.md
```

---

## 2) O que cada arquivo principal faz

- `docs/model_card.md`  
  Documento oficial do modelo. Detalha a arquitetura neural, volume de dados, resultados comparativos, limitações técnicas e análise de vieses comportamentais.

- `configs/params.yaml`  
  Arquivo centralizado de hiperparâmetros (parâmetros de split, feature engineering, arquitetura da MLP e hiperparâmetros dos baselines).

- `scripts/validate_env.py`  
  Script de diagnóstico pré-execução que verifica compatibilidade de ambiente, dependências instaladas e variáveis de ambiente.

- `scripts/download_dataset.py`  
  Script para download e extração do dataset MovieLens `ml-latest-small` em `data/raw/ratings.csv`.

- `src/recsys/config.py`  
  Carrega e valida as configurações de ambiente do arquivo `.env` usando Pydantic Settings.

- `src/recsys/seeding.py`  
  Centraliza o isolamento de aleatoriedade (`set_global_seed`) garantindo determinismo total no PyTorch, NumPy e Python.

- `src/recsys/preprocessing/base.py` & `strategies.py`  
  Aplica os padrões **Strategy** e **Template Method** para encapsular diferentes técnicas de escala de features (MinMax e Standard Scaler) extensíveis sem alterar o pipeline que os consome.

- `src/recsys/models/factory.py`  
  Aplica o padrão **Factory** (`ModelFactory`) para instanciação dinâmica dos modelos (`mlp`, `logistic`, `popularity`) a partir dos arquivos de configuração.

- `src/recsys/models/mlp.py` & `early_stopping.py`  
  Rede Neural (MLP com User/Item Embeddings de 32d) em PyTorch com mecanismo de **Early Stopping** monitorando a perda de validação.

- `src/recsys/tracking/`  
  Encapsula as interações com o MLflow: registro de experimentos, salvamento de modelos no formato `pyfunc` e promoção do modelo "champion" a `Production` no Model Registry.

- `dvc.yaml` & `dvc.lock`  
  Pipeline reprodutível de 6 estágios (`preprocess`, `feature_eng`, `train@mlp`, `train@popularity`, `train@logistic`, `evaluate`).

- `Dockerfile` & `docker-compose.yml`  
  Containerização completa multi-stage (imagem runtime otimizada rodando com usuário não-root `recsys`) e orquestração com servidor MLflow acoplado em banco SQLite.

- `tests/`  
  Suíte de 59 testes automatizados cobrindo configurações, fábricas, pré-processamento, split determinístico e reprodutibilidade do modelo.

---

## 3) Requisitos

- Python 3.12+
- Poetry 2.0+ (ou `python -m poetry`)
- Git
- Docker & Docker Compose (opcional para execução containerizada)

---

## 4) Passo a passo para rodar

### 4.1 Clonar o repositório
```bash
git clone https://github.com/AntonyHGB/tech-challenge-2.git
cd tech-challenge-2
```

### 4.2 Configurar o ambiente virtual e variáveis
```bash
# Instala todas as dependências (produção e desenvolvimento) a partir do lock file
poetry install

# Cria o arquivo de configuração local
cp .env.example .env
```

### 4.3 Validar o ambiente
```bash
poetry run python scripts/validate_env.py
```
*Saída esperada:* Todos os checadores indicando `[OK]` e mensagem `Ambiente OK.`

### 4.4 Baixar o dataset bruto
```bash
poetry run python scripts/download_dataset.py
```
*Saída esperada:* Dataset baixado e extraído em `data/raw/ratings.csv` (pronto para versionamento via DVC).

### 4.5 Executar o pipeline completo (DVC)

> **Importante:** O comando `dvc repro` executa automaticamente os 6 estágios sequenciais do pipeline: pré-processamento, geração de features, treinamento dos 3 modelos (MLP, Logistic Regression e Popularidade) e a avaliação comparativa com promoção do melhor modelo ao MLflow Registry.

```bash
poetry run dvc repro
```

### 4.6 Visualizar experimentos e Model Registry no MLflow
```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Acesse no navegador em: `http://localhost:5000`

### 4.7 Executar em contêineres (Docker Compose)

Caso prefira rodar toda a aplicação sem preparar o ambiente Python localmente:

```bash
# Sobe o servidor MLflow e executa o pipeline de treino em contêineres isolados
docker compose up --build
```
- **MLflow Server & Registry:** `http://localhost:5000`
- Os artefatos de saída (`data/`, `models/`, `reports/`) serão sincronizados com o seu host local através dos volumes do Docker.

### 4.8 Rodar testes automatizados
```bash
poetry run pytest
```
*Saída esperada:* 59 passed in ~10s.

### 4.9 Rodar verificação de qualidade e lint (Ruff)
```bash
# Checagem de código e padrões
poetry run ruff check .

# Instalação dos hooks de pre-commit
poetry run pre-commit install
```

---

## 5) Comandos rápidos

| Comando | Descrição |
|---|---|
| `poetry install` | Instala todas as dependências a partir do lock file commitado |
| `poetry run python scripts/validate_env.py` | Executa diagnóstico do ambiente de desenvolvimento |
| `poetry run python scripts/download_dataset.py` | Baixa o dataset bruto para a pasta `data/raw/` |
| `poetry run dvc repro` | Reproduz todo o pipeline de dados, treino e avaliação (6 estágios) |
| `poetry run pytest` | Executa a suíte de 59 testes automatizados |
| `poetry run ruff check .` | Verifica a conformidade de código e lint com ruff |
| `poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db` | Inicializa a interface gráfica do MLflow na porta 5000 |
| `docker compose up --build` | Executa o treino e o servidor MLflow via Docker |

---

## 6) Resultados obtidos

Abaixo está o comparativo de métricas obtido no split de teste (8.767 interações e 561 usuários avaliados):

| Modelo | ROC AUC | Acurácia | F1-Score | Log Loss | Precision@10 | NDCG@10 | Status no Registry |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **MLP (PyTorch)** | **0,7593** | **0,6948** | **0,6941** | **0,5899** | 0,6552 | 0,8163 | **Promovido a `Production` (`champion`)** |
| **Logistic Regression** | 0,7468 | 0,6859 | 0,6794 | 0,6049 | 0,6575 | 0,8160 | Evaluated |
| **Popularity Baseline** | 0,6762 | 0,6261 | 0,6258 | 0,6452 | 0,6561 | 0,8222 | Evaluated |

---

## 7) Checklist de entrega e boas práticas

1. [x] **Estrutura Clean Code & SOLID:** Módulos curtos, inversão de dependência e funções com no máximo 20 linhas de código.
2. [x] **Design Patterns:** Factory (`ModelFactory`), Strategy (`PreprocessingStrategy`) e Template Method (`SklearnScalerStrategy`).
3. [x] **Dependências e Reprodutibilidade:** `pyproject.toml` com Poetry, lock file commitado e seeds globais fixados em 42.
4. [x] **Pipeline DVC:** 6 estágios estruturados em [dvc.yaml](dvc.yaml) e dados versionados.
5. [x] **Modelo Neural PyTorch:** Embeddings de usuário e item, early stopping e comparação contra baselines do Scikit-Learn em 9 métricas.
6. [x] **MLflow & Model Registry:** Rastreamento de runs e promoção automatizada do melhor modelo para `Production`.
7. [x] **Containerização:** Dockerfile multi-stage com usuário não-root e `docker-compose.yml`.

---

## 8) Dependências principais

**Produção:**
- `torch` (PyTorch)
- `scikit-learn`
- `mlflow`
- `dvc`
- `pandas`, `numpy`, `pyyaml`
- `pydantic`, `pydantic-settings`

**Desenvolvimento e Qualidade:**
- `pytest`
- `ruff`
- `pre-commit`

---

## 9) Documentação Adicional

- **Model Card Completo:** Para a análise detalhada sobre transparência, limites e vieses do modelo, leia [docs/model_card.md](docs/model_card.md).

