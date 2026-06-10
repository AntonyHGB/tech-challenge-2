# Sistema de Recomendação para E-commerce

Sistema de recomendação de produtos para e-commerce baseado no comportamento de
navegação dos usuários. O modelo central é uma **rede neural (MLP / embedding-based)**
treinada com **PyTorch**, comparada contra **baselines do Scikit-Learn**, com pipeline
reprodutível versionado em **DVC** e experimentos rastreados no **MLflow**.

Projeto desenvolvido para o **Tech Challenge — Fase 02**.

> **Status:** Etapas 1 (Clean Code e Estrutura) e 2 (Ambiente e Dependências)
> concluídas. As Etapas 3 e 4 adicionam a containerização com Docker, o pipeline
> DVC, o tracking no MLflow e o modelo neural treinado.

## Contexto do problema

Uma empresa de e-commerce precisa recomendar produtos a partir das interações
usuário–item (visualizações, cliques, compras). O objetivo é treinar um modelo de
recomendação e compará-lo, com pelo menos quatro métricas, contra baselines
clássicos — tudo de forma reprodutível e seguindo boas práticas de engenharia.

## Tecnologias

| Camada | Ferramenta |
| --- | --- |
| Modelo neural | PyTorch |
| Baselines e pré-processamento | Scikit-Learn |
| Tracking de experimentos e Model Registry | MLflow |
| Versionamento de dados e pipeline | DVC |
| Gerenciamento de dependências | Poetry |
| Configuração tipada | Pydantic Settings |
| Lint e formatação | ruff |
| Testes | pytest |
| Containerização | Docker (Etapa 3) |

## Estrutura do projeto

```
src/recsys/
  config.py             # Settings (Pydantic) — configuração tipada lida do .env
  preprocessing/        # Padrão Strategy: transformadores de features intercambiáveis
    base.py             # PreprocessingStrategy (estratégia abstrata)
    strategies.py       # MinMaxScaler, StandardScaler (estratégias concretas)
    pipeline.py         # PreprocessingPipeline (contexto que aplica as estratégias)
  models/
    base.py             # RecommenderModel (interface comum dos modelos)
    factory.py          # ModelFactory (padrão Factory)
    registry.py         # build_default_factory() — ponto de composição
    mlp.py              # MLPRecommender (PyTorch, implementado na Etapa 4)
    baseline.py         # BaselineRecommender (Scikit-Learn, implementado na Etapa 4)
scripts/
  validate_env.py       # Validação de ambiente (versão, pacotes, settings)
tests/                  # Suíte de testes (pytest)
configs/                # Hiperparâmetros externalizados (model.yaml)
data/   models/         # Artefatos versionados via DVC (mantidos fora do git)
pyproject.toml          # Dependências (Poetry) + configuração de ferramentas (ruff, pytest)
poetry.lock             # Lock file commitado — instalação reprodutível
.env.example            # Exemplo de variáveis de ambiente
```

## Padrões de projeto aplicados

- **Strategy** — `PreprocessingStrategy` permite trocar o algoritmo de
  pré-processamento de cada coluna sem alterar o `PreprocessingPipeline` que o
  consome. Novos transformadores entram por subclasse (princípio Aberto/Fechado).
- **Factory** — `ModelFactory` constrói os recomendadores a partir de uma chave de
  texto. Novos modelos são adicionados por registro (`registry.py`), sem ramificar
  a lógica com `if/else`.

A interface comum `RecommenderModel` mantém o restante do sistema desacoplado de
qualquer framework específico (princípio da Inversão de Dependência).

## Convenções de código

- Funções com no máximo 20 linhas, nomes descritivos e SOLID.
- Type hints em todas as funções públicas e docstrings no padrão Google.
- `ruff` configurado e sem erros; hooks de pre-commit definidos.

## Como executar

O projeto usa **Poetry** para um ambiente reprodutível a partir do `poetry.lock`
commitado. Requer **Python 3.12**.

### 1. Instalação do zero

```bash
poetry install                 # cria o virtualenv e instala prod + dev a partir do lock
cp .env.example .env           # configurações locais (o .env é git-ignored)
```

### 2. Validação do ambiente

```bash
poetry run python scripts/validate_env.py   # checa Python, pacotes e settings
```

O script confirma a versão do Python, que cada dependência obrigatória importa e
que as `Settings` (Pydantic) carregam do `.env`. Sai com código diferente de zero
se algo faltar, servindo de portão para uma instalação limpa.

### 3. Qualidade

```bash
poetry run ruff check .        # lint (ruff, isort, pydocstyle/google, pep8-naming, annotations)
poetry run ruff format .       # formatação
poetry run pytest              # executa a suíte de testes
```

Para rodar um único teste:

```bash
poetry run pytest tests/test_factory.py::test_unknown_model_raises_key_error
```

Os hooks de pre-commit (lint + formatação do ruff e verificações comuns) estão em
`.pre-commit-config.yaml`. Ative-os com:

```bash
poetry run pre-commit install
```

### Configuração tipada

As configurações são externalizadas para o `.env` e carregadas por
`recsys.config.Settings` (Pydantic Settings), que valida e tipa cada variável —
incluindo o `RANDOM_SEED` fixo que garante a reprodutibilidade. Use
`get_settings()` para obter a instância única em cache.

## Roadmap das etapas

1. **Clean Code e Estrutura** — esqueleto do projeto, SOLID, padrões Factory e
   Strategy, type hints, docstrings, ruff e pre-commit. ✅ Concluída.
2. **Ambiente e Dependências** — `pyproject.toml` com Poetry, lock file commitado,
   `.env` via Pydantic Settings e `scripts/validate_env.py`. ✅ Concluída.
3. **Containerização e Versionamento** — Dockerfile multi-stage, `docker-compose`
   com servidor MLflow, inicialização do DVC e pipeline `preprocess → feature_eng →
   train → evaluate`.
4. **Rede Neural, Registry e Entrega** — treino do modelo PyTorch, comparação com
   baselines, promoção no MLflow Model Registry, Model Card e documentação final.
