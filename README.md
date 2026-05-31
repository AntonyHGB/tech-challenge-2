# Sistema de Recomendação para E-commerce

Sistema de recomendação de produtos para e-commerce baseado no comportamento de
navegação dos usuários. O modelo central é uma **rede neural (MLP / embedding-based)**
treinada com **PyTorch**, comparada contra **baselines do Scikit-Learn**, com pipeline
reprodutível versionado em **DVC** e experimentos rastreados no **MLflow**.

Projeto desenvolvido para o **Tech Challenge — Fase 02**.

> **Status:** Etapa 1 (Clean Code e Estrutura) concluída. As Etapas 2 a 4 adicionam
> o gerenciamento de dependências com Poetry, a containerização com Docker, o
> pipeline DVC, o tracking no MLflow e o modelo neural treinado.

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
| Gerenciamento de dependências | Poetry (Etapa 2) |
| Lint e formatação | ruff |
| Testes | pytest |
| Containerização | Docker (Etapa 3) |

## Estrutura do projeto

```
src/recsys/
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
tests/                  # Suíte de testes (pytest)
configs/                # Hiperparâmetros externalizados (model.yaml)
data/   models/         # Artefatos versionados via DVC (mantidos fora do git)
pyproject.toml          # Configuração de ferramentas (ruff, pytest)
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

O gerenciamento formal de dependências com Poetry (e o lock file commitado) chega na
Etapa 2. Por ora, as ferramentas de qualidade são configuradas no `pyproject.toml`.
Com `ruff` e `pytest` disponíveis no ambiente:

```bash
ruff check .      # lint (ruff, isort, pydocstyle/google, pep8-naming, annotations)
ruff format .     # formatação
pytest            # executa a suíte de testes (src/ entra no PYTHONPATH via pyproject)
```

Para rodar um único teste:

```bash
pytest tests/test_factory.py::test_unknown_model_raises_key_error
```

Os hooks de pre-commit (lint + formatação do ruff e verificações comuns) estão em
`.pre-commit-config.yaml`. Ative-os com:

```bash
pre-commit install
```

## Roadmap das etapas

1. **Clean Code e Estrutura** — esqueleto do projeto, SOLID, padrões Factory e
   Strategy, type hints, docstrings, ruff e pre-commit. ✅ Concluída.
2. **Ambiente e Dependências** — `pyproject.toml` com Poetry, lock file commitado,
   `.env` via Pydantic Settings e `scripts/validate_env.py`.
3. **Containerização e Versionamento** — Dockerfile multi-stage, `docker-compose`
   com servidor MLflow, inicialização do DVC e pipeline `preprocess → feature_eng →
   train → evaluate`.
4. **Rede Neural, Registry e Entrega** — treino do modelo PyTorch, comparação com
   baselines, promoção no MLflow Model Registry, Model Card e documentação final.
