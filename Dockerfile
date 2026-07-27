# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Estágio 1 (builder): resolve as dependências a partir do lock file commitado.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.4.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

WORKDIR /app

RUN pip install "poetry==${POETRY_VERSION}"

# Só os manifestos de dependência, para a camada ficar em cache até mudarem.
COPY pyproject.toml poetry.lock README.md ./

RUN poetry install --only main --no-root && rm -rf "${HOME}/.cache"

# ---------------------------------------------------------------------------
# Estágio 2 (runtime): imagem enxuta, apenas com o virtualenv e a aplicação.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:${PATH}" \
    # O DVC usa pygit2; sem o binário git, o GitPython do MLflow só avisaria.
    GIT_PYTHON_REFRESH=quiet

WORKDIR /app

RUN useradd --create-home --uid 1000 recsys \
    && mkdir -p /mlflow /app/data/raw /app/data/processed /app/models /app/reports \
    && chown -R recsys:recsys /mlflow /app

COPY --from=builder --chown=recsys:recsys /app/.venv /app/.venv
COPY --chown=recsys:recsys src ./src
COPY --chown=recsys:recsys configs ./configs
COPY --chown=recsys:recsys scripts ./scripts
COPY --chown=recsys:recsys dvc.yaml .dvcignore ./

USER recsys

CMD ["dvc", "repro"]
