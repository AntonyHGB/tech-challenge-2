# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 (builder): resolve the dependencies from the committed lock file.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.4.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

WORKDIR /app

RUN pip install "poetry==${POETRY_VERSION}"

# Only the dependency manifests, so the layer is cached until they change.
COPY pyproject.toml poetry.lock README.md ./

RUN poetry install --only main --no-root && rm -rf "${HOME}/.cache"

# ---------------------------------------------------------------------------
# Stage 2 (runtime): slim image carrying just the virtualenv and the app.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:${PATH}" \
    # O DVC usa pygit2; sem o binário git, o GitPython do MLflow apenas avisa.
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
