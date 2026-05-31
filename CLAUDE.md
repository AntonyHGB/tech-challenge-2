# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is in **greenfield / scaffolding phase**. The only source-of-truth artifact present is `Tech Challenge Fase 02.txt`, the assignment brief. There is no code, `pyproject.toml`, lock file, or git history yet. Everything below describes the **mandated target architecture** from that brief — treat it as the spec to build toward, and create the actual files/tooling as work progresses. Verify tooling exists before assuming a command runs.

## What this project is

A product **recommendation system** for e-commerce based on user browsing behavior. The core model is a neural network (MLP or embedding-based) trained with **PyTorch**, compared against **Scikit-Learn** baselines. The full pipeline is containerized (Docker), data is versioned with **DVC**, and experiments are tracked in **MLflow** with a Model Registry promotion flow (Staging → Production).

Dataset: e-commerce user-item interactions with ≥ 10,000 interactions (e.g. Instacart, RetailRocket, or MovieLens).

## Mandated stack & tooling

- **Dependency management**: Poetry (or uv) via `pyproject.toml`, with prod deps (`pytorch`, `scikit-learn`, `mlflow`, `dvc`) separated from dev deps (`pytest`, `ruff`). Lock file must be committed.
- **Config**: externalized to `.env` (+ `.env.example`) and loaded via **Pydantic Settings**. Seeds must be fixed for reproducibility.
- **Linting**: `ruff` must pass with no errors; pre-commit hooks configured.
- **Containerization**: multi-stage `Dockerfile` (builder for deps + slim runtime) and `docker-compose.yml` running a training service plus an MLflow server.

## Intended commands (create the config that backs these)

```bash
# Environment
poetry install                 # install from lock file (clean-install must work from scratch)
poetry run python scripts/validate_env.py   # environment validation script

# Quality
poetry run ruff check .        # lint (must pass clean)
poetry run pytest              # run test suite
poetry run pytest tests/path/to/test_file.py::test_name   # run a single test

# Data + pipeline (DVC, ≥ 3 stages)
dvc repro                      # reproduce full pipeline
dvc init                       # one-time, when introducing DVC

# Containers
docker compose up              # training service + MLflow server
```

## Target architecture

Standard layout the brief requires: `src/`, `tests/`, `data/`, `models/`, `configs/`.

The DVC pipeline (`dvc.yaml`) is the backbone and must have ≥ 3 stages, organized as:

```
preprocess → feature_eng → train → evaluate
```

Each stage is a discrete, reproducible step. MLflow logging (params, metrics, artifacts) happens inside the `train`/`evaluate` stages; ≥ 3 runs must be tracked and the best model promoted to Production in the Model Registry. Model comparison (PyTorch MLP vs. Scikit-Learn baseline) uses ≥ 4 metrics, with early stopping on the neural net.

## Conventions (enforced, not optional)

- **Functions ≤ 20 lines**, short modules, descriptive names, SOLID.
- **Type hints on all public functions** + Google-style docstrings.
- **Design patterns required** — at minimum apply Factory (for model creation) and Strategy (for preprocessors); Template Method is also acceptable. When adding a model or preprocessor, extend via these patterns rather than branching logic.
- **Semantic commit history.**

## Build order (the brief's 4 stages)

1. **Clean code & structure** — project skeleton, naming/SOLID, ≥ 1 design pattern, type hints + docstrings, ruff + pre-commit passing.
2. **Environment & deps** — `pyproject.toml` with Poetry, committed lock file, `.env` + Pydantic Settings, `scripts/validate_env.py`, verified clean install.
3. **Containerization & versioning** — multi-stage Dockerfile, compose with MLflow server, DVC init + versioned dataset + remote, the `dvc.yaml` pipeline, MLflow tracking.
4. **Neural net, registry & delivery** — train PyTorch MLP/embedding model, baseline comparison, MLflow Model Registry promotion, Model Card (performance, limitations, biases), final README.
