# E-commerce Recommendation System

Behaviour-based product recommendation system for e-commerce (Tech Challenge — Fase 02).
A PyTorch neural recommender (MLP / embedding-based) compared against Scikit-Learn
baselines, with a reproducible DVC pipeline and MLflow experiment tracking.

> **Status:** Stage 1 (Clean Code & Structure) complete. Stages 2–4 add Poetry/lock
> file, Docker, the DVC pipeline, MLflow tracking and the trained model.

## Project structure

```
src/recsys/
  preprocessing/   # Strategy pattern: interchangeable feature transformers
    base.py        # PreprocessingStrategy (abstract strategy)
    strategies.py  # MinMaxScaler, StandardScaler (concrete strategies)
    pipeline.py    # PreprocessingPipeline (strategy context)
  models/
    base.py        # RecommenderModel (shared interface)
    factory.py     # ModelFactory (Factory pattern)
    registry.py    # build_default_factory() composition root
    mlp.py         # MLPRecommender (PyTorch, implemented in Stage 4)
    baseline.py    # BaselineRecommender (Scikit-Learn, implemented in Stage 4)
tests/             # pytest suite
configs/           # externalized hyper-parameters
data/  models/     # DVC-tracked artifacts (kept out of git)
```

## Design patterns

- **Strategy** — `PreprocessingStrategy` lets feature transformers be swapped
  without touching the pipeline that consumes them.
- **Factory** — `ModelFactory` builds recommenders from a string key, so new
  models are added by registration rather than by editing call sites.

## Development

Tooling is configured in `pyproject.toml` (full dependency management with Poetry
lands in Stage 2). With `ruff` and `pytest` available:

```bash
ruff check .     # lint (configured: ruff, isort, pydocstyle/google, pep8-naming, annotations)
ruff format .    # format
pytest           # run the test suite (src/ is on the path via pyproject)
```

Pre-commit hooks (ruff lint + format and common checks) are configured in
`.pre-commit-config.yaml`; enable them with `pre-commit install`.
