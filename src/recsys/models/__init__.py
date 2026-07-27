"""Modelos de recomendação e a fábrica que os constrói."""

from recsys.models.base import RecommenderModel
from recsys.models.factory import (
    BASELINE_MODELS,
    NEURAL_MODEL,
    ModelFactory,
    build_default_factory,
)

__all__ = [
    "BASELINE_MODELS",
    "NEURAL_MODEL",
    "ModelFactory",
    "RecommenderModel",
    "build_default_factory",
]
