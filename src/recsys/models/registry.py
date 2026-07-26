"""Default model registry (Factory composition root)."""

from __future__ import annotations

from recsys.models.baseline import LogisticRecommender, PopularityRecommender
from recsys.models.factory import ModelFactory
from recsys.models.mlp import MLPRecommender

NEURAL_MODEL = MLPRecommender.name
BASELINE_MODELS: tuple[str, ...] = (
    PopularityRecommender.name,
    LogisticRecommender.name,
)


def build_default_factory() -> ModelFactory:
    """Create a :class:`ModelFactory` pre-loaded with the built-in models.

    Returns:
        A factory able to build the neural recommender and both baselines.
    """
    factory = ModelFactory()
    factory.register(MLPRecommender.name, MLPRecommender)
    factory.register(PopularityRecommender.name, PopularityRecommender)
    factory.register(LogisticRecommender.name, LogisticRecommender)
    return factory
