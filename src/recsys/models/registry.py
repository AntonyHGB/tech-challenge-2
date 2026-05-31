"""Default model registry (Factory composition root)."""

from __future__ import annotations

from recsys.models.baseline import BaselineRecommender
from recsys.models.factory import ModelFactory
from recsys.models.mlp import MLPRecommender


def build_default_factory() -> ModelFactory:
    """Create a :class:`ModelFactory` pre-loaded with the built-in models.

    Returns:
        A factory able to build the ``"mlp"`` and ``"baseline"`` recommenders.
    """
    factory = ModelFactory()
    factory.register("mlp", MLPRecommender)
    factory.register("baseline", BaselineRecommender)
    return factory
