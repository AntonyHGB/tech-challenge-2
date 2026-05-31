"""Tests for the model factory and the default registry."""

from __future__ import annotations

import pytest

from recsys.models.baseline import BaselineRecommender
from recsys.models.factory import ModelFactory
from recsys.models.mlp import MLPRecommender
from recsys.models.registry import build_default_factory


def test_default_factory_lists_registered_models():
    assert build_default_factory().available() == ["baseline", "mlp"]


def test_factory_creates_requested_model():
    factory = build_default_factory()
    assert isinstance(factory.create("mlp"), MLPRecommender)
    assert isinstance(factory.create("baseline"), BaselineRecommender)


def test_factory_forwards_keyword_arguments():
    model = build_default_factory().create("mlp", embedding_dim=16)
    assert model.embedding_dim == 16


def test_unknown_model_raises_key_error():
    with pytest.raises(KeyError, match="Unknown model"):
        build_default_factory().create("does-not-exist")


def test_duplicate_registration_raises_value_error():
    factory = ModelFactory()
    factory.register("dummy", MLPRecommender)
    with pytest.raises(ValueError, match="already registered"):
        factory.register("dummy", MLPRecommender)
