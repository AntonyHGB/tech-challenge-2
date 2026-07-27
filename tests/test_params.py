"""Tests for the typed hyper-parameter file and its use by the pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from recsys.features.statistics import InteractionStatistics
from recsys.features.store import FeatureStore
from recsys.pipelines.model_config import model_kwargs
from recsys.pipelines.params import Params, load_params

PARAMS_FILE = Path("configs/params.yaml")


def _store() -> FeatureStore:
    """Build a minimal feature store for the parameter tests.

    Returns:
        A store with two users and three items.
    """
    statistics = InteractionStatistics({}, {}, {}, {}, 3.5)
    return FeatureStore(
        statistics=statistics, user_classes=[1, 2], item_classes=[10, 11, 12]
    )


def test_project_params_file_is_valid():
    params = load_params(PARAMS_FILE)
    assert params.seed == 42
    assert params.features.scaler in {"standard", "minmax"}
    assert params.evaluation.top_k >= 1


def test_missing_params_file_raises():
    with pytest.raises(FileNotFoundError):
        load_params(Path("configs/does-not-exist.yaml"))


def test_unknown_parameters_are_rejected():
    with pytest.raises(ValidationError):
        Params.model_validate({"unexpected": 1})


def test_out_of_range_fractions_are_rejected():
    with pytest.raises(ValidationError):
        Params.model_validate({"data": {"test_fraction": 1.5}})


def test_flat_view_prefixes_each_section():
    flat = Params().flat()
    assert flat["seed"] == 42
    assert flat["training.epochs"] == 30
    assert flat["evaluation.primary_metric"] == "roc_auc"


def test_model_kwargs_wire_the_vocabulary_into_the_network():
    kwargs = model_kwargs("mlp", Params(), _store())
    assert kwargs["n_users"] == 2
    assert kwargs["n_items"] == 3
    assert kwargs["seed"] == 42


def test_model_kwargs_reject_unknown_models():
    with pytest.raises(KeyError, match="No parameters for model"):
        model_kwargs("unknown", Params(), _store())
