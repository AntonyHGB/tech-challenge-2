"""Tests for the recommenders and their shared contract."""

from __future__ import annotations

import numpy as np
import pytest

from recsys.metrics.evaluation import evaluate_model
from recsys.models.baseline import LogisticRecommender, PopularityRecommender
from recsys.models.early_stopping import EarlyStopping
from recsys.models.mlp import MLPRecommender
from recsys.models.persistence import load_model, save_model


def _mlp() -> MLPRecommender:
    """Build a small, fast neural recommender for the tests.

    Returns:
        An untrained :class:`MLPRecommender`.
    """
    return MLPRecommender(
        n_users=20,
        n_items=10,
        embedding_dim=8,
        hidden_dim=16,
        epochs=8,
        batch_size=64,
        early_stopping_patience=2,
        seed=42,
    )


@pytest.mark.parametrize(
    "model",
    [PopularityRecommender(), LogisticRecommender(max_iterations=200)],
)
def test_baselines_produce_probabilities(model, interaction_data):
    model.fit(interaction_data)
    scores = model.predict_proba(interaction_data)
    assert scores.shape == (len(interaction_data),)
    assert np.all((scores >= 0) & (scores <= 1))


def test_scoring_before_fitting_raises(interaction_data):
    with pytest.raises(RuntimeError, match="fitted"):
        LogisticRecommender().predict_proba(interaction_data)
    with pytest.raises(RuntimeError, match="fitted"):
        _mlp().predict_proba(interaction_data)


def test_mlp_learns_a_separable_signal(interaction_data):
    model = _mlp()
    model.fit(interaction_data, interaction_data)
    metrics = evaluate_model(model, interaction_data, top_k=5)
    assert metrics["roc_auc"] > 0.8
    assert model.epochs_run >= 1


def test_mlp_training_is_reproducible(interaction_data):
    first, second = _mlp(), _mlp()
    first.fit(interaction_data, interaction_data)
    second.fit(interaction_data, interaction_data)
    np.testing.assert_allclose(
        first.predict_proba(interaction_data),
        second.predict_proba(interaction_data),
    )


def test_mlp_records_learning_curves(interaction_data):
    model = _mlp()
    model.fit(interaction_data, interaction_data)
    history = model.training_history()
    assert len(history["train_loss"]) == model.epochs_run
    assert len(history["validation_loss"]) == model.epochs_run


def test_early_stopping_stops_after_patience():
    stopper = EarlyStopping(patience=2)
    assert stopper.update(1, 1.0, {}) is False
    assert stopper.update(2, 1.5, {}) is False
    assert stopper.update(3, 1.6, {}) is True
    assert stopper.best_epoch == 1
    assert stopper.best_loss == 1.0


def test_disabled_early_stopping_never_stops():
    stopper = EarlyStopping(patience=0)
    stopper.update(1, 1.0, {})
    assert stopper.update(2, 2.0, {}) is False


def test_persistence_round_trip(tmp_path, interaction_data):
    model = PopularityRecommender()
    model.fit(interaction_data)
    path = save_model(model, tmp_path / "model.joblib")
    restored = load_model(path)
    np.testing.assert_allclose(
        model.predict_proba(interaction_data),
        restored.predict_proba(interaction_data),
    )


def test_loading_a_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "absent.joblib")
