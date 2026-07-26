"""Tests for the classification and ranking metrics."""

from __future__ import annotations

import numpy as np
import pytest

from recsys.metrics.classification import classification_metrics
from recsys.metrics.ranking import ranking_metrics


def test_perfect_predictions_score_one():
    labels = np.array([0.0, 1.0, 1.0, 0.0])
    scores = np.array([0.01, 0.99, 0.98, 0.02])
    metrics = classification_metrics(labels, scores)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["log_loss"] < 0.05


def test_single_class_falls_back_to_neutral_auc():
    metrics = classification_metrics(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]))
    assert metrics["roc_auc"] == 0.5
    assert metrics["precision"] == 0.0


def test_metric_names_cover_the_four_required_families():
    metrics = classification_metrics(np.array([0.0, 1.0]), np.array([0.2, 0.8]))
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"} == set(
        metrics
    )


def test_ranking_metrics_reward_the_correct_order():
    users = np.array([1, 1, 1, 1])
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    metrics = ranking_metrics(users, labels, scores, top_k=2)
    assert metrics["precision_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["ndcg_at_k"] == pytest.approx(1.0)
    assert metrics["evaluated_users"] == 1.0


def test_ranking_metrics_penalise_the_inverted_order():
    users = np.array([1, 1, 1, 1])
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = ranking_metrics(users, labels, scores, top_k=2)
    assert metrics["precision_at_k"] == 0.0
    assert metrics["ndcg_at_k"] == 0.0


def test_users_without_relevant_items_are_skipped():
    metrics = ranking_metrics(
        np.array([1, 1]), np.zeros(2), np.array([0.9, 0.1]), top_k=2
    )
    assert metrics["evaluated_users"] == 0.0
