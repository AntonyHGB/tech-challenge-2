"""Single entry point that scores a recommender on an interaction split."""

from __future__ import annotations

from recsys.data.interactions import InteractionData
from recsys.metrics.classification import DECISION_THRESHOLD, classification_metrics
from recsys.metrics.ranking import ranking_metrics
from recsys.models.base import RecommenderModel


def evaluate_model(
    model: RecommenderModel,
    data: InteractionData,
    top_k: int = 10,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    """Score a fitted model with both classification and ranking metrics.

    Args:
        model: Fitted recommender.
        data: Interactions to evaluate on.
        top_k: Cut-off used by the ranking metrics.
        threshold: Probability above which an interaction counts as relevant.

    Returns:
        Mapping of metric name to value.
    """
    scores = model.predict_proba(data)
    metrics = classification_metrics(data.labels, scores, threshold)
    metrics.update(ranking_metrics(data.user_indices, data.labels, scores, top_k))
    metrics["n_interactions"] = float(len(data))
    return metrics
