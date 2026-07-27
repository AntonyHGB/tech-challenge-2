"""Ranking metrics evaluated per user over the holdout interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ranking_metrics(
    user_indices: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    """Average per-user ranking quality over the scored interactions.

    Each user's holdout interactions are re-ranked by predicted score and
    compared against the items they actually found relevant. Users without a
    single relevant item are skipped, since every ranking would score zero.

    Args:
        user_indices: Encoded user index of each row.
        labels: Ground-truth relevance of each row.
        scores: Predicted score of each row.
        top_k: Cut-off applied to every user's ranking.

    Returns:
        Mapping with ``precision_at_k``, ``recall_at_k``, ``ndcg_at_k`` and the
        number of users the average was taken over.
    """
    frame = pd.DataFrame(
        {"user": user_indices, "label": labels.astype(float), "score": scores}
    )
    per_user = [
        _user_metrics(group, top_k)
        for _, group in frame.groupby("user", sort=False)
        if group["label"].sum() > 0
    ]
    if not per_user:
        return _empty_metrics()
    summary = pd.DataFrame(per_user).mean().to_dict()
    return {
        "precision_at_k": float(summary["precision"]),
        "recall_at_k": float(summary["recall"]),
        "ndcg_at_k": float(summary["ndcg"]),
        "evaluated_users": float(len(per_user)),
    }


def _user_metrics(group: pd.DataFrame, top_k: int) -> dict[str, float]:
    """Compute the ranking metrics of a single user.

    Args:
        group: Rows belonging to one user, with ``label`` and ``score``.
        top_k: Ranking cut-off.

    Returns:
        Mapping with the user's precision, recall and NDCG.
    """
    relevance = group.sort_values("score", ascending=False)["label"].to_numpy()
    cut = relevance[:top_k]
    relevant_total = float(relevance.sum())
    return {
        "precision": float(cut.sum() / max(len(cut), 1)),
        "recall": float(cut.sum() / relevant_total),
        "ndcg": _ndcg(cut, relevant_total, top_k),
    }


def _ndcg(cut: np.ndarray, relevant_total: float, top_k: int) -> float:
    """Compute the normalised discounted cumulative gain of a ranking.

    Args:
        cut: Binary relevance of the top-k ranked items.
        relevant_total: Number of relevant items available to the user.
        top_k: Ranking cut-off.

    Returns:
        The NDCG in ``[0, 1]``.
    """
    discounts = 1.0 / np.log2(np.arange(2, len(cut) + 2))
    gain = float((cut * discounts).sum())
    ideal_hits = int(min(relevant_total, top_k))
    ideal = float(discounts[:ideal_hits].sum())
    return gain / ideal if ideal > 0 else 0.0


def _empty_metrics() -> dict[str, float]:
    """Return zeroed metrics for the degenerate no-relevant-items case.

    Returns:
        Mapping with every ranking metric set to zero.
    """
    return {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "ndcg_at_k": 0.0,
        "evaluated_users": 0.0,
    }
