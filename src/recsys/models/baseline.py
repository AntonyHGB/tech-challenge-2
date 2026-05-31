"""Scikit-Learn baseline recommender (logic implemented in Stage 4)."""

from __future__ import annotations

from collections.abc import Sequence

from recsys.models.base import RecommenderModel


class BaselineRecommender(RecommenderModel):
    """Reference baseline the PyTorch model is compared against.

    The concrete Scikit-Learn logic (e.g. popularity or nearest-neighbour
    ranking) is added in Stage 4. The interface is fixed here so the comparison
    harness can treat it interchangeably with the neural model.
    """

    def __init__(self, strategy: str = "popularity") -> None:
        """Store the baseline configuration.

        Args:
            strategy: Identifier of the baseline algorithm to use.
        """
        self.strategy = strategy

    def fit(
        self,
        user_ids: Sequence[int],
        item_ids: Sequence[int],
        labels: Sequence[float],
    ) -> None:
        """Fit the baseline (Stage 4).

        Args:
            user_ids: Encoded user identifiers.
            item_ids: Encoded item identifiers.
            labels: Interaction targets.

        Raises:
            NotImplementedError: Always, until Stage 4 implements fitting.
        """
        raise NotImplementedError("Baseline fitting arrives in Stage 4.")

    def predict(self, user_ids: Sequence[int], item_ids: Sequence[int]) -> list[float]:
        """Score user-item pairs (Stage 4).

        Args:
            user_ids: Encoded user identifiers.
            item_ids: Encoded item identifiers.

        Raises:
            NotImplementedError: Always, until Stage 4 implements scoring.
        """
        raise NotImplementedError("Baseline scoring arrives in Stage 4.")

    def recommend(self, user_id: int, top_k: int) -> list[int]:
        """Recommend items for a user (Stage 4).

        Args:
            user_id: Encoded user identifier.
            top_k: Number of items to recommend.

        Raises:
            NotImplementedError: Always, until Stage 4 implements ranking.
        """
        raise NotImplementedError("Baseline ranking arrives in Stage 4.")
