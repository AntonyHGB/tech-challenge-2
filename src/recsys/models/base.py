"""Abstract base class shared by every recommender model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class RecommenderModel(ABC):
    """Common interface every recommender must implement.

    Concrete models (the PyTorch neural network, the Scikit-Learn baseline, ...)
    depend only on this abstraction, which keeps the surrounding pipeline
    decoupled from any specific framework, honouring the Dependency Inversion
    Principle.
    """

    @abstractmethod
    def fit(
        self,
        user_ids: Sequence[int],
        item_ids: Sequence[int],
        labels: Sequence[float],
    ) -> None:
        """Train the model on user-item interactions.

        Args:
            user_ids: Encoded user identifiers.
            item_ids: Encoded item identifiers.
            labels: Interaction targets (e.g. implicit 0/1 or explicit ratings).
        """

    @abstractmethod
    def predict(self, user_ids: Sequence[int], item_ids: Sequence[int]) -> list[float]:
        """Predict interaction scores for user-item pairs.

        Args:
            user_ids: Encoded user identifiers.
            item_ids: Encoded item identifiers.

        Returns:
            Predicted score for each pair, in input order.
        """

    @abstractmethod
    def recommend(self, user_id: int, top_k: int) -> list[int]:
        """Return the top-``k`` recommended item ids for a user.

        Args:
            user_id: Encoded user identifier.
            top_k: Number of items to recommend.

        Returns:
            Item ids ordered from most to least relevant.
        """
