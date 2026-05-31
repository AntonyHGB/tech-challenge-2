"""PyTorch MLP recommender (network implemented in Stage 4)."""

from __future__ import annotations

from collections.abc import Sequence

from recsys.models.base import RecommenderModel


class MLPRecommender(RecommenderModel):
    """Embedding-based multilayer perceptron recommender.

    The PyTorch network is built and trained in Stage 4. This class fixes the
    public interface and hyper-parameters now, so the pipeline and the
    :class:`~recsys.models.factory.ModelFactory` can be wired up beforehand.
    """

    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 64) -> None:
        """Store architecture hyper-parameters.

        Args:
            embedding_dim: Size of the user/item embedding vectors.
            hidden_dim: Width of the hidden layer.
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

    def fit(
        self,
        user_ids: Sequence[int],
        item_ids: Sequence[int],
        labels: Sequence[float],
    ) -> None:
        """Train the network (Stage 4).

        Args:
            user_ids: Encoded user identifiers.
            item_ids: Encoded item identifiers.
            labels: Interaction targets.

        Raises:
            NotImplementedError: Always, until Stage 4 implements training.
        """
        raise NotImplementedError("PyTorch MLP training arrives in Stage 4.")

    def predict(self, user_ids: Sequence[int], item_ids: Sequence[int]) -> list[float]:
        """Score user-item pairs (Stage 4).

        Args:
            user_ids: Encoded user identifiers.
            item_ids: Encoded item identifiers.

        Raises:
            NotImplementedError: Always, until Stage 4 implements scoring.
        """
        raise NotImplementedError("PyTorch MLP scoring arrives in Stage 4.")

    def recommend(self, user_id: int, top_k: int) -> list[int]:
        """Recommend items for a user (Stage 4).

        Args:
            user_id: Encoded user identifier.
            top_k: Number of items to recommend.

        Raises:
            NotImplementedError: Always, until Stage 4 implements ranking.
        """
        raise NotImplementedError("PyTorch MLP ranking arrives in Stage 4.")
