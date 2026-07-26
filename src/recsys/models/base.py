"""Abstract base class shared by every recommender model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from recsys.data.interactions import InteractionData


class RecommenderModel(ABC):
    """Common interface every recommender must implement.

    Concrete models (the PyTorch neural network, the Scikit-Learn baselines, ...)
    depend only on this abstraction, which keeps the pipeline, the evaluation
    harness and the MLflow tracking decoupled from any specific framework,
    honouring the Dependency Inversion Principle.

    Attributes:
        name: Key the model is registered under in the factory.
    """

    name: ClassVar[str] = "recommender"

    @abstractmethod
    def fit(
        self, data: InteractionData, validation: InteractionData | None = None
    ) -> None:
        """Train the model on user-item interactions.

        Args:
            data: Training interactions.
            validation: Optional holdout used for early stopping or monitoring.
        """

    @abstractmethod
    def predict_proba(self, data: InteractionData) -> np.ndarray:
        """Score how likely each interaction is to be relevant.

        Args:
            data: Interactions to score.

        Returns:
            Probabilities in ``[0, 1]``, aligned with the input rows.
        """

    def hyperparameters(self) -> dict[str, Any]:
        """Describe the configuration MLflow should log for this model.

        Returns:
            Mapping of hyper-parameter name to value; empty by default.
        """
        return {}

    def training_history(self) -> dict[str, list[float]]:
        """Per-epoch learning curves, when the model produces them.

        Returns:
            Mapping of curve name to its per-epoch values; empty by default.
        """
        return {}
