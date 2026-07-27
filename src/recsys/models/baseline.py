"""Scikit-Learn baselines the neural recommender is compared against."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from sklearn.linear_model import LogisticRegression

from recsys.data.interactions import InteractionData
from recsys.models.base import RecommenderModel


class PopularityRecommender(RecommenderModel):
    """Non-personalised baseline that scores items by their historical appeal.

    Each item receives the smoothed share of positive interactions it collected
    during training, so popular items rank first for everyone. It is the sanity
    floor any personalised model must beat.
    """

    name: ClassVar[str] = "popularity"

    def __init__(self, smoothing: float = 10.0) -> None:
        """Configure the baseline.

        Args:
            smoothing: Strength of the pull towards the global positive rate,
                which protects items with very few interactions.
        """
        self.smoothing = smoothing
        self._item_scores: dict[int, float] = {}
        self._prior = 0.5

    def fit(
        self, data: InteractionData, validation: InteractionData | None = None
    ) -> None:
        """Compute the smoothed positive rate of every item.

        Args:
            data: Training interactions.
            validation: Unused; kept for interface compatibility.
        """
        self._prior = float(data.labels.mean()) if len(data) else 0.5
        positives = np.bincount(data.item_indices, weights=data.labels, minlength=1)
        counts = np.bincount(data.item_indices, minlength=1)
        smoothed = (positives + self.smoothing * self._prior) / (
            counts + self.smoothing
        )
        self._item_scores = {
            index: float(score)
            for index, score in enumerate(smoothed)
            if counts[index] > 0
        }

    def predict_proba(self, data: InteractionData) -> np.ndarray:
        """Score interactions by item popularity.

        Args:
            data: Interactions to score.

        Returns:
            Popularity score of each item, falling back to the global rate.
        """
        return np.array(
            [
                self._item_scores.get(int(item), self._prior)
                for item in data.item_indices
            ],
            dtype=np.float64,
        )

    def hyperparameters(self) -> dict[str, Any]:
        """Report the configuration MLflow logs for this run.

        Returns:
            Mapping of hyper-parameter name to value.
        """
        return {"smoothing": self.smoothing}


class LogisticRecommender(RecommenderModel):
    """Scikit-Learn logistic regression over the behavioural features.

    Unlike the neural model it has no notion of user or item identity: it only
    sees the aggregated browsing features, which makes it a fair reference for
    how much the learned embeddings actually add.
    """

    name: ClassVar[str] = "logistic"

    def __init__(
        self,
        penalty_strength: float = 1.0,
        max_iterations: int = 1000,
        seed: int = 42,
    ) -> None:
        """Configure the estimator.

        Args:
            penalty_strength: Inverse regularisation strength (``C``).
            max_iterations: Maximum solver iterations.
            seed: Seed handed to the solver for reproducibility.
        """
        self.penalty_strength = penalty_strength
        self.max_iterations = max_iterations
        self.seed = seed
        self._estimator: LogisticRegression | None = None

    def fit(
        self, data: InteractionData, validation: InteractionData | None = None
    ) -> None:
        """Fit the logistic regression on the behavioural features.

        Args:
            data: Training interactions.
            validation: Unused; kept for interface compatibility.
        """
        estimator = LogisticRegression(
            C=self.penalty_strength,
            max_iter=self.max_iterations,
            random_state=self.seed,
        )
        estimator.fit(data.features, data.labels)
        self._estimator = estimator

    def predict_proba(self, data: InteractionData) -> np.ndarray:
        """Score interactions with the fitted estimator.

        Args:
            data: Interactions to score.

        Returns:
            Probability of the positive class for each row.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self._estimator is None:
            raise RuntimeError("LogisticRecommender must be fitted before scoring.")
        return self._estimator.predict_proba(data.features)[:, 1].astype(np.float64)

    def hyperparameters(self) -> dict[str, Any]:
        """Report the configuration MLflow logs for this run.

        Returns:
            Mapping of hyper-parameter name to value.
        """
        return {
            "penalty_strength": self.penalty_strength,
            "max_iterations": self.max_iterations,
        }
