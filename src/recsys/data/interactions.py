"""Model-facing view of an interaction split."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "user_activity",
    "item_popularity",
    "user_mean_rating",
    "item_mean_rating",
)
LABEL_COLUMN = "label"


@dataclass(frozen=True)
class InteractionData:
    """Arrays every recommender consumes, decoupled from pandas and PyTorch.

    Attributes:
        user_indices: Encoded user index of each interaction.
        item_indices: Encoded item index of each interaction.
        features: Scaled behavioural features, shaped ``(n_rows, n_features)``.
        labels: Binary relevance target of each interaction.
    """

    user_indices: np.ndarray
    item_indices: np.ndarray
    features: np.ndarray
    labels: np.ndarray

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
    ) -> InteractionData:
        """Build the arrays from an engineered feature frame.

        Args:
            frame: Frame holding encoded indices, features and the label.
            feature_columns: Feature columns to expose to the models.

        Returns:
            The corresponding :class:`InteractionData`.

        Raises:
            KeyError: If a required column is missing from ``frame``.
        """
        required = {"user_index", "item_index", LABEL_COLUMN, *feature_columns}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"Feature frame is missing columns: {missing}.")
        return cls(
            user_indices=frame["user_index"].to_numpy(dtype=np.int64),
            item_indices=frame["item_index"].to_numpy(dtype=np.int64),
            features=frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float32),
            labels=frame[LABEL_COLUMN].to_numpy(dtype=np.float32),
        )

    @property
    def n_features(self) -> int:
        """Number of behavioural features per interaction.

        Returns:
            Feature count.
        """
        return int(self.features.shape[1])

    def __len__(self) -> int:
        """Return the number of interactions.

        Returns:
            Row count.
        """
        return int(self.user_indices.shape[0])
