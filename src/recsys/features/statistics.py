"""Behavioural aggregates learned from the training split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

BASE_FEATURES: tuple[str, ...] = (
    "user_activity",
    "item_popularity",
    "user_mean_rating",
    "item_mean_rating",
)


@dataclass(frozen=True)
class InteractionStatistics:
    """Browsing aggregates used as behavioural features.

    Computed on the training split only and then applied to the holdout splits,
    which keeps future information out of the features. Unknown entities fall
    back to neutral values.

    Attributes:
        user_activity: Interaction count per user.
        item_popularity: Interaction count per item.
        user_mean_rating: Mean rating given by each user.
        item_mean_rating: Mean rating received by each item.
        global_mean_rating: Mean rating over the whole training split.
    """

    user_activity: dict[int, float]
    item_popularity: dict[int, float]
    user_mean_rating: dict[int, float]
    item_mean_rating: dict[int, float]
    global_mean_rating: float

    @classmethod
    def from_frame(cls, train: pd.DataFrame) -> InteractionStatistics:
        """Compute the aggregates from the training interactions.

        Args:
            train: Training split with ``user_id``, ``item_id`` and ``rating``.

        Returns:
            The fitted statistics.
        """
        return cls(
            user_activity=_count(train, "user_id"),
            item_popularity=_count(train, "item_id"),
            user_mean_rating=_mean_rating(train, "user_id"),
            item_mean_rating=_mean_rating(train, "item_id"),
            global_mean_rating=float(train["rating"].mean()),
        )

    def attach(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add the behavioural feature columns to ``frame``.

        Args:
            frame: Interaction frame to enrich.

        Returns:
            A copy of ``frame`` with the four behavioural features appended.
        """
        enriched = frame.copy()
        enriched["user_activity"] = self._map(frame, "user_id", self.user_activity, 0.0)
        enriched["item_popularity"] = self._map(
            frame, "item_id", self.item_popularity, 0.0
        )
        enriched["user_mean_rating"] = self._map(
            frame, "user_id", self.user_mean_rating, self.global_mean_rating
        )
        enriched["item_mean_rating"] = self._map(
            frame, "item_id", self.item_mean_rating, self.global_mean_rating
        )
        return enriched

    def feature_row(self, user_id: int, item_id: int) -> dict[str, float]:
        """Build the feature values of a single candidate pair.

        Args:
            user_id: Raw user identifier.
            item_id: Raw item identifier.

        Returns:
            Mapping of feature name to value, using fallbacks when unseen.
        """
        return {
            "user_activity": self.user_activity.get(user_id, 0.0),
            "item_popularity": self.item_popularity.get(item_id, 0.0),
            "user_mean_rating": self.user_mean_rating.get(
                user_id, self.global_mean_rating
            ),
            "item_mean_rating": self.item_mean_rating.get(
                item_id, self.global_mean_rating
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the statistics to JSON-compatible primitives.

        Returns:
            Mapping ready to be written as JSON.
        """
        return {
            "user_activity": _stringify(self.user_activity),
            "item_popularity": _stringify(self.item_popularity),
            "user_mean_rating": _stringify(self.user_mean_rating),
            "item_mean_rating": _stringify(self.item_mean_rating),
            "global_mean_rating": self.global_mean_rating,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InteractionStatistics:
        """Rebuild statistics previously produced by :meth:`to_dict`.

        Args:
            payload: Mapping loaded from JSON.

        Returns:
            The restored statistics.
        """
        return cls(
            user_activity=_intify(payload["user_activity"]),
            item_popularity=_intify(payload["item_popularity"]),
            user_mean_rating=_intify(payload["user_mean_rating"]),
            item_mean_rating=_intify(payload["item_mean_rating"]),
            global_mean_rating=float(payload["global_mean_rating"]),
        )

    @staticmethod
    def _map(
        frame: pd.DataFrame, column: str, values: dict[int, float], default: float
    ) -> pd.Series:
        """Map a frame column through ``values`` with a default.

        Args:
            frame: Source frame.
            column: Column holding the identifiers.
            values: Aggregate keyed by identifier.
            default: Value used for identifiers absent from ``values``.

        Returns:
            Series of feature values aligned with ``frame``.
        """
        return frame[column].map(values).fillna(default).astype(float)


def _count(frame: pd.DataFrame, column: str) -> dict[int, float]:
    """Count interactions per identifier.

    Args:
        frame: Training interactions.
        column: Identifier column to group by.

    Returns:
        Mapping of identifier to interaction count.
    """
    counts = frame.groupby(column).size()
    return {int(key): float(value) for key, value in counts.items()}


def _mean_rating(frame: pd.DataFrame, column: str) -> dict[int, float]:
    """Average the rating per identifier.

    Args:
        frame: Training interactions.
        column: Identifier column to group by.

    Returns:
        Mapping of identifier to mean rating.
    """
    means = frame.groupby(column)["rating"].mean()
    return {int(key): float(value) for key, value in means.items()}


def _stringify(values: dict[int, float]) -> dict[str, float]:
    """Convert integer keys to strings for JSON serialisation.

    Args:
        values: Aggregate keyed by identifier.

    Returns:
        The same mapping with string keys.
    """
    return {str(key): value for key, value in values.items()}


def _intify(values: dict[str, float]) -> dict[int, float]:
    """Convert string keys back to integers after JSON loading.

    Args:
        values: Aggregate keyed by stringified identifier.

    Returns:
        The same mapping with integer keys.
    """
    return {int(key): float(value) for key, value in values.items()}
