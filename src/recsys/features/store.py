"""Persistable bundle with everything the serving path needs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recsys.data.encoders import IdEncoder
from recsys.data.interactions import FEATURE_COLUMNS
from recsys.features.statistics import InteractionStatistics


@dataclass(frozen=True)
class FeatureStore:
    """Fitted feature artefacts shared by training, evaluation and serving.

    Attributes:
        statistics: Behavioural aggregates learned on the training split.
        user_classes: Raw user identifiers in encoded-index order.
        item_classes: Raw item identifiers in encoded-index order.
        feature_columns: Feature columns handed to the models.
        positive_threshold: Rating above which an interaction is relevant.
    """

    statistics: InteractionStatistics
    user_classes: list[int]
    item_classes: list[int]
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    positive_threshold: float = 4.0

    @property
    def n_users(self) -> int:
        """Size of the user vocabulary.

        Returns:
            Number of distinct users seen during training.
        """
        return len(self.user_classes)

    @property
    def n_items(self) -> int:
        """Size of the item vocabulary.

        Returns:
            Number of distinct items seen during training.
        """
        return len(self.item_classes)

    def user_encoder(self) -> IdEncoder:
        """Rebuild the fitted user encoder.

        Returns:
            Encoder mapping raw user ids to indices.
        """
        return IdEncoder.from_classes(self.user_classes)

    def item_encoder(self) -> IdEncoder:
        """Rebuild the fitted item encoder.

        Returns:
            Encoder mapping raw item ids to indices.
        """
        return IdEncoder.from_classes(self.item_classes)

    def save(self, path: Path) -> Path:
        """Write the store as JSON.

        Args:
            path: Destination file; parent directories are created.

        Returns:
            The path written to.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._payload(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> FeatureStore:
        """Read a store previously written by :meth:`save`.

        Args:
            path: File holding the serialised store.

        Returns:
            The restored store.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            statistics=InteractionStatistics.from_dict(payload["statistics"]),
            user_classes=[int(value) for value in payload["user_classes"]],
            item_classes=[int(value) for value in payload["item_classes"]],
            feature_columns=tuple(payload["feature_columns"]),
            positive_threshold=float(payload["positive_threshold"]),
        )

    def _payload(self) -> dict[str, Any]:
        """Build the JSON-compatible representation of the store.

        Returns:
            Mapping ready to be serialised.
        """
        return {
            "statistics": self.statistics.to_dict(),
            "user_classes": self.user_classes,
            "item_classes": self.item_classes,
            "feature_columns": list(self.feature_columns),
            "positive_threshold": self.positive_threshold,
        }
