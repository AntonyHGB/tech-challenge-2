"""Top-k recommendation on top of any :class:`RecommenderModel`."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import pandas as pd

from recsys.data.interactions import InteractionData
from recsys.features.scaling import scale_frame
from recsys.features.store import FeatureStore
from recsys.models.base import RecommenderModel
from recsys.preprocessing.pipeline import PreprocessingPipeline


@dataclass(frozen=True)
class Recommendation:
    """A single ranked suggestion.

    Attributes:
        item_id: Raw identifier of the recommended item.
        score: Predicted relevance probability.
    """

    item_id: int
    score: float


class TopKRanker:
    """Rank the catalogue for a user by delegating the scoring to a model.

    Ranking is deliberately kept out of :class:`RecommenderModel`: the models
    only score interactions, while candidate generation, feature assembly and
    ordering live here (Single Responsibility Principle).
    """

    def __init__(
        self,
        model: RecommenderModel,
        store: FeatureStore,
        pipeline: PreprocessingPipeline,
    ) -> None:
        """Wire the ranker to a fitted model and its feature artefacts.

        Args:
            model: Fitted recommender used to score candidates.
            store: Fitted feature artefacts (statistics and vocabularies).
            pipeline: Fitted preprocessing pipeline for the feature columns.
        """
        self._model = model
        self._store = store
        self._pipeline = pipeline
        self._user_encoder = store.user_encoder()
        self._item_encoder = store.item_encoder()

    def recommend(
        self, user_id: int, top_k: int = 10, exclude: Collection[int] = ()
    ) -> list[Recommendation]:
        """Return the highest scoring items for a user.

        Args:
            user_id: Raw user identifier, which must be known to the encoder.
            top_k: Number of items to return.
            exclude: Items to skip, typically the ones already interacted with.

        Returns:
            Recommendations ordered from most to least relevant.

        Raises:
            KeyError: If ``user_id`` was not seen during training.
        """
        if not bool(self._user_encoder.known([user_id])[0]):
            raise KeyError(f"Unknown user '{user_id}'.")
        candidates = self._candidate_frame(user_id, exclude)
        if candidates.empty:
            return []
        scores = self._model.predict_proba(
            InteractionData.from_frame(candidates, self._store.feature_columns)
        )
        ranked = candidates.assign(score=scores).nlargest(top_k, "score")
        return [
            Recommendation(item_id=int(row.item_id), score=float(row.score))
            for row in ranked.itertuples()
        ]

    def _candidate_frame(self, user_id: int, exclude: Collection[int]) -> pd.DataFrame:
        """Assemble the scored feature frame for every candidate item.

        Args:
            user_id: Raw user identifier.
            exclude: Items to leave out of the candidate set.

        Returns:
            Frame with encoded indices, scaled features and a dummy label.
        """
        blocked = set(exclude)
        items = [item for item in self._store.item_classes if item not in blocked]
        frame = pd.DataFrame({"user_id": user_id, "item_id": items})
        frame = self._store.statistics.attach(frame)
        frame["label"] = 0.0
        frame["user_index"] = self._user_encoder.transform(frame["user_id"])
        frame["item_index"] = self._item_encoder.transform(frame["item_id"])
        return scale_frame(self._pipeline, frame)
