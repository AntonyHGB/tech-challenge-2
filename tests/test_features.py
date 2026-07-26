"""Tests for the feature engineering layer."""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.data.interactions import FEATURE_COLUMNS, InteractionData
from recsys.features.labels import add_binary_label, positive_rate
from recsys.features.scaling import fit_scale_frame
from recsys.features.statistics import InteractionStatistics
from recsys.features.store import FeatureStore
from recsys.preprocessing.registry import build_pipeline


def test_statistics_count_interactions_per_entity(interaction_frame):
    statistics = InteractionStatistics.from_frame(interaction_frame)
    assert statistics.user_activity[1] == 4
    assert statistics.item_popularity[10] == 3


def test_attach_adds_every_feature_column(interaction_frame):
    statistics = InteractionStatistics.from_frame(interaction_frame)
    enriched = statistics.attach(interaction_frame)
    assert set(FEATURE_COLUMNS).issubset(enriched.columns)


def test_unknown_entities_fall_back_to_neutral_values(interaction_frame):
    statistics = InteractionStatistics.from_frame(interaction_frame)
    row = statistics.feature_row(user_id=999, item_id=999)
    assert row["user_activity"] == 0.0
    assert row["item_mean_rating"] == statistics.global_mean_rating


def test_statistics_survive_a_json_round_trip(interaction_frame):
    statistics = InteractionStatistics.from_frame(interaction_frame)
    restored = InteractionStatistics.from_dict(statistics.to_dict())
    assert restored == statistics


def test_labels_use_the_configured_threshold(interaction_frame):
    labelled = add_binary_label(interaction_frame, positive_threshold=4.0)
    assert labelled["label"].tolist() == [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
    assert positive_rate(labelled) == pytest.approx(0.6)


def test_feature_store_round_trip(tmp_path, interaction_frame):
    store = FeatureStore(
        statistics=InteractionStatistics.from_frame(interaction_frame),
        user_classes=[1, 2, 3],
        item_classes=[10, 11, 12, 13],
        positive_threshold=4.0,
    )
    restored = FeatureStore.load(store.save(tmp_path / "store.json"))
    assert restored.n_users == 3
    assert restored.n_items == 4
    assert restored.item_encoder().transform([12]).tolist() == [2]


def test_scaling_standardises_the_feature_columns(interaction_frame):
    statistics = InteractionStatistics.from_frame(interaction_frame)
    enriched = add_binary_label(statistics.attach(interaction_frame), 4.0)
    pipeline = build_pipeline("standard", FEATURE_COLUMNS)
    scaled = fit_scale_frame(pipeline, enriched)
    assert scaled.shape == enriched.shape
    assert scaled["user_activity"].mean() == pytest.approx(0.0, abs=1e-9)
    assert scaled["item_popularity"].std(ddof=0) == pytest.approx(1.0)


def test_interaction_data_requires_the_feature_columns():
    frame = pd.DataFrame({"user_index": [0], "item_index": [0], "label": [1.0]})
    with pytest.raises(KeyError, match="missing columns"):
        InteractionData.from_frame(frame)
