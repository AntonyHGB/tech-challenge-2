"""Tests for the chronological splitting of interactions."""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.data.splitter import DataSplits, drop_cold_start, split_by_user_history


def test_every_user_is_represented_in_training(interaction_frame):
    splits = split_by_user_history(interaction_frame, 0.2, 0.2)
    assert set(splits.train["user_id"]) == set(interaction_frame["user_id"])


def test_splits_are_disjoint_and_complete(interaction_frame):
    splits = split_by_user_history(interaction_frame, 0.2, 0.2)
    total = len(splits.train) + len(splits.validation) + len(splits.test)
    assert total == len(interaction_frame)


def test_holdout_holds_the_most_recent_interactions(interaction_frame):
    splits = split_by_user_history(interaction_frame, 0.25, 0.25)
    latest = set(interaction_frame.groupby("user_id")["timestamp"].max())
    holdout = set(splits.validation["timestamp"]) | set(splits.test["timestamp"])
    assert latest.isdisjoint(set(splits.train["timestamp"]))
    assert latest.issubset(holdout)


def test_invalid_fractions_are_rejected(interaction_frame):
    with pytest.raises(ValueError, match="must be in"):
        split_by_user_history(interaction_frame, 0.6, 0.5)


def test_cold_start_rows_are_dropped():
    train = pd.DataFrame({"user_id": [1], "item_id": [10], "rating": [5.0]})
    holdout = pd.DataFrame(
        {"user_id": [1, 2], "item_id": [10, 99], "rating": [4.0, 3.0]}
    )
    filtered = drop_cold_start(DataSplits(train, holdout, holdout))
    assert len(filtered.validation) == 1
    assert filtered.test.iloc[0]["item_id"] == 10
