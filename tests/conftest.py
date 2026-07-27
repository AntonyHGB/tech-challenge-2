"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.data.interactions import FEATURE_COLUMNS, InteractionData


@pytest.fixture
def interaction_frame() -> pd.DataFrame:
    """Build a small synthetic interaction frame.

    Returns:
        Frame with three users, four items and ten interactions.
    """
    rows = [
        (1, 10, 5.0, 1000),
        (1, 11, 2.0, 1001),
        (1, 12, 4.0, 1002),
        (1, 13, 1.0, 1003),
        (2, 10, 4.5, 1004),
        (2, 11, 3.0, 1005),
        (2, 12, 5.0, 1006),
        (3, 10, 2.0, 1007),
        (3, 12, 4.0, 1008),
        (3, 13, 5.0, 1009),
    ]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


@pytest.fixture
def interaction_data() -> InteractionData:
    """Build a learnable synthetic split for the model tests.

    Returns:
        Interactions whose label correlates with the item index.
    """
    rng = np.random.default_rng(7)
    size = 400
    users = rng.integers(0, 20, size)
    items = rng.integers(0, 10, size)
    labels = (items >= 5).astype(np.float32)
    features = np.column_stack(
        [
            users / 20,
            items / 10,
            labels + rng.normal(0, 0.1, size),
            rng.normal(0, 1, size),
        ]
    ).astype(np.float32)
    assert features.shape[1] == len(FEATURE_COLUMNS)
    return InteractionData(
        user_indices=users.astype(np.int64),
        item_indices=items.astype(np.int64),
        features=features,
        labels=labels,
    )
