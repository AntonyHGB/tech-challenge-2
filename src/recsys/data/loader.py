"""Loading and cleaning of raw user-item interactions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_COLUMN_MAP: dict[str, str] = {
    "userId": "user_id",
    "movieId": "item_id",
    "rating": "rating",
    "timestamp": "timestamp",
}
INTERACTION_COLUMNS: tuple[str, ...] = ("user_id", "item_id", "rating", "timestamp")


def load_interactions(path: Path) -> pd.DataFrame:
    """Read the raw ratings file and normalise its column names.

    Args:
        path: Location of the raw ``ratings.csv``.

    Returns:
        Frame with the ``user_id``, ``item_id``, ``rating`` and ``timestamp``
        columns.

    Raises:
        ValueError: If the raw file lacks an expected column.
    """
    frame = pd.read_csv(path)
    missing = sorted(set(RAW_COLUMN_MAP) - set(frame.columns))
    if missing:
        raise ValueError(f"Raw dataset is missing columns: {', '.join(missing)}.")
    renamed = frame.rename(columns=RAW_COLUMN_MAP)
    return renamed.loc[:, list(INTERACTION_COLUMNS)]


def clean_interactions(
    frame: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
) -> pd.DataFrame:
    """Drop incomplete rows, duplicates and sparsely observed entities.

    Args:
        frame: Normalised interaction frame.
        min_user_interactions: Minimum interactions required to keep a user.
        min_item_interactions: Minimum interactions required to keep an item.

    Returns:
        Cleaned frame sorted chronologically, with a fresh index.
    """
    cleaned = frame.dropna(subset=list(INTERACTION_COLUMNS))
    cleaned = cleaned.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    cleaned = _filter_by_frequency(cleaned, "item_id", min_item_interactions)
    cleaned = _filter_by_frequency(cleaned, "user_id", min_user_interactions)
    return cleaned.sort_values("timestamp").reset_index(drop=True)


def _filter_by_frequency(
    frame: pd.DataFrame, column: str, minimum: int
) -> pd.DataFrame:
    """Keep only rows whose ``column`` value appears at least ``minimum`` times.

    Args:
        frame: Frame to filter.
        column: Column holding the entity identifier.
        minimum: Minimum number of occurrences required.

    Returns:
        The filtered frame.
    """
    if minimum <= 1:
        return frame
    counts = frame[column].value_counts()
    keep = counts[counts >= minimum].index
    return frame[frame[column].isin(keep)]
