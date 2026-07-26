"""Chronological train/validation/test splitting of interaction data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataSplits:
    """The three disjoint interaction frames consumed by the pipeline."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def as_mapping(self) -> dict[str, pd.DataFrame]:
        """Expose the splits keyed by name.

        Returns:
            Mapping of split name to frame.
        """
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def split_by_user_history(
    frame: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
) -> DataSplits:
    """Hold out the most recent interactions of every user.

    A global time cut would push whole users into the holdout, leaving them
    without a trained embedding. Splitting inside each user's own history keeps
    the backtest realistic — the model only ever sees the past — while every
    user remains represented in training.

    Args:
        frame: Interaction frame with ``user_id`` and ``timestamp``.
        validation_fraction: Share of each history used for validation.
        test_fraction: Share of each history used for testing.

    Returns:
        The chronological splits.

    Raises:
        ValueError: If the requested fractions do not leave data for training.
    """
    holdout = validation_fraction + test_fraction
    if not 0 < holdout < 1:
        raise ValueError("validation_fraction + test_fraction must be in (0, 1).")
    ordered = frame.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    position = _relative_position(ordered)
    return DataSplits(
        train=_subset(ordered, position < 1 - holdout),
        validation=_subset(
            ordered, (position >= 1 - holdout) & (position < 1 - test_fraction)
        ),
        test=_subset(ordered, position >= 1 - test_fraction),
    )


def drop_cold_start(splits: DataSplits) -> DataSplits:
    """Remove holdout rows whose user or item is absent from the training split.

    Cold-start entities have no learned embedding, so scoring them would measure
    the fallback rather than the model.

    Args:
        splits: Chronological splits.

    Returns:
        Splits whose validation and test frames only reference known entities.
    """
    users = set(splits.train["user_id"].unique())
    items = set(splits.train["item_id"].unique())
    return DataSplits(
        train=splits.train,
        validation=_keep_known(splits.validation, users, items),
        test=_keep_known(splits.test, users, items),
    )


def _relative_position(ordered: pd.DataFrame) -> pd.Series:
    """Locate each interaction inside its user's history.

    Args:
        ordered: Frame sorted by user and timestamp.

    Returns:
        Series in ``[0, 1)`` where ``0`` is the user's oldest interaction.
    """
    rank = ordered.groupby("user_id").cumcount()
    history_size = ordered.groupby("user_id")["item_id"].transform("size")
    return rank / history_size


def _subset(ordered: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Select the masked rows in chronological order.

    Args:
        ordered: Frame sorted by user and timestamp.
        mask: Boolean mask aligned with ``ordered``.

    Returns:
        The selected rows, sorted by timestamp with a fresh index.
    """
    return ordered[mask].sort_values("timestamp").reset_index(drop=True)


def _keep_known(frame: pd.DataFrame, users: set[int], items: set[int]) -> pd.DataFrame:
    """Filter ``frame`` down to rows with a known user and item.

    Args:
        frame: Holdout frame.
        users: User identifiers seen during training.
        items: Item identifiers seen during training.

    Returns:
        The filtered frame with a fresh index.
    """
    mask = frame["user_id"].isin(users) & frame["item_id"].isin(items)
    return frame[mask].reset_index(drop=True)
