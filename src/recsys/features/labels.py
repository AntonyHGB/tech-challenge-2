"""Conversion of explicit ratings into the implicit relevance target."""

from __future__ import annotations

import pandas as pd

LABEL_COLUMN = "label"


def add_binary_label(frame: pd.DataFrame, positive_threshold: float) -> pd.DataFrame:
    """Label an interaction as relevant when its rating clears the threshold.

    Browsing behaviour is modelled as implicit feedback: an interaction the user
    rated at or above ``positive_threshold`` counts as a positive signal, the
    remaining ones as negatives.

    Args:
        frame: Interaction frame containing a ``rating`` column.
        positive_threshold: Minimum rating considered a positive interaction.

    Returns:
        A copy of ``frame`` with the binary ``label`` column appended.
    """
    labelled = frame.copy()
    labelled[LABEL_COLUMN] = (labelled["rating"] >= positive_threshold).astype(
        "float32"
    )
    return labelled


def positive_rate(frame: pd.DataFrame) -> float:
    """Share of positive interactions in ``frame``.

    Args:
        frame: Labelled interaction frame.

    Returns:
        Fraction of rows labelled as relevant, or ``0.0`` when empty.
    """
    if frame.empty:
        return 0.0
    return float(frame[LABEL_COLUMN].mean())
