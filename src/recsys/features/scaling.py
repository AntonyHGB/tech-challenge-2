"""Bridge between the preprocessing strategies and pandas frames."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from recsys.preprocessing.pipeline import PreprocessingPipeline


def fit_scale_frame(
    pipeline: PreprocessingPipeline, frame: pd.DataFrame
) -> pd.DataFrame:
    """Fit the pipeline on ``frame`` and return the scaled frame.

    Args:
        pipeline: Pipeline configured for the feature columns.
        frame: Training frame holding the feature columns.

    Returns:
        A copy of ``frame`` with the feature columns scaled.
    """
    pipeline.fit(_columns_of(frame, pipeline.columns))
    return scale_frame(pipeline, frame)


def scale_frame(pipeline: PreprocessingPipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply an already fitted pipeline to ``frame``.

    Args:
        pipeline: Fitted pipeline.
        frame: Frame holding the feature columns.

    Returns:
        A copy of ``frame`` with the feature columns scaled.
    """
    scaled = frame.copy()
    for column, values in pipeline.transform(
        _columns_of(frame, pipeline.columns)
    ).items():
        scaled[column] = values
    return scaled


def _columns_of(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, list[float]]:
    """Extract the requested columns as plain Python lists.

    Args:
        frame: Source frame.
        columns: Column names to extract.

    Returns:
        Mapping of column name to its values.

    Raises:
        KeyError: If a column is missing from ``frame``.
    """
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"Frame is missing feature columns: {missing}.")
    return {column: frame[column].astype(float).tolist() for column in columns}
