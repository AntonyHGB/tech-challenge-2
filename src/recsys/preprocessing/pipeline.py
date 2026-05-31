"""Preprocessing pipeline that composes named strategies (Strategy context)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from recsys.preprocessing.base import PreprocessingStrategy


class PreprocessingPipeline:
    """Apply a :class:`PreprocessingStrategy` to each named feature column.

    Acts as the *context* of the Strategy pattern: it owns a mapping of column
    name to strategy and delegates the transformation to each strategy, staying
    agnostic to the concrete algorithm in use.
    """

    def __init__(self, strategies: Mapping[str, PreprocessingStrategy]) -> None:
        """Initialize the pipeline.

        Args:
            strategies: Mapping of feature-column name to the strategy that
                transforms it.
        """
        self._strategies = dict(strategies)

    def fit_transform(
        self, columns: Mapping[str, Sequence[float]]
    ) -> dict[str, list[float]]:
        """Fit and transform every configured column.

        Args:
            columns: Mapping of column name to the raw values of that column.

        Returns:
            Mapping of column name to transformed values.

        Raises:
            KeyError: If a configured column is missing from ``columns``.
        """
        return {
            name: strategy.fit_transform(columns[name])
            for name, strategy in self._strategies.items()
        }
