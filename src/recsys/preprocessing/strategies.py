"""Concrete preprocessing strategies (Strategy pattern)."""

from __future__ import annotations

from collections.abc import Sequence

from recsys.preprocessing.base import PreprocessingStrategy


class MinMaxScaler(PreprocessingStrategy):
    """Scale numeric values into the ``[0, 1]`` range."""

    def __init__(self) -> None:
        """Initialize an unfitted scaler."""
        self._minimum: float | None = None
        self._maximum: float | None = None

    def fit(self, values: Sequence[float]) -> MinMaxScaler:
        """Record the observed minimum and maximum.

        Args:
            values: Non-empty training values.

        Returns:
            The fitted scaler.

        Raises:
            ValueError: If ``values`` is empty.
        """
        if not values:
            raise ValueError("Cannot fit MinMaxScaler on an empty sequence.")
        self._minimum = min(values)
        self._maximum = max(values)
        return self

    def transform(self, values: Sequence[float]) -> list[float]:
        """Scale ``values`` using the fitted range.

        Args:
            values: Values to scale.

        Returns:
            Scaled values; a constant column maps to all zeros.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._minimum is None or self._maximum is None:
            raise RuntimeError("MinMaxScaler must be fitted before transform().")
        span = self._maximum - self._minimum
        if span == 0:
            return [0.0 for _ in values]
        return [(value - self._minimum) / span for value in values]


class StandardScaler(PreprocessingStrategy):
    """Standardize values to zero mean and unit variance."""

    def __init__(self) -> None:
        """Initialize an unfitted scaler."""
        self._mean: float | None = None
        self._std: float | None = None

    def fit(self, values: Sequence[float]) -> StandardScaler:
        """Estimate the mean and population standard deviation.

        Args:
            values: Non-empty training values.

        Returns:
            The fitted scaler.

        Raises:
            ValueError: If ``values`` is empty.
        """
        if not values:
            raise ValueError("Cannot fit StandardScaler on an empty sequence.")
        self._mean = sum(values) / len(values)
        self._std = self._population_std(values, self._mean)
        return self

    def transform(self, values: Sequence[float]) -> list[float]:
        """Standardize ``values`` using the fitted statistics.

        Args:
            values: Values to standardize.

        Returns:
            Standardized values; a constant column maps to all zeros.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("StandardScaler must be fitted before transform().")
        if self._std == 0:
            return [0.0 for _ in values]
        return [(value - self._mean) / self._std for value in values]

    @staticmethod
    def _population_std(values: Sequence[float], mean: float) -> float:
        """Compute the population standard deviation around ``mean``.

        Args:
            values: Values to summarize.
            mean: Pre-computed mean of ``values``.

        Returns:
            The population standard deviation.
        """
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return variance**0.5
