"""Concrete preprocessing strategies backed by Scikit-Learn scalers."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler as SklearnMinMax
from sklearn.preprocessing import StandardScaler as SklearnStandard

from recsys.preprocessing.base import PreprocessingStrategy


class SklearnScalerStrategy(PreprocessingStrategy):
    """Adapt a Scikit-Learn scaler to the strategy interface.

    Applies the *Template Method* pattern: this class owns the fit/transform
    skeleton (validation, reshaping, fitted-state guard) and defers the single
    varying step — which scaler to instantiate — to :meth:`_build_scaler`.
    """

    def __init__(self) -> None:
        """Initialize an unfitted strategy."""
        self._scaler: TransformerMixin | None = None

    @abstractmethod
    def _build_scaler(self) -> TransformerMixin:
        """Create the Scikit-Learn scaler this strategy delegates to.

        Returns:
            An unfitted scaler instance.
        """

    def fit(self, values: Sequence[float]) -> SklearnScalerStrategy:
        """Fit the underlying scaler on ``values``.

        Args:
            values: Non-empty training values.

        Returns:
            The fitted strategy.

        Raises:
            ValueError: If ``values`` is empty.
        """
        if len(values) == 0:
            raise ValueError(f"Cannot fit {type(self).__name__} on an empty sequence.")
        scaler = self._build_scaler()
        scaler.fit(_as_column(values))
        self._scaler = scaler
        return self

    def transform(self, values: Sequence[float]) -> list[float]:
        """Scale ``values`` with the fitted scaler.

        Args:
            values: Values to scale.

        Returns:
            Scaled values, in input order.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._scaler is None:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before transform()."
            )
        if len(values) == 0:
            return []
        scaled = self._scaler.transform(_as_column(values))
        return [float(value) for value in np.asarray(scaled).ravel()]


class MinMaxScaler(SklearnScalerStrategy):
    """Scale numeric values into the ``[0, 1]`` range."""

    def _build_scaler(self) -> TransformerMixin:
        """Return a Scikit-Learn min-max scaler.

        Returns:
            The unfitted scaler.
        """
        return SklearnMinMax()


class StandardScaler(SklearnScalerStrategy):
    """Standardize values to zero mean and unit variance."""

    def _build_scaler(self) -> TransformerMixin:
        """Return a Scikit-Learn standard scaler.

        Returns:
            The unfitted scaler.
        """
        return SklearnStandard()


def _as_column(values: Sequence[float]) -> np.ndarray:
    """Reshape a flat sequence into the 2D array Scikit-Learn expects.

    Args:
        values: Flat sequence of numbers.

    Returns:
        Array shaped ``(len(values), 1)``.
    """
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)
