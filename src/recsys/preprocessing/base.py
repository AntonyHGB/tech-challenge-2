"""Strategy interface shared by every feature preprocessing step."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class PreprocessingStrategy(ABC):
    """Interchangeable transformation applied to a single numeric feature column.

    Implements the *Strategy* design pattern: each concrete subclass encapsulates
    one transformation algorithm behind a common interface. New transformations
    are added by subclassing rather than by editing existing code, honouring the
    Open/Closed Principle.
    """

    @abstractmethod
    def fit(self, values: Sequence[float]) -> PreprocessingStrategy:
        """Learn any parameters required to transform future values.

        Args:
            values: Training values used to estimate transformation parameters.

        Returns:
            The fitted strategy instance, enabling call chaining.
        """

    @abstractmethod
    def transform(self, values: Sequence[float]) -> list[float]:
        """Apply the learned transformation to ``values``.

        Args:
            values: Values to transform.

        Returns:
            Transformed values, in the same order as the input.
        """

    def fit_transform(self, values: Sequence[float]) -> list[float]:
        """Fit the strategy and immediately transform the same values.

        Args:
            values: Values used both to fit and to transform.

        Returns:
            Transformed values.
        """
        return self.fit(values).transform(values)
