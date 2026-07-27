"""Encoding of raw identifiers into the contiguous indices models expect."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.preprocessing import LabelEncoder


class IdEncoder:
    """Map raw user/item identifiers to contiguous zero-based indices.

    Wraps :class:`sklearn.preprocessing.LabelEncoder` behind a small interface
    that also answers which identifiers were seen during ``fit``, which is what
    the splitter needs to drop cold-start rows.
    """

    def __init__(self) -> None:
        """Initialize an unfitted encoder."""
        self._encoder = LabelEncoder()
        self._fitted = False

    def fit(self, values: Sequence[int]) -> IdEncoder:
        """Learn the identifier vocabulary.

        Args:
            values: Identifiers observed in the training data.

        Returns:
            The fitted encoder.

        Raises:
            ValueError: If ``values`` is empty.
        """
        if len(values) == 0:
            raise ValueError("Cannot fit IdEncoder on an empty sequence.")
        self._encoder.fit(np.asarray(values))
        self._fitted = True
        return self

    def transform(self, values: Sequence[int]) -> np.ndarray:
        """Convert identifiers into indices.

        Args:
            values: Identifiers to encode; all must be known.

        Returns:
            Array of zero-based indices.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._require_fitted()
        return self._encoder.transform(np.asarray(values)).astype(np.int64)

    def known(self, values: Sequence[int]) -> np.ndarray:
        """Flag which identifiers belong to the learned vocabulary.

        Args:
            values: Identifiers to test.

        Returns:
            Boolean mask aligned with ``values``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._require_fitted()
        return np.isin(np.asarray(values), self._encoder.classes_)

    @property
    def size(self) -> int:
        """Number of distinct identifiers in the vocabulary.

        Returns:
            Vocabulary size.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._require_fitted()
        return len(self._encoder.classes_)

    def classes(self) -> list[int]:
        """Return the vocabulary in index order.

        Returns:
            Raw identifiers ordered by their encoded index.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._require_fitted()
        return [int(value) for value in self._encoder.classes_]

    @classmethod
    def from_classes(cls, classes: Sequence[int]) -> IdEncoder:
        """Rebuild an encoder from a persisted vocabulary.

        Args:
            classes: Raw identifiers in index order.

        Returns:
            A fitted encoder equivalent to the one that produced ``classes``.
        """
        return cls().fit(list(classes))

    def _require_fitted(self) -> None:
        """Guard methods that need a learned vocabulary.

        Raises:
            RuntimeError: If the encoder has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("IdEncoder must be fitted before use.")
