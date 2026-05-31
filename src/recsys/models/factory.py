"""Factory for constructing recommender models by name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from recsys.models.base import RecommenderModel


class ModelFactory:
    """Build :class:`RecommenderModel` instances from a string key.

    Implements the *Factory* design pattern. Models are registered under a name
    and callers construct them without importing the concrete classes, keeping
    creation logic in one place and open for extension (Open/Closed Principle).
    """

    def __init__(self) -> None:
        """Initialize an empty factory."""
        self._builders: dict[str, Callable[..., RecommenderModel]] = {}

    def register(self, name: str, builder: Callable[..., RecommenderModel]) -> None:
        """Register a builder under ``name``.

        Args:
            name: Unique key identifying the model.
            builder: Callable returning a new ``RecommenderModel``.

        Raises:
            ValueError: If ``name`` is already registered.
        """
        if name in self._builders:
            raise ValueError(f"Model '{name}' is already registered.")
        self._builders[name] = builder

    def create(self, name: str, **kwargs: Any) -> RecommenderModel:
        """Instantiate the model registered under ``name``.

        Args:
            name: Key of the model to build.
            **kwargs: Keyword arguments forwarded to the builder.

        Returns:
            A new ``RecommenderModel`` instance.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in self._builders:
            available = ", ".join(self.available()) or "<none>"
            raise KeyError(f"Unknown model '{name}'. Available: {available}.")
        return self._builders[name](**kwargs)

    def available(self) -> list[str]:
        """Return the registered model names in sorted order.

        Returns:
            Sorted list of registered model names.
        """
        return sorted(self._builders)
