"""Registry that builds preprocessing strategies and pipelines by name."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from recsys.preprocessing.base import PreprocessingStrategy
from recsys.preprocessing.pipeline import PreprocessingPipeline
from recsys.preprocessing.strategies import MinMaxScaler, StandardScaler

STRATEGY_BUILDERS: dict[str, Callable[[], PreprocessingStrategy]] = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
}


def available_strategies() -> list[str]:
    """List the strategy names available to the configuration.

    Returns:
        Sorted strategy names.
    """
    return sorted(STRATEGY_BUILDERS)


def build_strategy(name: str) -> PreprocessingStrategy:
    """Instantiate the strategy registered under ``name``.

    Args:
        name: Strategy key, as written in the params file.

    Returns:
        A fresh, unfitted strategy.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in STRATEGY_BUILDERS:
        available = ", ".join(available_strategies())
        raise KeyError(f"Unknown strategy '{name}'. Available: {available}.")
    return STRATEGY_BUILDERS[name]()


def build_pipeline(name: str, columns: Sequence[str]) -> PreprocessingPipeline:
    """Build a pipeline applying the same strategy to every column.

    Args:
        name: Strategy key applied to each column.
        columns: Feature columns to scale.

    Returns:
        A pipeline with one strategy instance per column.
    """
    return PreprocessingPipeline({column: build_strategy(name) for column in columns})
