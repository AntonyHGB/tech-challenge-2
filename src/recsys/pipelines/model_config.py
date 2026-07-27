"""Translation of the parameter file into model constructor arguments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from recsys.features.store import FeatureStore
from recsys.models.baseline import LogisticRecommender, PopularityRecommender
from recsys.models.mlp import MLPRecommender
from recsys.pipelines.params import Params


def _mlp_kwargs(params: Params, store: FeatureStore) -> dict[str, Any]:
    """Build the neural recommender arguments.

    Args:
        params: Validated pipeline parameters.
        store: Fitted feature artefacts providing the vocabulary sizes.

    Returns:
        Keyword arguments for :class:`MLPRecommender`.
    """
    return {
        "n_users": store.n_users,
        "n_items": store.n_items,
        "embedding_dim": params.model.embedding_dim,
        "hidden_dim": params.model.hidden_dim,
        "dropout": params.model.dropout,
        "epochs": params.training.epochs,
        "batch_size": params.training.batch_size,
        "learning_rate": params.training.learning_rate,
        "weight_decay": params.training.weight_decay,
        "early_stopping_patience": params.training.early_stopping_patience,
        "seed": params.seed,
    }


def _popularity_kwargs(params: Params, store: FeatureStore) -> dict[str, Any]:
    """Build the popularity baseline arguments.

    Args:
        params: Validated pipeline parameters.
        store: Unused; kept for a uniform builder signature.

    Returns:
        Keyword arguments for :class:`PopularityRecommender`.
    """
    return {"smoothing": params.baselines.popularity_smoothing}


def _logistic_kwargs(params: Params, store: FeatureStore) -> dict[str, Any]:
    """Build the logistic baseline arguments.

    Args:
        params: Validated pipeline parameters.
        store: Unused; kept for a uniform builder signature.

    Returns:
        Keyword arguments for :class:`LogisticRecommender`.
    """
    return {
        "penalty_strength": params.baselines.logistic_penalty_strength,
        "max_iterations": params.baselines.logistic_max_iterations,
        "seed": params.seed,
    }


KWARGS_BUILDERS: dict[str, Callable[[Params, FeatureStore], dict[str, Any]]] = {
    MLPRecommender.name: _mlp_kwargs,
    PopularityRecommender.name: _popularity_kwargs,
    LogisticRecommender.name: _logistic_kwargs,
}


def model_kwargs(
    model_name: str, params: Params, store: FeatureStore
) -> dict[str, Any]:
    """Resolve the constructor arguments of a registered model.

    Args:
        model_name: Registered model key.
        params: Validated pipeline parameters.
        store: Fitted feature artefacts.

    Returns:
        Keyword arguments to hand to the factory.

    Raises:
        KeyError: If no argument builder is registered for ``model_name``.
    """
    if model_name not in KWARGS_BUILDERS:
        available = ", ".join(sorted(KWARGS_BUILDERS))
        raise KeyError(f"No parameters for model '{model_name}'. Known: {available}.")
    return KWARGS_BUILDERS[model_name](params, store)
