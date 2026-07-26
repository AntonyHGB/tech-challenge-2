"""Serialisation of trained recommenders to disk."""

from __future__ import annotations

from pathlib import Path

import joblib

from recsys.models.base import RecommenderModel


def save_model(model: RecommenderModel, path: Path) -> Path:
    """Persist a trained recommender.

    Args:
        model: Fitted recommender.
        path: Destination file; parent directories are created.

    Returns:
        The path the model was written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> RecommenderModel:
    """Load a recommender previously written by :func:`save_model`.

    Args:
        path: File holding the serialised model.

    Returns:
        The deserialised recommender.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the file does not hold a recommender.
    """
    if not path.exists():
        raise FileNotFoundError(f"No model artifact at '{path}'.")
    model = joblib.load(path)
    if not isinstance(model, RecommenderModel):
        raise TypeError(f"'{path}' does not contain a RecommenderModel.")
    return model
