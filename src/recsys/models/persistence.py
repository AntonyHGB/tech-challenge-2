"""Gravação e leitura em disco dos recomendadores treinados."""

from __future__ import annotations

from pathlib import Path

import joblib

from recsys.models.base import RecommenderModel


def save_model(model: RecommenderModel, path: Path) -> Path:
    """Grava um recomendador treinado.

    Args:
        model: Recomendador já treinado.
        path: Arquivo de destino; os diretórios são criados.

    Returns:
        O caminho onde o modelo foi gravado.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path) -> RecommenderModel:
    """Lê um recomendador gravado por :func:`save_model`.

    Args:
        path: Arquivo com o modelo serializado.

    Returns:
        O recomendador desserializado.

    Raises:
        FileNotFoundError: Se ``path`` não existir.
        TypeError: Se o arquivo não contiver um recomendador.
    """
    if not path.exists():
        raise FileNotFoundError(f"Nenhum modelo em '{path}'.")
    model = joblib.load(path)
    if not isinstance(model, RecommenderModel):
        raise TypeError(f"'{path}' não contém um RecommenderModel.")
    return model
