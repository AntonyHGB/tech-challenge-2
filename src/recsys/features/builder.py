"""Rótulo implícito e aplicação das estratégias de escala aos frames."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from recsys.data.interactions import LABEL_COLUMN
from recsys.preprocessing.pipeline import PreprocessingPipeline


def add_binary_label(frame: pd.DataFrame, positive_threshold: float) -> pd.DataFrame:
    """Marca como relevante a interação cuja nota atinge o limiar.

    O comportamento de navegação é modelado como feedback implícito: uma nota
    maior ou igual a ``positive_threshold`` conta como sinal positivo; as
    demais, como negativo.

    Args:
        frame: Frame de interações com a coluna ``rating``.
        positive_threshold: Nota mínima considerada uma interação positiva.

    Returns:
        Cópia de ``frame`` com a coluna binária ``label``.
    """
    label = (frame["rating"] >= positive_threshold).astype("float32")
    return frame.assign(**{LABEL_COLUMN: label})


def positive_rate(frame: pd.DataFrame) -> float:
    """Proporção de interações positivas em ``frame``.

    Args:
        frame: Frame de interações já rotulado.

    Returns:
        Fração de linhas relevantes, ou ``0.0`` se o frame estiver vazio.
    """
    if frame.empty:
        return 0.0
    return float(frame[LABEL_COLUMN].mean())


def fit_scale_frame(
    pipeline: PreprocessingPipeline, frame: pd.DataFrame
) -> pd.DataFrame:
    """Ajusta o pipeline em ``frame`` e devolve o frame escalonado.

    Args:
        pipeline: Pipeline configurado para as colunas de features.
        frame: Frame de treino com as colunas de features.

    Returns:
        Cópia de ``frame`` com as features escalonadas.
    """
    pipeline.fit(_columns_of(frame, pipeline.columns))
    return scale_frame(pipeline, frame)


def scale_frame(pipeline: PreprocessingPipeline, frame: pd.DataFrame) -> pd.DataFrame:
    """Aplica um pipeline já ajustado a ``frame``.

    Args:
        pipeline: Pipeline ajustado.
        frame: Frame com as colunas de features.

    Returns:
        Cópia de ``frame`` com as features escalonadas.
    """
    scaled = pipeline.transform(_columns_of(frame, pipeline.columns))
    return frame.assign(**scaled)


def _columns_of(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, list[float]]:
    """Extrai as colunas pedidas como listas simples de Python.

    Raises:
        KeyError: Se faltar alguma coluna em ``frame``.
    """
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"Colunas de features ausentes no frame: {missing}.")
    return {column: frame[column].astype(float).tolist() for column in columns}
