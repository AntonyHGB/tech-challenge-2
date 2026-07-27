"""Visão que os modelos têm de um conjunto de interações."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "user_activity",
    "item_popularity",
    "user_mean_rating",
    "item_mean_rating",
)
LABEL_COLUMN = "label"


@dataclass(frozen=True)
class InteractionData:
    """Arrays consumidos por todo recomendador, sem pandas nem PyTorch.

    Attributes:
        user_indices: Índice codificado do usuário de cada interação.
        item_indices: Índice codificado do item de cada interação.
        features: Features comportamentais, no formato ``(linhas, features)``.
        labels: Alvo binário de relevância de cada interação.
    """

    user_indices: np.ndarray
    item_indices: np.ndarray
    features: np.ndarray
    labels: np.ndarray

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
    ) -> InteractionData:
        """Monta os arrays a partir de um frame já preparado.

        Args:
            frame: Frame com índices codificados, features e rótulo.
            feature_columns: Colunas de features expostas aos modelos.

        Returns:
            O :class:`InteractionData` correspondente.

        Raises:
            KeyError: Se faltar alguma coluna obrigatória em ``frame``.
        """
        required = {"user_index", "item_index", LABEL_COLUMN, *feature_columns}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"Colunas ausentes no frame de features: {missing}.")
        return cls(
            user_indices=frame["user_index"].to_numpy(dtype=np.int64),
            item_indices=frame["item_index"].to_numpy(dtype=np.int64),
            features=frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float32),
            labels=frame[LABEL_COLUMN].to_numpy(dtype=np.float32),
        )

    @property
    def n_features(self) -> int:
        """Quantidade de features comportamentais por interação.

        Returns:
            Número de features.
        """
        return int(self.features.shape[1])

    def __len__(self) -> int:
        """Retorna a quantidade de interações.

        Returns:
            Número de linhas.
        """
        return int(self.user_indices.shape[0])
