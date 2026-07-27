"""Agregados de comportamento e o pacote de artefatos de features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from recsys.data.encoders import IdEncoder
from recsys.data.interactions import FEATURE_COLUMNS


class InteractionStatistics(BaseModel):
    """Agregados de navegação usados como features comportamentais.

    São calculados apenas no split de treino e depois aplicados aos splits de
    holdout, o que mantém informação do futuro fora das features. Entidades
    desconhecidas recaem em valores neutros.

    Attributes:
        user_activity: Quantidade de interações por usuário.
        item_popularity: Quantidade de interações por item.
        user_mean_rating: Nota média atribuída por cada usuário.
        item_mean_rating: Nota média recebida por cada item.
        global_mean_rating: Nota média de todo o split de treino.
    """

    model_config = ConfigDict(frozen=True)

    user_activity: dict[int, float]
    item_popularity: dict[int, float]
    user_mean_rating: dict[int, float]
    item_mean_rating: dict[int, float]
    global_mean_rating: float

    @classmethod
    def from_frame(cls, train: pd.DataFrame) -> InteractionStatistics:
        """Calcula os agregados a partir das interações de treino.

        Args:
            train: Split de treino com ``user_id``, ``item_id`` e ``rating``.

        Returns:
            As estatísticas calculadas.
        """
        return cls(
            user_activity=_count(train, "user_id"),
            item_popularity=_count(train, "item_id"),
            user_mean_rating=_mean_rating(train, "user_id"),
            item_mean_rating=_mean_rating(train, "item_id"),
            global_mean_rating=float(train["rating"].mean()),
        )

    def attach(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Acrescenta as colunas de features comportamentais a ``frame``.

        Args:
            frame: Frame de interações a enriquecer.

        Returns:
            Cópia de ``frame`` com as quatro features anexadas.
        """
        mean = self.global_mean_rating
        return frame.assign(
            user_activity=_map(frame, "user_id", self.user_activity, 0.0),
            item_popularity=_map(frame, "item_id", self.item_popularity, 0.0),
            user_mean_rating=_map(frame, "user_id", self.user_mean_rating, mean),
            item_mean_rating=_map(frame, "item_id", self.item_mean_rating, mean),
        )


class FeatureStore(BaseModel):
    """Artefatos de features compartilhados por treino e avaliação.

    Attributes:
        statistics: Agregados aprendidos no split de treino.
        user_classes: Identificadores de usuário na ordem dos índices.
        item_classes: Identificadores de item na ordem dos índices.
        feature_columns: Colunas de features entregues aos modelos.
        positive_threshold: Nota a partir da qual a interação é relevante.
    """

    model_config = ConfigDict(frozen=True)

    statistics: InteractionStatistics
    user_classes: list[int]
    item_classes: list[int]
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS
    positive_threshold: float = 4.0

    @property
    def n_users(self) -> int:
        """Tamanho do vocabulário de usuários."""
        return len(self.user_classes)

    @property
    def n_items(self) -> int:
        """Tamanho do vocabulário de itens."""
        return len(self.item_classes)

    def user_encoder(self) -> IdEncoder:
        """Recria o encoder de usuários."""
        return IdEncoder(self.user_classes)

    def item_encoder(self) -> IdEncoder:
        """Recria o encoder de itens."""
        return IdEncoder(self.item_classes)

    def save(self, path: Path) -> Path:
        """Grava o pacote de artefatos em JSON.

        Args:
            path: Arquivo de destino; os diretórios são criados.

        Returns:
            O caminho onde o arquivo foi escrito.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> FeatureStore:
        """Lê um pacote gravado por :meth:`save`.

        Args:
            path: Arquivo com o pacote serializado.

        Returns:
            O pacote restaurado.
        """
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def _count(frame: pd.DataFrame, column: str) -> dict[int, float]:
    """Conta as interações por identificador."""
    counts = frame.groupby(column).size()
    return {int(key): float(value) for key, value in counts.items()}


def _mean_rating(frame: pd.DataFrame, column: str) -> dict[int, float]:
    """Calcula a nota média por identificador."""
    means = frame.groupby(column)["rating"].mean()
    return {int(key): float(value) for key, value in means.items()}


def _map(
    frame: pd.DataFrame, column: str, values: dict[int, float], default: float
) -> pd.Series:
    """Traduz uma coluna pelo agregado, usando ``default`` no que faltar."""
    return frame[column].map(values).fillna(default).astype(float)
