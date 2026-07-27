"""Divisão cronológica das interações em treino, validação e teste."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataSplits:
    """Os três conjuntos disjuntos consumidos pelo pipeline."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def as_mapping(self) -> dict[str, pd.DataFrame]:
        """Expõe os splits indexados pelo nome.

        Returns:
            Mapa de nome do split para o frame correspondente.
        """
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def split_by_user_history(
    frame: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
) -> DataSplits:
    """Separa as interações mais recentes de cada usuário.

    Um corte temporal global jogaria usuários inteiros para o holdout, deixando
    esses usuários sem embedding treinado. Dividir dentro do histórico de cada
    usuário mantém o backtest realista — o modelo só enxerga o passado — sem
    perder nenhum usuário no treino.

    Args:
        frame: Interações com as colunas ``user_id`` e ``timestamp``.
        validation_fraction: Fração de cada histórico usada na validação.
        test_fraction: Fração de cada histórico usada no teste.

    Returns:
        Os splits cronológicos.

    Raises:
        ValueError: Se as frações não deixarem dados para o treino.
    """
    holdout = validation_fraction + test_fraction
    if not 0 < holdout < 1:
        raise ValueError("validation_fraction + test_fraction deve estar em (0, 1).")
    ordered = frame.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    position = _relative_position(ordered)
    return DataSplits(
        train=_subset(ordered, position < 1 - holdout),
        validation=_subset(
            ordered, (position >= 1 - holdout) & (position < 1 - test_fraction)
        ),
        test=_subset(ordered, position >= 1 - test_fraction),
    )


def drop_cold_start(splits: DataSplits) -> DataSplits:
    """Remove do holdout as linhas cujo usuário ou item não está no treino.

    Entidades em cold start não têm embedding aprendido: avaliá-las mediria o
    comportamento de fallback, não o do modelo.

    Args:
        splits: Splits cronológicos.

    Returns:
        Splits cujos conjuntos de validação e teste só citam entidades vistas.
    """
    users = set(splits.train["user_id"].unique())
    items = set(splits.train["item_id"].unique())
    return DataSplits(
        train=splits.train,
        validation=_keep_known(splits.validation, users, items),
        test=_keep_known(splits.test, users, items),
    )


def _relative_position(ordered: pd.DataFrame) -> pd.Series:
    """Localiza cada interação dentro do histórico do seu usuário.

    Args:
        ordered: Frame ordenado por usuário e timestamp.

    Returns:
        Série em ``[0, 1)``, onde ``0`` é a interação mais antiga do usuário.
    """
    rank = ordered.groupby("user_id").cumcount()
    history_size = ordered.groupby("user_id")["item_id"].transform("size")
    return rank / history_size


def _subset(ordered: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Seleciona as linhas marcadas, em ordem cronológica.

    Args:
        ordered: Frame ordenado por usuário e timestamp.
        mask: Máscara booleana alinhada a ``ordered``.

    Returns:
        As linhas selecionadas, ordenadas por timestamp e com índice novo.
    """
    return ordered[mask].sort_values("timestamp").reset_index(drop=True)


def _keep_known(frame: pd.DataFrame, users: set[int], items: set[int]) -> pd.DataFrame:
    """Filtra ``frame`` mantendo só linhas com usuário e item conhecidos.

    Args:
        frame: Frame de holdout.
        users: Identificadores de usuário vistos no treino.
        items: Identificadores de item vistos no treino.

    Returns:
        O frame filtrado, com índice novo.
    """
    mask = frame["user_id"].isin(users) & frame["item_id"].isin(items)
    return frame[mask].reset_index(drop=True)
