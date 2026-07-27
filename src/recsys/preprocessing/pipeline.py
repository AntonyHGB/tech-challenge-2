"""Pipeline que compõe estratégias por coluna (contexto do Strategy)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from recsys.preprocessing.base import PreprocessingStrategy


class PreprocessingPipeline:
    """Aplica uma :class:`PreprocessingStrategy` a cada coluna de feature.

    É o *contexto* do padrão Strategy: guarda o mapa de coluna para estratégia
    e delega a transformação a cada uma delas, sem conhecer o algoritmo usado.
    """

    def __init__(self, strategies: Mapping[str, PreprocessingStrategy]) -> None:
        """Inicializa o pipeline.

        Args:
            strategies: Mapa de nome da coluna para a estratégia que a
                transforma.
        """
        self._strategies = dict(strategies)

    @property
    def columns(self) -> list[str]:
        """Colunas tratadas pelo pipeline, na ordem de inserção."""
        return list(self._strategies)

    def fit(self, columns: Mapping[str, Sequence[float]]) -> PreprocessingPipeline:
        """Ajusta cada estratégia na sua coluna.

        Args:
            columns: Mapa de nome da coluna para os valores de treino.

        Returns:
            O pipeline ajustado.

        Raises:
            KeyError: Se faltar em ``columns`` alguma coluna configurada.
        """
        for name, strategy in self._strategies.items():
            strategy.fit(columns[name])
        return self

    def transform(
        self, columns: Mapping[str, Sequence[float]]
    ) -> dict[str, list[float]]:
        """Transforma cada coluna com a sua estratégia já ajustada.

        Args:
            columns: Mapa de nome da coluna para os valores a transformar.

        Returns:
            Mapa de nome da coluna para os valores transformados.

        Raises:
            KeyError: Se faltar em ``columns`` alguma coluna configurada.
        """
        return {
            name: strategy.transform(columns[name])
            for name, strategy in self._strategies.items()
        }

    def fit_transform(
        self, columns: Mapping[str, Sequence[float]]
    ) -> dict[str, list[float]]:
        """Ajusta e transforma todas as colunas configuradas.

        Args:
            columns: Mapa de nome da coluna para os valores originais.

        Returns:
            Mapa de nome da coluna para os valores transformados.

        Raises:
            KeyError: Se faltar em ``columns`` alguma coluna configurada.
        """
        return {
            name: strategy.fit_transform(columns[name])
            for name, strategy in self._strategies.items()
        }
