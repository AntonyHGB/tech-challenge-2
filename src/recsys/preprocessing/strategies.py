"""Estratégias de pré-processamento apoiadas em scalers do Scikit-Learn."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler as SklearnMinMax
from sklearn.preprocessing import StandardScaler as SklearnStandard

from recsys.preprocessing.base import PreprocessingStrategy
from recsys.preprocessing.pipeline import PreprocessingPipeline


class SklearnScalerStrategy(PreprocessingStrategy):
    """Adapta um scaler do Scikit-Learn à interface de estratégia.

    Aplica o padrão *Template Method*: esta classe detém o esqueleto de
    fit/transform (validação, mudança de formato e guarda de estado) e delega
    o único passo variável — qual scaler instanciar — a :meth:`_build_scaler`.
    """

    def __init__(self) -> None:
        """Cria a estratégia ainda não ajustada."""
        self._scaler: TransformerMixin | None = None

    @abstractmethod
    def _build_scaler(self) -> TransformerMixin:
        """Cria o scaler do Scikit-Learn ao qual a estratégia delega.

        Returns:
            Uma instância de scaler ainda não ajustada.
        """

    def fit(self, values: Sequence[float]) -> SklearnScalerStrategy:
        """Ajusta o scaler interno em ``values``.

        Args:
            values: Valores de treino, não vazios.

        Returns:
            A estratégia ajustada.

        Raises:
            ValueError: Se ``values`` estiver vazio.
        """
        if len(values) == 0:
            raise ValueError(f"{type(self).__name__} não ajusta sequência vazia.")
        scaler = self._build_scaler()
        scaler.fit(_as_column(values))
        self._scaler = scaler
        return self

    def transform(self, values: Sequence[float]) -> list[float]:
        """Escalona ``values`` com o scaler já ajustado.

        Args:
            values: Valores a escalonar.

        Returns:
            Valores escalonados, na ordem da entrada.

        Raises:
            RuntimeError: Se chamado antes de :meth:`fit`.
        """
        if self._scaler is None:
            raise RuntimeError(f"{type(self).__name__} exige fit antes de transform.")
        if len(values) == 0:
            return []
        scaled = self._scaler.transform(_as_column(values))
        return [float(value) for value in np.asarray(scaled).ravel()]


class MinMaxScaler(SklearnScalerStrategy):
    """Escalona os valores numéricos para o intervalo ``[0, 1]``."""

    def _build_scaler(self) -> TransformerMixin:
        """Retorna um scaler min-max do Scikit-Learn.

        Returns:
            O scaler ainda não ajustado.
        """
        return SklearnMinMax()


class StandardScaler(SklearnScalerStrategy):
    """Padroniza os valores para média zero e variância unitária."""

    def _build_scaler(self) -> TransformerMixin:
        """Retorna um scaler padrão do Scikit-Learn.

        Returns:
            O scaler ainda não ajustado.
        """
        return SklearnStandard()


STRATEGY_BUILDERS: dict[str, Callable[[], PreprocessingStrategy]] = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
}


def available_strategies() -> list[str]:
    """Lista as estratégias que a configuração pode escolher.

    Returns:
        Nomes das estratégias, ordenados.
    """
    return sorted(STRATEGY_BUILDERS)


def build_strategy(name: str) -> PreprocessingStrategy:
    """Instancia a estratégia registrada sob ``name``.

    Args:
        name: Chave da estratégia, como escrita no arquivo de parâmetros.

    Returns:
        Uma estratégia nova, ainda não ajustada.

    Raises:
        KeyError: Se ``name`` não estiver registrado.
    """
    if name not in STRATEGY_BUILDERS:
        available = ", ".join(available_strategies())
        raise KeyError(f"Estratégia '{name}' desconhecida. Disponíveis: {available}.")
    return STRATEGY_BUILDERS[name]()


def build_pipeline(name: str, columns: Sequence[str]) -> PreprocessingPipeline:
    """Monta um pipeline com a mesma estratégia para todas as colunas.

    Args:
        name: Chave da estratégia aplicada a cada coluna.
        columns: Colunas de features a escalonar.

    Returns:
        Pipeline com uma instância de estratégia por coluna.
    """
    return PreprocessingPipeline({column: build_strategy(name) for column in columns})


def _as_column(values: Sequence[float]) -> np.ndarray:
    """Converte uma sequência simples no array 2D que o Scikit-Learn espera.

    Args:
        values: Sequência plana de números.

    Returns:
        Array no formato ``(len(values), 1)``.
    """
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)
