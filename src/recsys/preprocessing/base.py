"""Interface Strategy comum a todo passo de pré-processamento."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class PreprocessingStrategy(ABC):
    """Transformação intercambiável aplicada a uma coluna numérica.

    Implementa o padrão *Strategy*: cada subclasse concreta encapsula um
    algoritmo de transformação atrás de uma interface comum. Transformações
    novas entram por subclasse, sem alterar o código existente (princípio
    Aberto/Fechado).
    """

    @abstractmethod
    def fit(self, values: Sequence[float]) -> PreprocessingStrategy:
        """Aprende os parâmetros necessários para transformar valores futuros.

        Args:
            values: Valores de treino usados para estimar os parâmetros.

        Returns:
            A própria estratégia ajustada, permitindo encadear chamadas.
        """

    @abstractmethod
    def transform(self, values: Sequence[float]) -> list[float]:
        """Aplica a transformação aprendida a ``values``.

        Args:
            values: Valores a transformar.

        Returns:
            Valores transformados, na mesma ordem da entrada.
        """

    def fit_transform(self, values: Sequence[float]) -> list[float]:
        """Ajusta a estratégia e já transforma os mesmos valores.

        Args:
            values: Valores usados para ajustar e transformar.

        Returns:
            Valores transformados.
        """
        return self.fit(values).transform(values)
