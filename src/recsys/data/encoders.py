"""Codificação dos identificadores brutos em índices contíguos."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class IdEncoder:
    """Mapeia identificadores de usuário/item para índices iniciados em zero.

    O vocabulário é fixado na construção, o que torna o encoder imutável e
    dispensa qualquer controle de estado "treinado/não treinado".
    """

    def __init__(self, classes: Sequence[int]) -> None:
        """Cria o encoder a partir de um vocabulário.

        Args:
            classes: Identificadores brutos, na ordem dos índices.

        Raises:
            ValueError: Se ``classes`` estiver vazio.
        """
        if len(classes) == 0:
            raise ValueError("IdEncoder precisa de ao menos um identificador.")
        self._classes = tuple(dict.fromkeys(int(value) for value in classes))
        self._positions = {value: index for index, value in enumerate(self._classes)}

    @classmethod
    def from_values(cls, values: Sequence[int]) -> IdEncoder:
        """Constrói o encoder com os valores distintos observados, ordenados.

        Args:
            values: Identificadores vistos no treino, com repetições.

        Returns:
            O encoder correspondente.
        """
        return cls(sorted({int(value) for value in values}))

    def transform(self, values: Sequence[int]) -> np.ndarray:
        """Converte identificadores em índices.

        Args:
            values: Identificadores a codificar; todos devem ser conhecidos.

        Returns:
            Array de índices iniciados em zero.

        Raises:
            KeyError: Se algum identificador estiver fora do vocabulário.
        """
        try:
            return np.array(
                [self._positions[int(value)] for value in values], dtype=np.int64
            )
        except KeyError as error:
            raise KeyError(f"Identificador desconhecido: {error.args[0]}.") from error

    def contains(self, value: int) -> bool:
        """Informa se um identificador pertence ao vocabulário.

        Args:
            value: Identificador a testar.

        Returns:
            ``True`` quando o identificador é conhecido.
        """
        return int(value) in self._positions

    @property
    def classes(self) -> list[int]:
        """Vocabulário na ordem dos índices."""
        return list(self._classes)

    def __len__(self) -> int:
        """Tamanho do vocabulário."""
        return len(self._classes)
