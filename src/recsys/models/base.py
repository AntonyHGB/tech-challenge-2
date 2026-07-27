"""Classe base compartilhada por todos os recomendadores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from recsys.data.interactions import InteractionData


class RecommenderModel(ABC):
    """Interface comum que todo recomendador implementa.

    Os modelos concretos (a rede neural em PyTorch, os baselines do
    Scikit-Learn) dependem apenas desta abstração, o que mantém o pipeline, a
    avaliação e o tracking desacoplados de qualquer framework — princípio da
    Inversão de Dependência.

    Attributes:
        name: Chave sob a qual o modelo é registrado na fábrica.
    """

    name: ClassVar[str] = "recommender"

    @abstractmethod
    def fit(
        self, data: InteractionData, validation: InteractionData | None = None
    ) -> None:
        """Treina o modelo com as interações usuário-item.

        Args:
            data: Interações de treino.
            validation: Holdout opcional para early stopping ou monitoramento.
        """

    @abstractmethod
    def predict_proba(self, data: InteractionData) -> np.ndarray:
        """Calcula a probabilidade de cada interação ser relevante.

        Args:
            data: Interações a pontuar.

        Returns:
            Probabilidades em ``[0, 1]``, alinhadas com as linhas de entrada.
        """

    def hyperparameters(self) -> dict[str, Any]:
        """Descreve a configuração que o MLflow deve registrar.

        Returns:
            Mapa de hiperparâmetro para valor; vazio por padrão.
        """
        return {}

    def training_history(self) -> dict[str, list[float]]:
        """Curvas de aprendizado por época, quando o modelo as produz.

        Returns:
            Mapa de nome da curva para os valores por época; vazio por padrão.
        """
        return {}
