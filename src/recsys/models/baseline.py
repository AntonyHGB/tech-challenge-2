"""Baselines do Scikit-Learn usados como referência para a rede neural."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from sklearn.linear_model import LogisticRegression

from recsys.data.interactions import InteractionData
from recsys.models.base import RecommenderModel


class PopularityRecommender(RecommenderModel):
    """Baseline não personalizado, que pontua o item pelo apelo histórico.

    Cada item recebe a proporção suavizada de interações positivas que obteve
    no treino, então os itens populares aparecem primeiro para todo mundo. É o
    piso que qualquer modelo personalizado precisa superar.
    """

    name: ClassVar[str] = "popularity"

    def __init__(self, smoothing: float = 10.0) -> None:
        """Configura o baseline.

        Args:
            smoothing: Intensidade da atração para a taxa global de positivos,
                que protege itens com pouquíssimas interações.
        """
        self.smoothing = smoothing
        self._item_scores: dict[int, float] = {}
        self._prior = 0.5

    def fit(
        self, data: InteractionData, validation: InteractionData | None = None
    ) -> None:
        """Calcula a taxa suavizada de positivos de cada item.

        Args:
            data: Interações de treino.
            validation: Não utilizado; existe por compatibilidade de interface.
        """
        self._prior = float(data.labels.mean()) if len(data) else 0.5
        positives = np.bincount(data.item_indices, weights=data.labels, minlength=1)
        counts = np.bincount(data.item_indices, minlength=1)
        smoothed = (positives + self.smoothing * self._prior) / (
            counts + self.smoothing
        )
        self._item_scores = {
            index: float(score)
            for index, score in enumerate(smoothed)
            if counts[index] > 0
        }

    def predict_proba(self, data: InteractionData) -> np.ndarray:
        """Pontua as interações pela popularidade do item.

        Args:
            data: Interações a pontuar.

        Returns:
            Popularidade de cada item, com a taxa global como fallback.
        """
        return np.array(
            [
                self._item_scores.get(int(item), self._prior)
                for item in data.item_indices
            ],
            dtype=np.float64,
        )

    def hyperparameters(self) -> dict[str, Any]:
        """Descreve a configuração registrada no MLflow.

        Returns:
            Mapa de hiperparâmetro para valor.
        """
        return {"smoothing": self.smoothing}


class LogisticRecommender(RecommenderModel):
    """Regressão logística do Scikit-Learn sobre as features comportamentais.

    Diferente da rede neural, não faz ideia de quem é o usuário nem qual é o
    item: enxerga apenas as features agregadas de navegação. Isso a torna uma
    referência justa para medir o quanto os embeddings realmente agregam.
    """

    name: ClassVar[str] = "logistic"

    def __init__(
        self,
        penalty_strength: float = 1.0,
        max_iterations: int = 1000,
        seed: int = 42,
    ) -> None:
        """Configura o estimador.

        Args:
            penalty_strength: Inverso da força de regularização (``C``).
            max_iterations: Máximo de iterações do solver.
            seed: Semente entregue ao solver, para reprodutibilidade.
        """
        self.penalty_strength = penalty_strength
        self.max_iterations = max_iterations
        self.seed = seed
        self._estimator: LogisticRegression | None = None

    def fit(
        self, data: InteractionData, validation: InteractionData | None = None
    ) -> None:
        """Ajusta a regressão logística nas features comportamentais.

        Args:
            data: Interações de treino.
            validation: Não utilizado; existe por compatibilidade de interface.
        """
        estimator = LogisticRegression(
            C=self.penalty_strength,
            max_iter=self.max_iterations,
            random_state=self.seed,
        )
        estimator.fit(data.features, data.labels)
        self._estimator = estimator

    def predict_proba(self, data: InteractionData) -> np.ndarray:
        """Pontua as interações com o estimador ajustado.

        Args:
            data: Interações a pontuar.

        Returns:
            Probabilidade da classe positiva para cada linha.

        Raises:
            RuntimeError: Se o modelo ainda não tiver sido treinado.
        """
        if self._estimator is None:
            raise RuntimeError("LogisticRecommender exige fit antes de pontuar.")
        return self._estimator.predict_proba(data.features)[:, 1].astype(np.float64)

    def hyperparameters(self) -> dict[str, Any]:
        """Descreve a configuração registrada no MLflow.

        Returns:
            Mapa de hiperparâmetro para valor.
        """
        return {
            "penalty_strength": self.penalty_strength,
            "max_iterations": self.max_iterations,
        }
