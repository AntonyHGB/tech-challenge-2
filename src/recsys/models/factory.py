"""Fábrica que constrói recomendadores a partir de um nome."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from recsys.models.base import RecommenderModel
from recsys.models.baseline import LogisticRecommender, PopularityRecommender
from recsys.models.mlp import MLPRecommender

NEURAL_MODEL = MLPRecommender.name
BASELINE_MODELS: tuple[str, ...] = (
    PopularityRecommender.name,
    LogisticRecommender.name,
)


class ModelFactory:
    """Cria instâncias de :class:`RecommenderModel` a partir de uma chave.

    Implementa o padrão *Factory*. Os modelos são registrados sob um nome e
    quem chama os constrói sem importar as classes concretas, o que concentra a
    criação em um só lugar e mantém o código aberto para extensão.
    """

    def __init__(self) -> None:
        """Cria uma fábrica vazia."""
        self._builders: dict[str, Callable[..., RecommenderModel]] = {}

    def register(self, name: str, builder: Callable[..., RecommenderModel]) -> None:
        """Registra um construtor sob ``name``.

        Args:
            name: Chave única que identifica o modelo.
            builder: Chamável que devolve um novo ``RecommenderModel``.

        Raises:
            ValueError: Se ``name`` já estiver registrado.
        """
        if name in self._builders:
            raise ValueError(f"O modelo '{name}' já está registrado.")
        self._builders[name] = builder

    def create(self, name: str, **kwargs: Any) -> RecommenderModel:
        """Instancia o modelo registrado sob ``name``.

        Args:
            name: Chave do modelo a construir.
            **kwargs: Argumentos repassados ao construtor.

        Returns:
            Uma nova instância de ``RecommenderModel``.

        Raises:
            KeyError: Se ``name`` não estiver registrado.
        """
        if name not in self._builders:
            available = ", ".join(self.available()) or "<nenhum>"
            raise KeyError(f"Modelo '{name}' desconhecido. Disponíveis: {available}.")
        return self._builders[name](**kwargs)

    def available(self) -> list[str]:
        """Lista os modelos registrados.

        Returns:
            Nomes dos modelos registrados, ordenados.
        """
        return sorted(self._builders)


def build_default_factory() -> ModelFactory:
    """Cria a fábrica já carregada com os modelos do projeto.

    Returns:
        Fábrica capaz de construir a rede neural e os dois baselines.
    """
    factory = ModelFactory()
    factory.register(MLPRecommender.name, MLPRecommender)
    factory.register(PopularityRecommender.name, PopularityRecommender)
    factory.register(LogisticRecommender.name, LogisticRecommender)
    return factory
