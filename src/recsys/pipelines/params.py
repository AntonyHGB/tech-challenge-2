"""Hiperparâmetros validados e sua tradução em argumentos de modelo."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from recsys.features.store import FeatureStore
from recsys.models.baseline import LogisticRecommender, PopularityRecommender
from recsys.models.mlp import MLPRecommender


class StrictModel(BaseModel):
    """Base que rejeita parâmetros desconhecidos e proíbe mutação."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataParams(StrictModel):
    """Parâmetros de limpeza e divisão dos dados."""

    min_user_interactions: int = Field(default=5, ge=1)
    min_item_interactions: int = Field(default=5, ge=1)
    validation_fraction: float = Field(default=0.1, gt=0, lt=1)
    test_fraction: float = Field(default=0.1, gt=0, lt=1)


class FeatureParams(StrictModel):
    """Parâmetros de engenharia de features."""

    positive_threshold: float = Field(default=4.0, gt=0)
    scaler: str = Field(default="standard")


class ModelParams(StrictModel):
    """Arquitetura do recomendador neural."""

    embedding_dim: int = Field(default=32, ge=2)
    hidden_dim: int = Field(default=64, ge=2)
    dropout: float = Field(default=0.2, ge=0, lt=1)


class TrainingParams(StrictModel):
    """Parâmetros de otimização do recomendador neural."""

    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=512, ge=1)
    learning_rate: float = Field(default=3e-3, gt=0)
    weight_decay: float = Field(default=1e-5, ge=0)
    early_stopping_patience: int = Field(default=3, ge=0)


class BaselineParams(StrictModel):
    """Configuração dos baselines do Scikit-Learn."""

    popularity_smoothing: float = Field(default=10.0, ge=0)
    logistic_penalty_strength: float = Field(default=1.0, gt=0)
    logistic_max_iterations: int = Field(default=1000, ge=1)


class EvaluationParams(StrictModel):
    """Parâmetros de comparação e de promoção."""

    top_k: int = Field(default=10, ge=1)
    decision_threshold: float = Field(default=0.5, gt=0, lt=1)
    primary_metric: str = Field(default="roc_auc")
    sample_users: int = Field(default=3, ge=0)


class Params(StrictModel):
    """Raiz do arquivo de hiperparâmetros."""

    seed: int = Field(default=42)
    data: DataParams = DataParams()
    features: FeatureParams = FeatureParams()
    model: ModelParams = ModelParams()
    training: TrainingParams = TrainingParams()
    baselines: BaselineParams = BaselineParams()
    evaluation: EvaluationParams = EvaluationParams()

    def flat(self) -> dict[str, float | int | str]:
        """Achata os parâmetros para registro no MLflow.

        Returns:
            Mapa de ``secao.nome`` para o valor.
        """
        flattened: dict[str, float | int | str] = {"seed": self.seed}
        for section, values in self.model_dump(exclude={"seed"}).items():
            for name, value in values.items():
                flattened[f"{section}.{name}"] = value
        return flattened


def load_params(path: Path) -> Params:
    """Lê e valida o arquivo de hiperparâmetros.

    Args:
        path: Localização do arquivo YAML de parâmetros.

    Returns:
        Os parâmetros validados.

    Raises:
        FileNotFoundError: Se ``path`` não existir.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de parâmetros '{path}' não encontrado.")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Params.model_validate(payload)


def _mlp_kwargs(params: Params, store: FeatureStore) -> dict[str, Any]:
    """Monta os argumentos do recomendador neural.

    Args:
        params: Parâmetros validados do pipeline.
        store: Artefatos de features, que informam o tamanho dos vocabulários.

    Returns:
        Argumentos nomeados de :class:`MLPRecommender`.
    """
    return {
        "n_users": store.n_users,
        "n_items": store.n_items,
        "embedding_dim": params.model.embedding_dim,
        "hidden_dim": params.model.hidden_dim,
        "dropout": params.model.dropout,
        "epochs": params.training.epochs,
        "batch_size": params.training.batch_size,
        "learning_rate": params.training.learning_rate,
        "weight_decay": params.training.weight_decay,
        "early_stopping_patience": params.training.early_stopping_patience,
        "seed": params.seed,
    }


def _popularity_kwargs(params: Params, store: FeatureStore) -> dict[str, Any]:
    """Monta os argumentos do baseline de popularidade.

    Args:
        params: Parâmetros validados do pipeline.
        store: Não utilizado; mantém a assinatura uniforme.

    Returns:
        Argumentos nomeados de :class:`PopularityRecommender`.
    """
    return {"smoothing": params.baselines.popularity_smoothing}


def _logistic_kwargs(params: Params, store: FeatureStore) -> dict[str, Any]:
    """Monta os argumentos do baseline logístico.

    Args:
        params: Parâmetros validados do pipeline.
        store: Não utilizado; mantém a assinatura uniforme.

    Returns:
        Argumentos nomeados de :class:`LogisticRecommender`.
    """
    return {
        "penalty_strength": params.baselines.logistic_penalty_strength,
        "max_iterations": params.baselines.logistic_max_iterations,
        "seed": params.seed,
    }


KWARGS_BUILDERS: dict[str, Callable[[Params, FeatureStore], dict[str, Any]]] = {
    MLPRecommender.name: _mlp_kwargs,
    PopularityRecommender.name: _popularity_kwargs,
    LogisticRecommender.name: _logistic_kwargs,
}


def model_kwargs(
    model_name: str, params: Params, store: FeatureStore
) -> dict[str, Any]:
    """Resolve os argumentos de construção de um modelo registrado.

    Args:
        model_name: Chave do modelo registrado.
        params: Parâmetros validados do pipeline.
        store: Artefatos de features.

    Returns:
        Argumentos nomeados a entregar à fábrica.

    Raises:
        KeyError: Se não houver construtor de argumentos para ``model_name``.
    """
    if model_name not in KWARGS_BUILDERS:
        available = ", ".join(sorted(KWARGS_BUILDERS))
        raise KeyError(
            f"Sem parâmetros para o modelo '{model_name}'. Conhecidos: {available}."
        )
    return KWARGS_BUILDERS[model_name](params, store)
