"""Stage 3 do DVC: treina um recomendador e rastreia a execução no MLflow.

Execute com ``poetry run python -m recsys.pipelines.train --model mlp``.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from recsys.data.interactions import InteractionData
from recsys.features.store import FeatureStore
from recsys.metrics import evaluate_model
from recsys.models import RecommenderModel, build_default_factory
from recsys.models.persistence import save_model
from recsys.pipelines.context import StageContext
from recsys.pipelines.params import model_kwargs
from recsys.tracking.pyfunc import build_input_example
from recsys.tracking.tracker import ExperimentTracker


def main() -> int:
    """Treina o modelo pedido, registra a execução e grava o relatório.

    Returns:
        ``0`` em caso de sucesso.
    """
    model_name = _parse_args().model
    context = StageContext.load()
    store = FeatureStore.load(context.layout.feature_store)
    train_data = _load_split(context, "train", store)
    validation_data = _load_split(context, "validation", store)
    model = build_default_factory().create(
        model_name, **model_kwargs(model_name, context.params, store)
    )
    report = _run_training(context, model, train_data, validation_data, store)
    target = context.layout.train_report(model_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[train] relatório gravado em {target}")
    return 0


def _parse_args() -> argparse.Namespace:
    """Lê os argumentos de linha de comando do stage.

    Returns:
        Namespace com a chave do modelo pedido.
    """
    parser = argparse.ArgumentParser(description="Treina um recomendador.")
    parser.add_argument(
        "--model",
        required=True,
        help="Chave do modelo na fábrica (mlp, popularity, logistic).",
    )
    return parser.parse_args()


def _load_split(
    context: StageContext, name: str, store: FeatureStore
) -> InteractionData:
    """Lê um split preparado e o converte nos arrays dos modelos.

    Args:
        context: Contexto do stage.
        name: Nome do split.
        store: Artefatos de features, que definem as colunas usadas.

    Returns:
        O split como :class:`InteractionData`.
    """
    frame = pd.read_parquet(context.layout.split(name))
    return InteractionData.from_frame(frame, store.feature_columns)


def _run_training(
    context: StageContext,
    model: RecommenderModel,
    train_data: InteractionData,
    validation_data: InteractionData,
    store: FeatureStore,
) -> dict[str, Any]:
    """Treina o modelo dentro de uma execução rastreada do MLflow.

    Args:
        context: Contexto do stage.
        model: Instância criada pela fábrica.
        train_data: Interações de treino.
        validation_data: Interações de validação, usadas no early stopping.
        store: Artefatos de features que descrevem a carga de inferência.

    Returns:
        O relatório de treino consumido pelo stage de avaliação.
    """
    tracker = context.tracker()
    tags = {"stage": "train", "model": model.name}
    with tracker.run(run_name=f"train-{model.name}", tags=tags) as run_id:
        tracker.log_params(context.params.flat())
        tracker.log_params(
            {f"model.{key}": value for key, value in model.hyperparameters().items()}
        )
        model.fit(train_data, validation_data)
        tracker.log_curves(model.training_history())
        metrics = _log_validation(tracker, context, model, validation_data)
        model_uri = _log_artifact(tracker, context, model, train_data, store)
        return {
            "model": model.name,
            "run_id": run_id,
            "model_uri": model_uri,
            "hyperparameters": model.hyperparameters(),
            "validation_metrics": metrics,
        }


def _log_validation(
    tracker: ExperimentTracker,
    context: StageContext,
    model: RecommenderModel,
    validation_data: InteractionData,
) -> dict[str, float]:
    """Avalia o modelo na validação e registra as métricas.

    Args:
        tracker: Tracker ativo do MLflow.
        context: Contexto do stage.
        model: Modelo já treinado.
        validation_data: Interações de validação.

    Returns:
        As métricas de validação.
    """
    metrics = evaluate_model(
        model,
        validation_data,
        top_k=context.params.evaluation.top_k,
        threshold=context.params.evaluation.decision_threshold,
    )
    tracker.log_metrics(metrics, prefix="validation_")
    print(f"[train] {model.name}: validação {format_metrics(metrics)}")
    return metrics


def _log_artifact(
    tracker: ExperimentTracker,
    context: StageContext,
    model: RecommenderModel,
    train_data: InteractionData,
    store: FeatureStore,
) -> str:
    """Grava o modelo e o registra como modelo implantável do MLflow.

    Args:
        tracker: Tracker ativo do MLflow.
        context: Contexto do stage.
        model: Modelo já treinado.
        train_data: Interações de treino, amostradas para o exemplo de entrada.
        store: Artefatos de features que descrevem a carga de inferência.

    Returns:
        URI do modelo registrado.
    """
    model_path = save_model(model, context.layout.model(model.name))
    example = build_input_example(train_data, store.feature_columns)
    return tracker.log_recommender(model_path, example)


def format_metrics(metrics: dict[str, float]) -> str:
    """Resume as métricas principais para o log do stage.

    Args:
        metrics: Mapa de métricas.

    Returns:
        Um resumo compacto e legível.
    """
    keys = ("roc_auc", "f1", "precision_at_k", "ndcg_at_k")
    return " ".join(f"{key}={metrics[key]:.4f}" for key in keys if key in metrics)


if __name__ == "__main__":
    raise SystemExit(main())
