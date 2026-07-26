"""DVC stage 3: train one recommender and track the run in MLflow.

Run with ``poetry run python -m recsys.pipelines.train --model mlp``.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from recsys.data.interactions import InteractionData
from recsys.features.store import FeatureStore
from recsys.metrics.evaluation import evaluate_model
from recsys.models.base import RecommenderModel
from recsys.models.persistence import save_model
from recsys.models.registry import build_default_factory
from recsys.pipelines.context import StageContext
from recsys.pipelines.model_config import model_kwargs
from recsys.tracking.pyfunc import build_input_example
from recsys.tracking.tracker import ExperimentTracker


def main() -> int:
    """Train the requested model, log the run and write its report.

    Returns:
        ``0`` on success.
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
    _write_report(context, model_name, report)
    return 0


def _parse_args() -> argparse.Namespace:
    """Parse the command line arguments of the stage.

    Returns:
        Namespace holding the requested model key.
    """
    parser = argparse.ArgumentParser(description="Train a recommender model.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model key registered in the factory (mlp, popularity, logistic).",
    )
    return parser.parse_args()


def _load_split(
    context: StageContext, name: str, store: FeatureStore
) -> InteractionData:
    """Read an engineered split into model-facing arrays.

    Args:
        context: Stage context.
        name: Split name.
        store: Feature artefacts describing the feature columns.

    Returns:
        The split as :class:`InteractionData`.
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
    """Fit the model inside a tracked MLflow run.

    Args:
        context: Stage context.
        model: Model instance built by the factory.
        train_data: Training interactions.
        validation_data: Validation interactions used for early stopping.
        store: Feature artefacts describing the serving payload.

    Returns:
        The training report persisted for the evaluate stage.
    """
    tracker = context.tracker()
    tags = {"stage": "train", "model": model.name}
    with tracker.run(run_name=f"train-{model.name}", tags=tags) as run_id:
        _log_configuration(tracker, context, model)
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


def _log_configuration(
    tracker: ExperimentTracker, context: StageContext, model: RecommenderModel
) -> None:
    """Log the pipeline parameters and the model hyper-parameters.

    Args:
        tracker: Active MLflow tracker.
        context: Stage context.
        model: Model about to be trained.
    """
    tracker.log_params(context.params.flat())
    tracker.log_params({f"model.{k}": v for k, v in model.hyperparameters().items()})


def _log_validation(
    tracker: ExperimentTracker,
    context: StageContext,
    model: RecommenderModel,
    validation_data: InteractionData,
) -> dict[str, float]:
    """Evaluate the trained model on the validation split and log the metrics.

    Args:
        tracker: Active MLflow tracker.
        context: Stage context.
        model: Trained model.
        validation_data: Validation interactions.

    Returns:
        The validation metrics.
    """
    metrics = evaluate_model(
        model,
        validation_data,
        top_k=context.params.evaluation.top_k,
        threshold=context.params.evaluation.decision_threshold,
    )
    tracker.log_metrics(metrics, prefix="validation_")
    print(f"[train] {model.name}: validation metrics {_format(metrics)}")
    return metrics


def _log_artifact(
    tracker: ExperimentTracker,
    context: StageContext,
    model: RecommenderModel,
    train_data: InteractionData,
    store: FeatureStore,
) -> str:
    """Persist the model and log it as a deployable MLflow model.

    Args:
        tracker: Active MLflow tracker.
        context: Stage context.
        model: Trained model.
        train_data: Training interactions, sampled for the input example.
        store: Feature artefacts describing the serving payload.

    Returns:
        URI of the logged model.
    """
    model_path = save_model(model, context.layout.model(model.name))
    example = build_input_example(train_data, store.feature_columns)
    return tracker.log_recommender(model_path, example)


def _write_report(
    context: StageContext, model_name: str, report: dict[str, Any]
) -> None:
    """Persist the training report as a DVC metric file.

    Args:
        context: Stage context.
        model_name: Registered model key.
        report: Report produced by the training run.
    """
    target = context.layout.train_report(model_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[train] report written to {target}")


def _format(metrics: dict[str, float]) -> str:
    """Render the headline metrics for the stage log.

    Args:
        metrics: Metric mapping.

    Returns:
        A compact, human-readable summary.
    """
    keys = ("roc_auc", "f1", "precision_at_k", "ndcg_at_k")
    return " ".join(f"{key}={metrics[key]:.4f}" for key in keys if key in metrics)


if __name__ == "__main__":
    raise SystemExit(main())
