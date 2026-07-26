"""DVC stage 4: compare the models and promote the best one.

Run with ``poetry run python -m recsys.pipelines.evaluate``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import joblib
import pandas as pd

from recsys.data.interactions import InteractionData
from recsys.features.store import FeatureStore
from recsys.metrics.evaluation import evaluate_model
from recsys.models.persistence import load_model
from recsys.models.registry import BASELINE_MODELS, NEURAL_MODEL
from recsys.pipelines.artifacts import ArtifactLayout
from recsys.pipelines.context import StageContext
from recsys.serving.ranker import TopKRanker
from recsys.tracking.promotion import register_and_promote
from recsys.tracking.tracker import ExperimentTracker

EVALUATED_MODELS: tuple[str, ...] = (NEURAL_MODEL, *BASELINE_MODELS)


def main() -> int:
    """Evaluate every trained model on the test split and promote the winner.

    Returns:
        ``0`` on success.
    """
    context = StageContext.load()
    store = FeatureStore.load(context.layout.feature_store)
    test_frame = pd.read_parquet(context.layout.split("test"))
    test_data = InteractionData.from_frame(test_frame, store.feature_columns)
    tracker = context.tracker()
    with tracker.run(run_name="evaluate", tags={"stage": "evaluate"}):
        results = _evaluate_all(context, tracker, test_data)
        best = _select_best(results, context.params.evaluation.primary_metric)
        promoted = _promote(context, tracker, best)
        _write_outputs(context, store, test_frame, results, best, promoted)
        _log_reports(tracker, context.layout)
    return 0


def _log_reports(tracker: ExperimentTracker, layout: ArtifactLayout) -> None:
    """Attach the generated reports to the evaluation run.

    Args:
        tracker: Active MLflow tracker.
        layout: Resolved artifact paths.
    """
    for path in (
        layout.metrics,
        layout.comparison,
        layout.registry,
        layout.recommendations,
    ):
        tracker.log_file(path)


def _evaluate_all(
    context: StageContext, tracker: ExperimentTracker, test_data: InteractionData
) -> dict[str, dict[str, float]]:
    """Score every trained model on the test split.

    Args:
        context: Stage context.
        tracker: Active MLflow tracker.
        test_data: Test interactions.

    Returns:
        Mapping of model key to its test metrics.
    """
    results: dict[str, dict[str, float]] = {}
    for name in EVALUATED_MODELS:
        model = load_model(context.layout.model(name))
        metrics = evaluate_model(
            model,
            test_data,
            top_k=context.params.evaluation.top_k,
            threshold=context.params.evaluation.decision_threshold,
        )
        tracker.log_metrics(metrics, prefix=f"test_{name}_")
        results[name] = metrics
        print(f"[evaluate] {name:<12} " + _format(metrics))
    return results


def _select_best(results: dict[str, dict[str, float]], primary_metric: str) -> str:
    """Pick the model with the highest primary metric.

    Args:
        results: Test metrics of every model.
        primary_metric: Metric used to rank the models.

    Returns:
        The winning model key.

    Raises:
        KeyError: If the primary metric was not computed.
    """
    missing = [name for name, values in results.items() if primary_metric not in values]
    if missing:
        raise KeyError(f"Metric '{primary_metric}' missing for: {missing}.")
    return max(results, key=lambda name: results[name][primary_metric])


def _promote(
    context: StageContext, tracker: ExperimentTracker, best: str
) -> dict[str, Any]:
    """Register the winning model and promote it to Production.

    Args:
        context: Stage context.
        tracker: Active MLflow tracker.
        best: Winning model key.

    Returns:
        Mapping describing the promoted model version.
    """
    report = json.loads(context.layout.train_report(best).read_text(encoding="utf-8"))
    promoted = register_and_promote(
        client=tracker.client,
        name=context.settings.mlflow_registered_model_name,
        run_id=report["run_id"],
        description=(
            f"Best model of the comparison: {best} "
            f"({context.params.evaluation.primary_metric} on the test split)."
        ),
    )
    print(
        f"[evaluate] promoted {promoted.name} v{promoted.version} "
        f"to {promoted.stage} (alias '{promoted.alias}')"
    )
    return asdict(promoted)


def _write_outputs(
    context: StageContext,
    store: FeatureStore,
    test_frame: pd.DataFrame,
    results: dict[str, dict[str, float]],
    best: str,
    promoted: dict[str, Any],
) -> None:
    """Persist metrics, the comparison table and sample recommendations.

    Args:
        context: Stage context.
        store: Feature artefacts.
        test_frame: Test split frame, used to pick sample users.
        results: Test metrics of every model.
        best: Winning model key.
        promoted: Description of the promoted model version.
    """
    layout = context.layout
    layout.reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_model": best,
        "primary_metric": context.params.evaluation.primary_metric,
        "test": results,
    }
    layout.metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    layout.comparison.write_text(_comparison_table(results, best), encoding="utf-8")
    layout.registry.write_text(json.dumps(promoted, indent=2), encoding="utf-8")
    samples = _sample_recommendations(context, store, test_frame, best)
    layout.recommendations.write_text(json.dumps(samples, indent=2), encoding="utf-8")


def _comparison_table(results: dict[str, dict[str, float]], best: str) -> str:
    """Render the model comparison as a markdown table.

    Args:
        results: Test metrics of every model.
        best: Winning model key.

    Returns:
        Markdown document comparing every model.
    """
    metrics = sorted({name for values in results.values() for name in values})
    header = "| modelo | " + " | ".join(metrics) + " |"
    divider = "| --- " * (len(metrics) + 1) + "|"
    rows = [
        f"| {name}{' (melhor)' if name == best else ''} | "
        + " | ".join(
            f"{results[name].get(metric, float('nan')):.4f}" for metric in metrics
        )
        + " |"
        for name in results
    ]
    body = "\n".join([header, divider, *rows])
    return f"# Comparacao de modelos (split de teste)\n\n{body}\n"


def _sample_recommendations(
    context: StageContext,
    store: FeatureStore,
    test_frame: pd.DataFrame,
    best: str,
) -> list[dict[str, Any]]:
    """Produce top-k recommendations for a few users with the winning model.

    Args:
        context: Stage context.
        store: Feature artefacts.
        test_frame: Test split frame used to pick sample users.
        best: Winning model key.

    Returns:
        List of per-user recommendation payloads.
    """
    sample_size = context.params.evaluation.sample_users
    if sample_size == 0 or test_frame.empty:
        return []
    ranker = TopKRanker(
        model=load_model(context.layout.model(best)),
        store=store,
        pipeline=joblib.load(context.layout.preprocessor),
    )
    users = test_frame["user_id"].drop_duplicates().head(sample_size)
    top_k = context.params.evaluation.top_k
    return [_user_payload(ranker, best, int(user_id), top_k) for user_id in users]


def _user_payload(
    ranker: TopKRanker, model_name: str, user_id: int, top_k: int
) -> dict[str, Any]:
    """Build the recommendation payload of a single user.

    Args:
        ranker: Ranker wired to the winning model.
        model_name: Key of the model producing the ranking.
        user_id: Raw user identifier.
        top_k: Number of items to recommend.

    Returns:
        Mapping with the user and their ranked items.
    """
    return {
        "model": model_name,
        "user_id": user_id,
        "recommendations": [
            asdict(item) for item in ranker.recommend(user_id, top_k=top_k)
        ],
    }


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
