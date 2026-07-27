"""Stage 4 do DVC: compara os modelos e promove o melhor.

Execute com ``poetry run python -m recsys.pipelines.evaluate``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import pandas as pd

from recsys.data.interactions import InteractionData
from recsys.features.store import FeatureStore
from recsys.metrics import evaluate_model
from recsys.models import BASELINE_MODELS, NEURAL_MODEL
from recsys.models.persistence import load_model
from recsys.pipelines.context import ArtifactLayout, StageContext
from recsys.pipelines.train import format_metrics
from recsys.tracking.promotion import register_and_promote
from recsys.tracking.tracker import ExperimentTracker

EVALUATED_MODELS: tuple[str, ...] = (NEURAL_MODEL, *BASELINE_MODELS)


def main() -> int:
    """Avalia todos os modelos no teste e promove o vencedor.

    Returns:
        ``0`` em caso de sucesso.
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
        _write_outputs(context, results, best, promoted)
        _log_reports(tracker, context.layout)
    return 0


def _evaluate_all(
    context: StageContext, tracker: ExperimentTracker, test_data: InteractionData
) -> dict[str, dict[str, float]]:
    """Pontua todos os modelos treinados no split de teste."""
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
        print(f"[evaluate] {name:<12} " + format_metrics(metrics))
    return results


def _select_best(results: dict[str, dict[str, float]], primary_metric: str) -> str:
    """Escolhe o modelo com a maior métrica primária.

    Raises:
        KeyError: Se a métrica primária não tiver sido calculada.
    """
    missing = [name for name, values in results.items() if primary_metric not in values]
    if missing:
        raise KeyError(f"Métrica '{primary_metric}' ausente em: {missing}.")
    return max(results, key=lambda name: results[name][primary_metric])


def _promote(
    context: StageContext, tracker: ExperimentTracker, best: str
) -> dict[str, Any]:
    """Registra o modelo vencedor e o promove a Production."""
    report = json.loads(context.layout.train_report(best).read_text(encoding="utf-8"))
    promoted = register_and_promote(
        client=tracker.client,
        name=context.settings.mlflow_registered_model_name,
        run_id=report["run_id"],
        description=(
            f"Melhor modelo da comparação: {best} "
            f"({context.params.evaluation.primary_metric} no split de teste)."
        ),
    )
    print(
        f"[evaluate] {promoted.name} v{promoted.version} promovido para "
        f"{promoted.stage} (alias '{promoted.alias}')"
    )
    return asdict(promoted)


def _write_outputs(
    context: StageContext,
    results: dict[str, dict[str, float]],
    best: str,
    promoted: dict[str, Any],
) -> None:
    """Grava as métricas, a tabela de comparação e o registro da promoção."""
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


def _log_reports(tracker: ExperimentTracker, layout: ArtifactLayout) -> None:
    """Anexa os relatórios gerados à execução de avaliação."""
    for path in (layout.metrics, layout.comparison, layout.registry):
        tracker.log_file(path)


def _comparison_table(results: dict[str, dict[str, float]], best: str) -> str:
    """Renderiza a comparação dos modelos como tabela markdown."""
    metrics = sorted({name for values in results.values() for name in values})
    header = "| modelo | " + " | ".join(metrics) + " |"
    divider = "| --- " * (len(metrics) + 1) + "|"
    rows = [
        f"| {name}{' (melhor)' if name == best else ''} | "
        + " | ".join(f"{results[name][metric]:.4f}" for metric in metrics)
        + " |"
        for name in results
    ]
    body = "\n".join([header, divider, *rows])
    return f"# Comparação de modelos (split de teste)\n\n{body}\n"


if __name__ == "__main__":
    raise SystemExit(main())
