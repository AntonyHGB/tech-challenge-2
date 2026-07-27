"""Stage 4 do DVC: compara os modelos e promove o melhor.

Execute com ``poetry run python -m recsys.pipelines.evaluate``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import joblib
import pandas as pd

from recsys.data.interactions import InteractionData
from recsys.features.store import FeatureStore
from recsys.metrics import evaluate_model
from recsys.models import BASELINE_MODELS, NEURAL_MODEL
from recsys.models.persistence import load_model
from recsys.pipelines.context import ArtifactLayout, StageContext
from recsys.pipelines.train import format_metrics
from recsys.ranker import TopKRanker
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
        _write_outputs(context, store, test_frame, results, best, promoted)
        _log_reports(tracker, context.layout)
    return 0


def _evaluate_all(
    context: StageContext, tracker: ExperimentTracker, test_data: InteractionData
) -> dict[str, dict[str, float]]:
    """Pontua todos os modelos treinados no split de teste.

    Args:
        context: Contexto do stage.
        tracker: Tracker ativo do MLflow.
        test_data: Interações de teste.

    Returns:
        Mapa de chave do modelo para as métricas de teste.
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
        print(f"[evaluate] {name:<12} " + format_metrics(metrics))
    return results


def _select_best(results: dict[str, dict[str, float]], primary_metric: str) -> str:
    """Escolhe o modelo com a maior métrica primária.

    Args:
        results: Métricas de teste de cada modelo.
        primary_metric: Métrica usada para ordenar os modelos.

    Returns:
        A chave do modelo vencedor.

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
    """Registra o modelo vencedor e o promove a Production.

    Args:
        context: Contexto do stage.
        tracker: Tracker ativo do MLflow.
        best: Chave do modelo vencedor.

    Returns:
        Mapa que descreve a versão promovida.
    """
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
    store: FeatureStore,
    test_frame: pd.DataFrame,
    results: dict[str, dict[str, float]],
    best: str,
    promoted: dict[str, Any],
) -> None:
    """Grava métricas, tabela de comparação e recomendações de exemplo.

    Args:
        context: Contexto do stage.
        store: Artefatos de features.
        test_frame: Frame de teste, usado para escolher usuários de exemplo.
        results: Métricas de teste de cada modelo.
        best: Chave do modelo vencedor.
        promoted: Descrição da versão promovida.
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


def _log_reports(tracker: ExperimentTracker, layout: ArtifactLayout) -> None:
    """Anexa os relatórios gerados à execução de avaliação.

    Args:
        tracker: Tracker ativo do MLflow.
        layout: Caminhos de artefatos resolvidos.
    """
    for path in (
        layout.metrics,
        layout.comparison,
        layout.registry,
        layout.recommendations,
    ):
        tracker.log_file(path)


def _comparison_table(results: dict[str, dict[str, float]], best: str) -> str:
    """Renderiza a comparação dos modelos como tabela markdown.

    Args:
        results: Métricas de teste de cada modelo.
        best: Chave do modelo vencedor.

    Returns:
        Documento markdown comparando todos os modelos.
    """
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


def _sample_recommendations(
    context: StageContext,
    store: FeatureStore,
    test_frame: pd.DataFrame,
    best: str,
) -> list[dict[str, Any]]:
    """Gera recomendações top-k de alguns usuários com o modelo vencedor.

    Args:
        context: Contexto do stage.
        store: Artefatos de features.
        test_frame: Frame de teste usado para escolher os usuários.
        best: Chave do modelo vencedor.

    Returns:
        Lista com as recomendações de cada usuário de exemplo.
    """
    sample_size = context.params.evaluation.sample_users
    if sample_size == 0 or test_frame.empty:
        return []
    ranker = TopKRanker(
        model=load_model(context.layout.model(best)),
        store=store,
        pipeline=joblib.load(context.layout.preprocessor),
    )
    top_k = context.params.evaluation.top_k
    users = test_frame["user_id"].drop_duplicates().head(sample_size)
    return [
        {
            "model": best,
            "user_id": int(user_id),
            "recommendations": [
                asdict(item) for item in ranker.recommend(int(user_id), top_k=top_k)
            ],
        }
        for user_id in users
    ]


if __name__ == "__main__":
    raise SystemExit(main())
