"""Stage 2 do DVC: monta splits, rótulos e features comportamentais.

Execute com ``poetry run python -m recsys.pipelines.feature_eng``.
"""

from __future__ import annotations

import pandas as pd

from recsys.data.encoders import IdEncoder
from recsys.data.interactions import FEATURE_COLUMNS
from recsys.data.splitter import DataSplits, drop_cold_start, split_by_user_history
from recsys.features.builder import (
    add_binary_label,
    fit_scale_frame,
    positive_rate,
    scale_frame,
)
from recsys.features.store import FeatureStore, InteractionStatistics
from recsys.pipelines.context import StageContext
from recsys.preprocessing import PreprocessingPipeline, build_pipeline

OUTPUT_COLUMNS: tuple[str, ...] = (
    "user_id",
    "item_id",
    "user_index",
    "item_index",
    "rating",
    "timestamp",
    "label",
    *FEATURE_COLUMNS,
)


def main() -> int:
    """Prepara as features de cada split e grava os artefatos.

    Returns:
        ``0`` em caso de sucesso.
    """
    context = StageContext.load()
    interactions = pd.read_parquet(context.layout.interactions)
    splits = drop_cold_start(
        split_by_user_history(
            interactions,
            validation_fraction=context.params.data.validation_fraction,
            test_fraction=context.params.data.test_fraction,
        )
    )
    statistics = InteractionStatistics.from_frame(splits.train)
    encoders = (
        IdEncoder.from_values(splits.train["user_id"]),
        IdEncoder.from_values(splits.train["item_id"]),
    )
    pipeline = build_pipeline(context.params.features.scaler, FEATURE_COLUMNS)
    _write_splits(context, splits, statistics, encoders, pipeline)
    _write_store(context, statistics, encoders)
    return 0


def _write_splits(
    context: StageContext,
    splits: DataSplits,
    statistics: InteractionStatistics,
    encoders: tuple[IdEncoder, IdEncoder],
    pipeline: PreprocessingPipeline,
) -> None:
    """Prepara e grava os três splits, ajustando o scaler só no treino."""
    threshold = context.params.features.positive_threshold
    for name, frame in splits.as_mapping().items():
        engineered = _engineer(frame, statistics, encoders, threshold)
        scaled = (
            fit_scale_frame(pipeline, engineered)
            if name == "train"
            else scale_frame(pipeline, engineered)
        )
        target = context.layout.split(name)
        scaled.loc[:, list(OUTPUT_COLUMNS)].to_parquet(target, index=False)
        print(
            f"[feature_eng] {name:<10} linhas={len(scaled):>6} "
            f"positivos={positive_rate(scaled):.3f} -> {target}"
        )


def _engineer(
    frame: pd.DataFrame,
    statistics: InteractionStatistics,
    encoders: tuple[IdEncoder, IdEncoder],
    threshold: float,
) -> pd.DataFrame:
    """Anexa features, rótulo e índices codificados a um split."""
    user_encoder, item_encoder = encoders
    engineered = add_binary_label(statistics.attach(frame), threshold)
    return engineered.assign(
        user_index=user_encoder.transform(engineered["user_id"]),
        item_index=item_encoder.transform(engineered["item_id"]),
    )


def _write_store(
    context: StageContext,
    statistics: InteractionStatistics,
    encoders: tuple[IdEncoder, IdEncoder],
) -> None:
    """Grava o feature store consumido pelo treino e pela avaliação."""
    user_encoder, item_encoder = encoders
    store = FeatureStore(
        statistics=statistics,
        user_classes=user_encoder.classes,
        item_classes=item_encoder.classes,
        feature_columns=FEATURE_COLUMNS,
        positive_threshold=context.params.features.positive_threshold,
    )
    store.save(context.layout.feature_store)
    print(
        f"[feature_eng] vocabulário usuários={store.n_users} "
        f"itens={store.n_items} -> {context.layout.feature_store}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
