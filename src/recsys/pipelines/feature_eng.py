"""DVC stage 2: build splits, labels and behavioural features.

Run with ``poetry run python -m recsys.pipelines.feature_eng``.
"""

from __future__ import annotations

import joblib
import pandas as pd

from recsys.data.encoders import IdEncoder
from recsys.data.interactions import FEATURE_COLUMNS
from recsys.data.splitter import DataSplits, drop_cold_start, split_by_user_history
from recsys.features.labels import add_binary_label, positive_rate
from recsys.features.scaling import fit_scale_frame, scale_frame
from recsys.features.statistics import InteractionStatistics
from recsys.features.store import FeatureStore
from recsys.pipelines.context import StageContext
from recsys.preprocessing.pipeline import PreprocessingPipeline
from recsys.preprocessing.registry import build_pipeline

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
    """Engineer the features of every split and persist the artefacts.

    Returns:
        ``0`` on success.
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
    encoders = _fit_encoders(splits)
    pipeline = build_pipeline(context.params.features.scaler, FEATURE_COLUMNS)
    _write_splits(context, splits, statistics, encoders, pipeline)
    _write_artifacts(context, statistics, encoders, pipeline)
    return 0


def _fit_encoders(splits: DataSplits) -> tuple[IdEncoder, IdEncoder]:
    """Fit the user and item encoders on the training split.

    Args:
        splits: Chronological splits.

    Returns:
        The fitted user and item encoders.
    """
    user_encoder = IdEncoder().fit(splits.train["user_id"].tolist())
    item_encoder = IdEncoder().fit(splits.train["item_id"].tolist())
    return user_encoder, item_encoder


def _write_splits(
    context: StageContext,
    splits: DataSplits,
    statistics: InteractionStatistics,
    encoders: tuple[IdEncoder, IdEncoder],
    pipeline: PreprocessingPipeline,
) -> None:
    """Engineer and persist the three splits, fitting the scaler on train only.

    Args:
        context: Stage context.
        splits: Chronological splits.
        statistics: Aggregates learned on the training split.
        encoders: Fitted user and item encoders.
        pipeline: Preprocessing pipeline to fit on the training split.
    """
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
            f"[feature_eng] {name:<10} rows={len(scaled):>6} "
            f"positive_rate={positive_rate(scaled):.3f} -> {target}"
        )


def _engineer(
    frame: pd.DataFrame,
    statistics: InteractionStatistics,
    encoders: tuple[IdEncoder, IdEncoder],
    threshold: float,
) -> pd.DataFrame:
    """Attach features, labels and encoded indices to a split.

    Args:
        frame: Raw split frame.
        statistics: Aggregates learned on the training split.
        encoders: Fitted user and item encoders.
        threshold: Rating above which an interaction is relevant.

    Returns:
        The engineered frame.
    """
    user_encoder, item_encoder = encoders
    engineered = add_binary_label(statistics.attach(frame), threshold)
    engineered["user_index"] = user_encoder.transform(engineered["user_id"].tolist())
    engineered["item_index"] = item_encoder.transform(engineered["item_id"].tolist())
    return engineered


def _write_artifacts(
    context: StageContext,
    statistics: InteractionStatistics,
    encoders: tuple[IdEncoder, IdEncoder],
    pipeline: PreprocessingPipeline,
) -> None:
    """Persist the feature store and the fitted preprocessing pipeline.

    Args:
        context: Stage context.
        statistics: Aggregates learned on the training split.
        encoders: Fitted user and item encoders.
        pipeline: Pipeline already fitted on the training split.
    """
    user_encoder, item_encoder = encoders
    store = FeatureStore(
        statistics=statistics,
        user_classes=user_encoder.classes(),
        item_classes=item_encoder.classes(),
        feature_columns=FEATURE_COLUMNS,
        positive_threshold=context.params.features.positive_threshold,
    )
    store.save(context.layout.feature_store)
    joblib.dump(pipeline, context.layout.preprocessor)
    print(
        f"[feature_eng] vocabulary users={store.n_users} items={store.n_items} "
        f"-> {context.layout.feature_store}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
