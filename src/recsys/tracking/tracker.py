"""Thin MLflow facade used by the pipeline stages."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.entities import Experiment
from mlflow.tracking import MlflowClient

import recsys
from recsys.tracking.pyfunc import PIP_REQUIREMENTS, RecommenderPyfunc

MODEL_ARTIFACT_PATH = "model"


class ExperimentTracker:
    """Log params, metrics and artifacts of a stage to MLflow.

    Concentrating the MLflow calls here keeps the pipeline stages readable and
    makes the tracking backend a configuration detail rather than a dependency
    spread across the code base.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        artifact_location: str | None = None,
    ) -> None:
        """Point MLflow at the configured backend and experiment.

        Args:
            tracking_uri: Tracking server or local store URI.
            experiment_name: Experiment grouping every run of the project.
            artifact_location: Where a newly created experiment stores its
                artifacts. Ignored when the experiment already exists; empty
                means "let the tracking server decide".
        """
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._experiment = self._ensure_experiment(experiment_name, artifact_location)
        mlflow.set_experiment(experiment_name)

    @property
    def client(self) -> MlflowClient:
        """Underlying MLflow client, for registry operations.

        Returns:
            The configured client.
        """
        return self._client

    @property
    def experiment_id(self) -> str:
        """Identifier of the experiment every run is logged to.

        Returns:
            The experiment id.
        """
        return self._experiment.experiment_id

    @contextmanager
    def run(
        self, run_name: str, tags: Mapping[str, str] | None = None
    ) -> Iterator[str]:
        """Open an MLflow run as a context manager.

        Args:
            run_name: Human-readable name of the run.
            tags: Optional tags attached to the run.

        Yields:
            The active run id.
        """
        with mlflow.start_run(run_name=run_name, tags=dict(tags or {})) as active:
            yield active.info.run_id

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log flat hyper-parameters of the active run.

        Args:
            params: Mapping of parameter name to value.
        """
        mlflow.log_params(dict(params))

    def log_metrics(self, metrics: Mapping[str, float], prefix: str = "") -> None:
        """Log scalar metrics of the active run.

        Args:
            metrics: Mapping of metric name to value.
            prefix: Optional prefix, e.g. ``"test"`` or ``"validation"``.
        """
        mlflow.log_metrics(
            {f"{prefix}{name}": float(value) for name, value in metrics.items()}
        )

    def log_curves(self, curves: Mapping[str, Sequence[float]]) -> None:
        """Log per-epoch learning curves as stepped metrics.

        Args:
            curves: Mapping of curve name to its per-epoch values.
        """
        for name, values in curves.items():
            for step, value in enumerate(values, start=1):
                mlflow.log_metric(name, float(value), step=step)

    def log_file(self, path: Path) -> None:
        """Log an existing file as a run artifact.

        Args:
            path: File to upload.
        """
        if path.exists():
            mlflow.log_artifact(str(path))

    def log_recommender(
        self, model_path: Path, input_example: pd.DataFrame | None = None
    ) -> str:
        """Log a persisted recommender as an MLflow ``pyfunc`` model.

        Args:
            model_path: Serialised recommender to bundle with the wrapper.
            input_example: Sample payload used to infer the model signature.

        Returns:
            URI of the logged model inside the active run.
        """
        info = mlflow.pyfunc.log_model(
            artifact_path=MODEL_ARTIFACT_PATH,
            python_model=RecommenderPyfunc(),
            artifacts={"model": str(model_path)},
            code_paths=[str(Path(recsys.__file__).parent)],
            pip_requirements=PIP_REQUIREMENTS,
            input_example=input_example,
        )
        return info.model_uri

    def _ensure_experiment(
        self, name: str, artifact_location: str | None
    ) -> Experiment:
        """Fetch the experiment, creating it on first use.

        Args:
            name: Experiment name.
            artifact_location: Artifact root used when creating it.

        Returns:
            The experiment entity.
        """
        existing = self._client.get_experiment_by_name(name)
        if existing is not None:
            return existing
        experiment_id = self._client.create_experiment(
            name, artifact_location=artifact_location or None
        )
        return self._client.get_experiment(experiment_id)
