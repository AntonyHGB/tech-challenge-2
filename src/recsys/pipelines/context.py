"""Shared bootstrap for every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass

from recsys.config import Settings, get_settings
from recsys.pipelines.artifacts import ArtifactLayout
from recsys.pipelines.params import Params, load_params
from recsys.seeding import set_global_seed
from recsys.tracking.tracker import ExperimentTracker


@dataclass(frozen=True)
class StageContext:
    """Settings, parameters and paths a stage needs to run.

    Attributes:
        settings: Environment-driven application settings.
        params: Validated hyper-parameters of the pipeline.
        layout: Resolved artifact paths.
    """

    settings: Settings
    params: Params
    layout: ArtifactLayout

    @classmethod
    def load(cls) -> StageContext:
        """Load the configuration and seed every generator.

        Returns:
            The initialised stage context.
        """
        settings = get_settings()
        params = load_params(settings.params_file)
        set_global_seed(params.seed)
        return cls(
            settings=settings,
            params=params,
            layout=ArtifactLayout.from_settings(settings),
        )

    def tracker(self) -> ExperimentTracker:
        """Build the MLflow tracker for the configured backend.

        Returns:
            A tracker bound to the project experiment.
        """
        return ExperimentTracker(
            tracking_uri=self.settings.mlflow_tracking_uri,
            experiment_name=self.settings.mlflow_experiment_name,
            artifact_location=self.settings.mlflow_artifact_location,
        )
