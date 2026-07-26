"""Model Registry promotion flow (Staging then Production)."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

PRODUCTION_ALIAS = "champion"
STAGING_ALIAS = "candidate"


@dataclass(frozen=True)
class PromotedModel:
    """Outcome of registering and promoting a model version.

    Attributes:
        name: Registered model name.
        version: Version created by the registration.
        stage: Final stage of the version.
        alias: Alias pointing at the production version.
        run_id: Run that produced the promoted model.
    """

    name: str
    version: str
    stage: str
    alias: str
    run_id: str


def register_and_promote(
    client: MlflowClient, name: str, run_id: str, description: str
) -> PromotedModel:
    """Register a run's model and walk it through Staging into Production.

    Args:
        client: MLflow client bound to the tracking backend.
        name: Registered model name.
        run_id: Run holding the logged ``model`` artifact.
        description: Human-readable note stored on the version.

    Returns:
        The promoted model version.
    """
    version = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=name)
    client.update_model_version(
        name=name, version=version.version, description=description
    )
    _transition(client, name, version.version, "Staging")
    _set_alias(client, name, version.version, STAGING_ALIAS)
    stage = _transition(client, name, version.version, "Production")
    _set_alias(client, name, version.version, PRODUCTION_ALIAS)
    return PromotedModel(
        name=name,
        version=str(version.version),
        stage=stage,
        alias=PRODUCTION_ALIAS,
        run_id=run_id,
    )


def _transition(client: MlflowClient, name: str, version: str, stage: str) -> str:
    """Move a version to ``stage``, tolerating backends without stages.

    Args:
        client: MLflow client.
        name: Registered model name.
        version: Version to move.
        stage: Target stage.

    Returns:
        The stage reached, or ``"aliased"`` when the backend dropped stages.
    """
    try:
        client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=stage == "Production",
        )
    except (MlflowException, AttributeError):
        return "aliased"
    return stage


def _set_alias(client: MlflowClient, name: str, version: str, alias: str) -> None:
    """Point an alias at a model version when the backend supports aliases.

    Args:
        client: MLflow client.
        name: Registered model name.
        version: Version the alias should point at.
        alias: Alias name.
    """
    try:
        client.set_registered_model_alias(name=name, alias=alias, version=version)
    except (MlflowException, AttributeError):
        return
