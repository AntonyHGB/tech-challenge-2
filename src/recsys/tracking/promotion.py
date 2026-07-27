"""Fluxo de promoção no Model Registry (Staging e depois Production)."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

PRODUCTION_ALIAS = "champion"
STAGING_ALIAS = "candidate"


@dataclass(frozen=True)
class PromotedModel:
    """Resultado do registro e da promoção de uma versão de modelo.

    Attributes:
        name: Nome do modelo registrado.
        version: Versão criada pelo registro.
        stage: Estágio final da versão.
        alias: Alias que aponta para a versão em produção.
        run_id: Execução que produziu o modelo promovido.
    """

    name: str
    version: str
    stage: str
    alias: str
    run_id: str


def register_and_promote(
    client: MlflowClient, name: str, run_id: str, description: str
) -> PromotedModel:
    """Registra o modelo de uma execução e o leva de Staging a Production.

    Args:
        client: Cliente do MLflow ligado ao backend de tracking.
        name: Nome do modelo registrado.
        run_id: Execução que contém o artefato ``model``.
        description: Observação legível gravada na versão.

    Returns:
        A versão promovida do modelo.
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
    """Move a versão para ``stage``, tolerando backends sem estágios.

    Args:
        client: Cliente do MLflow.
        name: Nome do modelo registrado.
        version: Versão a mover.
        stage: Estágio de destino.

    Returns:
        O estágio alcançado, ou ``"aliased"`` se o backend não tiver estágios.
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
    """Aponta um alias para a versão, se o backend suportar aliases.

    Args:
        client: Cliente do MLflow.
        name: Nome do modelo registrado.
        version: Versão para a qual o alias deve apontar.
        alias: Nome do alias.
    """
    try:
        client.set_registered_model_alias(name=name, alias=alias, version=version)
    except (MlflowException, AttributeError):
        return
