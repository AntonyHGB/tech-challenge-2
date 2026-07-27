"""Fachada enxuta do MLflow usada pelos stages do pipeline."""

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
    """Registra parâmetros, métricas e artefatos de um stage no MLflow.

    Concentrar aqui as chamadas ao MLflow mantém os stages legíveis e faz do
    backend de tracking um detalhe de configuração, em vez de uma dependência
    espalhada pelo código.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
        artifact_location: str | None = None,
    ) -> None:
        """Aponta o MLflow para o backend e o experimento configurados.

        Args:
            tracking_uri: URI do servidor ou do store local de tracking.
            experiment_name: Experimento que agrupa todas as execuções.
            artifact_location: Onde um experimento recém-criado guarda os
                artefatos. É ignorado se o experimento já existir; vazio
                significa "deixar o servidor de tracking decidir".
        """
        mlflow.set_tracking_uri(tracking_uri)
        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._experiment = self._ensure_experiment(experiment_name, artifact_location)
        mlflow.set_experiment(experiment_name)

    @property
    def client(self) -> MlflowClient:
        """Cliente do MLflow, para as operações de registry.

        Returns:
            O cliente configurado.
        """
        return self._client

    @contextmanager
    def run(
        self, run_name: str, tags: Mapping[str, str] | None = None
    ) -> Iterator[str]:
        """Abre uma execução do MLflow como gerenciador de contexto.

        Args:
            run_name: Nome legível da execução.
            tags: Tags opcionais anexadas à execução.

        Yields:
            O identificador da execução ativa.
        """
        with mlflow.start_run(run_name=run_name, tags=dict(tags or {})) as active:
            yield active.info.run_id

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Registra os hiperparâmetros da execução ativa.

        Args:
            params: Mapa de nome do parâmetro para o valor.
        """
        mlflow.log_params(dict(params))

    def log_metrics(self, metrics: Mapping[str, float], prefix: str = "") -> None:
        """Registra as métricas escalares da execução ativa.

        Args:
            metrics: Mapa de nome da métrica para o valor.
            prefix: Prefixo opcional, por exemplo ``"test"`` ou ``"validation"``.
        """
        mlflow.log_metrics(
            {f"{prefix}{name}": float(value) for name, value in metrics.items()}
        )

    def log_curves(self, curves: Mapping[str, Sequence[float]]) -> None:
        """Registra as curvas de aprendizado como métricas por época.

        Args:
            curves: Mapa de nome da curva para os valores por época.
        """
        for name, values in curves.items():
            for step, value in enumerate(values, start=1):
                mlflow.log_metric(name, float(value), step=step)

    def log_file(self, path: Path) -> None:
        """Anexa um arquivo já existente como artefato da execução.

        Args:
            path: Arquivo a enviar.
        """
        if path.exists():
            mlflow.log_artifact(str(path))

    def log_recommender(
        self, model_path: Path, input_example: pd.DataFrame | None = None
    ) -> str:
        """Registra um recomendador salvo como modelo ``pyfunc`` do MLflow.

        Args:
            model_path: Recomendador serializado a empacotar com o wrapper.
            input_example: Amostra usada para inferir a assinatura do modelo.

        Returns:
            URI do modelo registrado dentro da execução ativa.
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
        """Busca o experimento, criando-o no primeiro uso.

        Args:
            name: Nome do experimento.
            artifact_location: Raiz de artefatos usada na criação.

        Returns:
            A entidade do experimento.
        """
        existing = self._client.get_experiment_by_name(name)
        if existing is not None:
            return existing
        experiment_id = self._client.create_experiment(
            name, artifact_location=artifact_location or None
        )
        return self._client.get_experiment(experiment_id)
