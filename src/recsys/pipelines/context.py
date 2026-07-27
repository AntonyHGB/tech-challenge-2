"""Caminhos e configuração compartilhados por todos os stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from recsys.config import Settings, get_settings
from recsys.pipelines.params import Params, load_params
from recsys.seeding import set_global_seed
from recsys.tracking.tracker import ExperimentTracker


@dataclass(frozen=True)
class ArtifactLayout:
    """Resolve os caminhos que cada stage lê e escreve.

    Centralizar o layout mantém os stages livres de caminhos fixos e permite
    mudar os diretórios apenas pela configuração.
    """

    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path

    @property
    def raw_interactions(self) -> Path:
        """Dataset bruto versionado pelo DVC."""
        return self.raw_dir / "ratings.csv"

    @property
    def interactions(self) -> Path:
        """Interações limpas produzidas pelo stage de preprocess."""
        return self.processed_dir / "interactions.parquet"

    @property
    def feature_store(self) -> Path:
        """Artefatos de features produzidos pelo stage de feature_eng."""
        return self.processed_dir / "feature_store.json"

    @property
    def metrics(self) -> Path:
        """Métricas de comparação acompanhadas pelo DVC."""
        return self.reports_dir / "metrics.json"

    @property
    def comparison(self) -> Path:
        """Tabela de comparação legível por pessoas."""
        return self.reports_dir / "comparison.md"

    @property
    def registry(self) -> Path:
        """Registro da versão promovida a Production."""
        return self.reports_dir / "model_registry.json"

    def split(self, name: str) -> Path:
        """Caminho do parquet de um split já preparado.

        Args:
            name: Nome do split (``train``, ``validation`` ou ``test``).

        Returns:
            Caminho do arquivo.
        """
        return self.processed_dir / f"{name}.parquet"

    def model(self, model_name: str) -> Path:
        """Caminho do artefato de um modelo treinado.

        Args:
            model_name: Chave do modelo registrado.

        Returns:
            Caminho do arquivo.
        """
        return self.models_dir / model_name / "model.joblib"

    def train_report(self, model_name: str) -> Path:
        """Caminho do relatório de treino de um modelo.

        Args:
            model_name: Chave do modelo registrado.

        Returns:
            Caminho do arquivo.
        """
        return self.reports_dir / "train" / f"{model_name}.json"


@dataclass(frozen=True)
class StageContext:
    """Configuração, parâmetros e caminhos de que um stage precisa.

    Attributes:
        settings: Configurações da aplicação vindas do ambiente.
        params: Hiperparâmetros validados do pipeline.
        layout: Caminhos de artefatos resolvidos.
    """

    settings: Settings
    params: Params
    layout: ArtifactLayout

    @classmethod
    def load(cls) -> StageContext:
        """Carrega a configuração e semeia todos os geradores aleatórios.

        Returns:
            O contexto do stage, pronto para uso.
        """
        settings = get_settings()
        params = load_params(settings.params_file)
        set_global_seed(params.seed)
        layout = ArtifactLayout(
            raw_dir=settings.data_raw_dir,
            processed_dir=settings.data_processed_dir,
            models_dir=settings.models_dir,
            reports_dir=settings.reports_dir,
        )
        return cls(settings=settings, params=params, layout=layout)

    def tracker(self) -> ExperimentTracker:
        """Cria o tracker do MLflow para o backend configurado.

        Returns:
            Um tracker ligado ao experimento do projeto.
        """
        return ExperimentTracker(
            tracking_uri=self.settings.mlflow_tracking_uri,
            experiment_name=self.settings.mlflow_experiment_name,
            artifact_location=self.settings.mlflow_artifact_location,
        )
