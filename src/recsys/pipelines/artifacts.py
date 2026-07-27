"""Filesystem layout shared by every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from recsys.config import Settings


@dataclass(frozen=True)
class ArtifactLayout:
    """Resolve the paths each stage reads from and writes to.

    Centralising the layout keeps the stages free of hard-coded paths and lets
    the directories be relocated through configuration alone.
    """

    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path

    @classmethod
    def from_settings(cls, settings: Settings) -> ArtifactLayout:
        """Build the layout from the typed application settings.

        Args:
            settings: Loaded application settings.

        Returns:
            The resolved layout.
        """
        return cls(
            raw_dir=settings.data_raw_dir,
            processed_dir=settings.data_processed_dir,
            models_dir=settings.models_dir,
            reports_dir=settings.reports_dir,
        )

    @property
    def raw_interactions(self) -> Path:
        """Raw dataset tracked by DVC.

        Returns:
            Path of ``ratings.csv``.
        """
        return self.raw_dir / "ratings.csv"

    @property
    def interactions(self) -> Path:
        """Cleaned interactions produced by the preprocess stage.

        Returns:
            Path of the interactions parquet file.
        """
        return self.processed_dir / "interactions.parquet"

    @property
    def feature_store(self) -> Path:
        """Feature artefacts produced by the feature engineering stage.

        Returns:
            Path of the feature store JSON file.
        """
        return self.processed_dir / "feature_store.json"

    @property
    def preprocessor(self) -> Path:
        """Fitted preprocessing pipeline.

        Returns:
            Path of the serialised pipeline.
        """
        return self.processed_dir / "preprocessor.joblib"

    @property
    def metrics(self) -> Path:
        """Comparison metrics tracked by DVC.

        Returns:
            Path of the metrics JSON file.
        """
        return self.reports_dir / "metrics.json"

    @property
    def comparison(self) -> Path:
        """Human-readable model comparison table.

        Returns:
            Path of the comparison markdown file.
        """
        return self.reports_dir / "comparison.md"

    @property
    def recommendations(self) -> Path:
        """Sample top-k recommendations of the best model.

        Returns:
            Path of the recommendations JSON file.
        """
        return self.reports_dir / "sample_recommendations.json"

    @property
    def registry(self) -> Path:
        """Record of the model version promoted to Production.

        Returns:
            Path of the registry JSON file.
        """
        return self.reports_dir / "model_registry.json"

    def split(self, name: str) -> Path:
        """Path of an engineered split.

        Args:
            name: Split name (``train``, ``validation`` or ``test``).

        Returns:
            Path of the split parquet file.
        """
        return self.processed_dir / f"{name}.parquet"

    def model(self, model_name: str) -> Path:
        """Path of a trained model artifact.

        Args:
            model_name: Registered model key.

        Returns:
            Path of the serialised model.
        """
        return self.models_dir / model_name / "model.joblib"

    def train_report(self, model_name: str) -> Path:
        """Path of a training report.

        Args:
            model_name: Registered model key.

        Returns:
            Path of the training report JSON file.
        """
        return self.reports_dir / "train" / f"{model_name}.json"
