"""Typed, validated view of the pipeline hyper-parameter file."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base class rejecting unknown or mutated parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataParams(StrictModel):
    """Cleaning and splitting parameters."""

    min_user_interactions: int = Field(default=5, ge=1)
    min_item_interactions: int = Field(default=5, ge=1)
    validation_fraction: float = Field(default=0.1, gt=0, lt=1)
    test_fraction: float = Field(default=0.1, gt=0, lt=1)


class FeatureParams(StrictModel):
    """Feature engineering parameters."""

    positive_threshold: float = Field(default=4.0, gt=0)
    scaler: str = Field(default="standard")


class ModelParams(StrictModel):
    """Architecture of the neural recommender."""

    embedding_dim: int = Field(default=32, ge=2)
    hidden_dim: int = Field(default=64, ge=2)
    dropout: float = Field(default=0.2, ge=0, lt=1)


class TrainingParams(StrictModel):
    """Optimisation parameters of the neural recommender."""

    epochs: int = Field(default=30, ge=1)
    batch_size: int = Field(default=512, ge=1)
    learning_rate: float = Field(default=3e-3, gt=0)
    weight_decay: float = Field(default=1e-5, ge=0)
    early_stopping_patience: int = Field(default=3, ge=0)


class BaselineParams(StrictModel):
    """Configuration of the Scikit-Learn baselines."""

    popularity_smoothing: float = Field(default=10.0, ge=0)
    logistic_penalty_strength: float = Field(default=1.0, gt=0)
    logistic_max_iterations: int = Field(default=1000, ge=1)


class EvaluationParams(StrictModel):
    """Comparison and promotion parameters."""

    top_k: int = Field(default=10, ge=1)
    decision_threshold: float = Field(default=0.5, gt=0, lt=1)
    primary_metric: str = Field(default="roc_auc")
    sample_users: int = Field(default=3, ge=0)


class Params(StrictModel):
    """Root of the hyper-parameter file."""

    seed: int = Field(default=42)
    data: DataParams = DataParams()
    features: FeatureParams = FeatureParams()
    model: ModelParams = ModelParams()
    training: TrainingParams = TrainingParams()
    baselines: BaselineParams = BaselineParams()
    evaluation: EvaluationParams = EvaluationParams()

    def flat(self) -> dict[str, float | int | str]:
        """Flatten the parameters for MLflow logging.

        Returns:
            Mapping of ``section.name`` to value.
        """
        flattened: dict[str, float | int | str] = {"seed": self.seed}
        for section, values in self.model_dump(exclude={"seed"}).items():
            for name, value in values.items():
                flattened[f"{section}.{name}"] = value
        return flattened


def load_params(path: Path) -> Params:
    """Read and validate the hyper-parameter file.

    Args:
        path: Location of the YAML parameter file.

    Returns:
        The validated parameters.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Parameter file '{path}' not found.")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Params.model_validate(payload)
