"""Typed application settings loaded from the environment via Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed configuration sourced from environment variables / ``.env``.

    Centralises every externalised setting so the rest of the codebase reads
    configuration from one validated object instead of touching ``os.environ``
    directly. Values are coerced and validated by Pydantic on load, and a fixed
    :attr:`random_seed` keeps experiments reproducible.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    random_seed: int = Field(
        default=42, description="Global seed shared by every stochastic component."
    )
    data_raw_dir: Path = Field(
        default=Path("data/raw"), description="Directory holding the raw dataset."
    )
    data_processed_dir: Path = Field(
        default=Path("data/processed"),
        description="Directory holding the processed dataset.",
    )
    models_dir: Path = Field(
        default=Path("models"), description="Directory where trained models land."
    )
    reports_dir: Path = Field(
        default=Path("reports"),
        description="Directory holding metrics and comparison reports.",
    )
    params_file: Path = Field(
        default=Path("configs/params.yaml"),
        description="Hyper-parameter file consumed by the DVC pipeline.",
    )
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        description=(
            "MLflow tracking backend. Defaults to a local SQLite store so the "
            "Model Registry works without a server; docker compose points it "
            "at the MLflow service instead."
        ),
    )
    mlflow_experiment_name: str = Field(
        default="ecommerce-recsys",
        description="MLflow experiment grouping every tracked run.",
    )
    mlflow_artifact_location: str = Field(
        default="./mlartifacts",
        description="Artifact root used when the experiment is first created.",
    )
    mlflow_registered_model_name: str = Field(
        default="ecommerce-recsys-recommender",
        description="Name of the model registered and promoted in MLflow.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached, process-wide application settings.

    Returns:
        The singleton :class:`Settings` instance loaded from the environment.
    """
    return Settings()
