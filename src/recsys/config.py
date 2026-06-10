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
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="URI of the MLflow tracking server.",
    )
    mlflow_experiment_name: str = Field(
        default="ecommerce-recsys",
        description="MLflow experiment grouping every tracked run.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the cached, process-wide application settings.

    Returns:
        The singleton :class:`Settings` instance loaded from the environment.
    """
    return Settings()
