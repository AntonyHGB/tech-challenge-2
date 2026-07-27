"""Configurações tipadas, carregadas do ambiente via Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuração validada, lida das variáveis de ambiente e do ``.env``.

    Centraliza tudo o que é externalizado, para que o restante do código leia a
    configuração de um único objeto validado em vez de acessar ``os.environ``.
    A semente fixa em :attr:`random_seed` mantém os experimentos reprodutíveis.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    random_seed: int = Field(
        default=42, description="Semente global de todo componente estocástico."
    )
    data_raw_dir: Path = Field(
        default=Path("data/raw"), description="Diretório do dataset bruto."
    )
    data_processed_dir: Path = Field(
        default=Path("data/processed"),
        description="Diretório dos dados processados.",
    )
    models_dir: Path = Field(
        default=Path("models"), description="Diretório dos modelos treinados."
    )
    reports_dir: Path = Field(
        default=Path("reports"),
        description="Diretório das métricas e relatórios de comparação.",
    )
    params_file: Path = Field(
        default=Path("configs/params.yaml"),
        description="Arquivo de hiperparâmetros consumido pelo pipeline DVC.",
    )
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        description=(
            "Backend de tracking do MLflow. O padrão é um SQLite local, "
            "necessário para o Model Registry funcionar sem servidor; o "
            "docker compose aponta para o serviço MLflow."
        ),
    )
    mlflow_experiment_name: str = Field(
        default="ecommerce-recsys",
        description="Experimento que agrupa todas as execuções rastreadas.",
    )
    mlflow_artifact_location: str = Field(
        default="./mlartifacts",
        description="Raiz de artefatos usada ao criar o experimento.",
    )
    mlflow_registered_model_name: str = Field(
        default="ecommerce-recsys-recommender",
        description="Nome do modelo registrado e promovido no MLflow.",
    )


@lru_cache
def get_settings() -> Settings:
    """Retorna as configurações da aplicação, em cache por processo.

    Returns:
        A instância única de :class:`Settings` carregada do ambiente.
    """
    return Settings()
