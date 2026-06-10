"""Tests for the typed application settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from recsys.config import Settings, get_settings


def test_defaults_are_applied(monkeypatch):
    monkeypatch.delenv("RANDOM_SEED", raising=False)
    settings = Settings(_env_file=None)
    assert settings.random_seed == 42
    assert settings.mlflow_experiment_name == "ecommerce-recsys"
    assert settings.data_raw_dir == Path("data/raw")


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("RANDOM_SEED", "7")
    monkeypatch.setenv("MODELS_DIR", "artifacts/models")
    settings = Settings(_env_file=None)
    assert settings.random_seed == 7
    assert settings.models_dir == Path("artifacts/models")


def test_settings_are_immutable():
    settings = Settings(_env_file=None)
    with pytest.raises(ValidationError):
        settings.random_seed = 1


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
