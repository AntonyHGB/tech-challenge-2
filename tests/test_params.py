"""Testes do arquivo tipado de hiperparâmetros e do seu uso no pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from recsys.features.store import FeatureStore, InteractionStatistics
from recsys.pipelines.params import Params, load_params, model_kwargs

PARAMS_FILE = Path("configs/params.yaml")


def _store() -> FeatureStore:
    """Cria um feature store mínimo para os testes de parâmetros.

    Returns:
        Um store com dois usuários e três itens.
    """
    statistics = InteractionStatistics(
        user_activity={},
        item_popularity={},
        user_mean_rating={},
        item_mean_rating={},
        global_mean_rating=3.5,
    )
    return FeatureStore(
        statistics=statistics, user_classes=[1, 2], item_classes=[10, 11, 12]
    )


def test_arquivo_de_parametros_do_projeto_e_valido():
    params = load_params(PARAMS_FILE)
    assert params.seed == 42
    assert params.features.scaler in {"standard", "minmax"}
    assert params.evaluation.top_k >= 1


def test_arquivo_de_parametros_ausente_levanta_erro():
    with pytest.raises(FileNotFoundError):
        load_params(Path("configs/nao-existe.yaml"))


def test_parametros_desconhecidos_sao_rejeitados():
    with pytest.raises(ValidationError):
        Params.model_validate({"inesperado": 1})


def test_fracoes_fora_do_intervalo_sao_rejeitadas():
    with pytest.raises(ValidationError):
        Params.model_validate({"data": {"test_fraction": 1.5}})


def test_visao_achatada_prefixa_cada_secao():
    flat = Params().flat()
    assert flat["seed"] == 42
    assert flat["training.epochs"] == 30
    assert flat["evaluation.primary_metric"] == "roc_auc"


def test_kwargs_levam_o_vocabulario_para_a_rede():
    kwargs = model_kwargs("mlp", Params(), _store())
    assert kwargs["n_users"] == 2
    assert kwargs["n_items"] == 3
    assert kwargs["seed"] == 42


def test_kwargs_rejeitam_modelos_desconhecidos():
    with pytest.raises(KeyError, match="Sem parâmetros"):
        model_kwargs("desconhecido", Params(), _store())
