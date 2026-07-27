"""Testes da fábrica de modelos e do registro padrão."""

from __future__ import annotations

import pytest

from recsys.models import (
    BASELINE_MODELS,
    NEURAL_MODEL,
    ModelFactory,
    build_default_factory,
)
from recsys.models.baseline import LogisticRecommender, PopularityRecommender
from recsys.models.mlp import MLPRecommender


def test_fabrica_padrao_lista_os_modelos_registrados():
    assert build_default_factory().available() == ["logistic", "mlp", "popularity"]


def test_registro_expoe_a_rede_neural_e_seus_baselines():
    assert NEURAL_MODEL == "mlp"
    assert set(BASELINE_MODELS) == {"popularity", "logistic"}


def test_fabrica_cria_o_modelo_pedido():
    factory = build_default_factory()
    assert isinstance(factory.create("mlp", n_users=5, n_items=3), MLPRecommender)
    assert isinstance(factory.create("popularity"), PopularityRecommender)
    assert isinstance(factory.create("logistic"), LogisticRecommender)


def test_fabrica_repassa_os_argumentos_nomeados():
    model = build_default_factory().create(
        "mlp", n_users=5, n_items=3, embedding_dim=16
    )
    assert model.embedding_dim == 16


def test_modelo_desconhecido_levanta_key_error():
    with pytest.raises(KeyError, match="desconhecido"):
        build_default_factory().create("nao-existe")


def test_registro_duplicado_levanta_value_error():
    factory = ModelFactory()
    factory.register("dummy", MLPRecommender)
    with pytest.raises(ValueError, match="já está registrado"):
        factory.register("dummy", MLPRecommender)
