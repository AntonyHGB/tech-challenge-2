"""Testes das estratégias de pré-processamento e do pipeline."""

from __future__ import annotations

import pytest

from recsys.preprocessing import (
    MinMaxScaler,
    PreprocessingPipeline,
    StandardScaler,
    build_pipeline,
    build_strategy,
)
from recsys.preprocessing.strategies import available_strategies


def test_min_max_escala_para_intervalo_unitario():
    assert MinMaxScaler().fit_transform([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_min_max_lida_com_coluna_constante():
    assert MinMaxScaler().fit_transform([3.0, 3.0]) == [0.0, 0.0]


def test_standard_scaler_produz_media_zero():
    result = StandardScaler().fit_transform([1.0, 2.0, 3.0])
    assert sum(result) == pytest.approx(0.0)


def test_transform_antes_do_fit_levanta_erro():
    with pytest.raises(RuntimeError):
        MinMaxScaler().transform([1.0])


def test_fit_em_sequencia_vazia_levanta_erro():
    with pytest.raises(ValueError, match="vazia"):
        MinMaxScaler().fit([])


def test_pipeline_aplica_uma_estrategia_por_coluna():
    pipeline = PreprocessingPipeline(
        {"price": MinMaxScaler(), "rating": StandardScaler()}
    )
    out = pipeline.fit_transform({"price": [0.0, 10.0], "rating": [1.0, 3.0]})
    assert out["price"] == [0.0, 1.0]
    assert out["rating"][0] == pytest.approx(-1.0)


def test_pipeline_reaproveita_o_ajuste_do_treino():
    pipeline = PreprocessingPipeline({"price": MinMaxScaler()})
    pipeline.fit({"price": [0.0, 10.0]})
    assert pipeline.transform({"price": [5.0]})["price"] == [0.5]


def test_registro_lista_e_constroi_estrategias():
    assert available_strategies() == ["minmax", "standard"]
    assert isinstance(build_strategy("standard"), StandardScaler)


def test_registro_rejeita_estrategia_desconhecida():
    with pytest.raises(KeyError, match="desconhecida"):
        build_strategy("nao-existe")


def test_pipeline_construido_cobre_todas_as_colunas():
    assert build_pipeline("minmax", ["a", "b"]).columns == ["a", "b"]
