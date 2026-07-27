"""Testes da camada de engenharia de features."""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.data.encoders import IdEncoder
from recsys.data.interactions import FEATURE_COLUMNS, InteractionData
from recsys.features.builder import add_binary_label, fit_scale_frame, positive_rate
from recsys.features.store import FeatureStore, InteractionStatistics
from recsys.preprocessing import build_pipeline


def test_estatisticas_contam_interacoes_por_entidade(interaction_frame):
    statistics = InteractionStatistics.from_frame(interaction_frame)
    assert statistics.user_activity[1] == 4
    assert statistics.item_popularity[10] == 3


def test_attach_adiciona_todas_as_colunas_de_feature(interaction_frame):
    statistics = InteractionStatistics.from_frame(interaction_frame)
    enriched = statistics.attach(interaction_frame)
    assert set(FEATURE_COLUMNS).issubset(enriched.columns)


def test_entidades_desconhecidas_usam_valores_neutros(interaction_frame):
    statistics = InteractionStatistics.from_frame(interaction_frame)
    unseen = pd.DataFrame({"user_id": [999], "item_id": [999], "rating": [5.0]})
    row = statistics.attach(unseen).iloc[0]
    assert row["user_activity"] == 0.0
    assert row["item_popularity"] == 0.0
    assert row["item_mean_rating"] == statistics.global_mean_rating


def test_estatisticas_sobrevivem_ao_json(interaction_frame):
    statistics = InteractionStatistics.from_frame(interaction_frame)
    restored = InteractionStatistics.model_validate_json(statistics.model_dump_json())
    assert restored == statistics


def test_rotulo_usa_o_limiar_configurado(interaction_frame):
    labelled = add_binary_label(interaction_frame, positive_threshold=4.0)
    assert labelled["label"].tolist() == [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
    assert positive_rate(labelled) == pytest.approx(0.6)


def test_feature_store_sobrevive_ao_json(tmp_path, interaction_frame):
    store = FeatureStore(
        statistics=InteractionStatistics.from_frame(interaction_frame),
        user_classes=[1, 2, 3],
        item_classes=[10, 11, 12, 13],
        positive_threshold=4.0,
    )
    restored = FeatureStore.load(store.save(tmp_path / "store.json"))
    assert restored.n_users == 3
    assert restored.n_items == 4
    assert restored.item_encoder().transform([12]).tolist() == [2]


def test_escala_padroniza_as_colunas_de_feature(interaction_frame):
    statistics = InteractionStatistics.from_frame(interaction_frame)
    enriched = add_binary_label(statistics.attach(interaction_frame), 4.0)
    pipeline = build_pipeline("standard", FEATURE_COLUMNS)
    scaled = fit_scale_frame(pipeline, enriched)
    assert scaled.shape == enriched.shape
    assert scaled["user_activity"].mean() == pytest.approx(0.0, abs=1e-9)
    assert scaled["item_popularity"].std(ddof=0) == pytest.approx(1.0)


def test_interaction_data_exige_as_colunas_de_feature():
    frame = pd.DataFrame({"user_index": [0], "item_index": [0], "label": [1.0]})
    with pytest.raises(KeyError, match="Colunas ausentes"):
        InteractionData.from_frame(frame)


def test_encoder_mapeia_identificadores_para_indices():
    encoder = IdEncoder.from_values([30, 10, 20, 10])
    assert encoder.classes == [10, 20, 30]
    assert encoder.transform([30, 10]).tolist() == [2, 0]
    assert len(encoder) == 3


def test_encoder_rejeita_identificador_desconhecido():
    encoder = IdEncoder.from_values([1, 2])
    assert not encoder.contains(99)
    with pytest.raises(KeyError, match="desconhecido"):
        encoder.transform([99])
