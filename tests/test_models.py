"""Testes dos recomendadores e do contrato que compartilham."""

from __future__ import annotations

import numpy as np
import pytest

from recsys.metrics import evaluate_model
from recsys.models.baseline import LogisticRecommender, PopularityRecommender
from recsys.models.early_stopping import EarlyStopping
from recsys.models.mlp import MLPRecommender
from recsys.models.persistence import load_model, save_model


def _mlp() -> MLPRecommender:
    """Cria um recomendador neural pequeno e rápido para os testes.

    Returns:
        Um :class:`MLPRecommender` ainda não treinado.
    """
    return MLPRecommender(
        n_users=20,
        n_items=10,
        embedding_dim=8,
        hidden_dim=16,
        epochs=8,
        batch_size=64,
        early_stopping_patience=2,
        seed=42,
    )


@pytest.mark.parametrize(
    "model",
    [PopularityRecommender(), LogisticRecommender(max_iterations=200)],
)
def test_baselines_produzem_probabilidades(model, interaction_data):
    model.fit(interaction_data)
    scores = model.predict_proba(interaction_data)
    assert scores.shape == (len(interaction_data),)
    assert np.all((scores >= 0) & (scores <= 1))


def test_pontuar_antes_de_treinar_levanta_erro(interaction_data):
    with pytest.raises(RuntimeError, match="fit"):
        LogisticRecommender().predict_proba(interaction_data)
    with pytest.raises(RuntimeError, match="fit"):
        _mlp().predict_proba(interaction_data)


def test_mlp_aprende_um_sinal_separavel(interaction_data):
    model = _mlp()
    model.fit(interaction_data, interaction_data)
    metrics = evaluate_model(model, interaction_data, top_k=5)
    assert metrics["roc_auc"] > 0.8
    assert model.epochs_run >= 1


def test_treino_do_mlp_e_reprodutivel(interaction_data):
    first, second = _mlp(), _mlp()
    first.fit(interaction_data, interaction_data)
    second.fit(interaction_data, interaction_data)
    np.testing.assert_allclose(
        first.predict_proba(interaction_data),
        second.predict_proba(interaction_data),
    )


def test_mlp_registra_as_curvas_de_aprendizado(interaction_data):
    model = _mlp()
    model.fit(interaction_data, interaction_data)
    history = model.training_history()
    assert len(history["train_loss"]) == model.epochs_run
    assert len(history["validation_loss"]) == model.epochs_run


def test_early_stopping_para_apos_a_paciencia():
    stopper = EarlyStopping(patience=2)
    assert stopper.update(1, 1.0, {}) is False
    assert stopper.update(2, 1.5, {}) is False
    assert stopper.update(3, 1.6, {}) is True
    assert stopper.best_epoch == 1
    assert stopper.best_loss == 1.0


def test_early_stopping_desligado_nunca_para():
    stopper = EarlyStopping(patience=0)
    stopper.update(1, 1.0, {})
    assert stopper.update(2, 2.0, {}) is False


def test_persistencia_preserva_as_previsoes(tmp_path, interaction_data):
    model = PopularityRecommender()
    model.fit(interaction_data)
    restored = load_model(save_model(model, tmp_path / "model.joblib"))
    np.testing.assert_allclose(
        model.predict_proba(interaction_data),
        restored.predict_proba(interaction_data),
    )


def test_carregar_modelo_inexistente_levanta_erro(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "ausente.joblib")
