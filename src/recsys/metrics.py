"""Métricas de classificação e de ranking usadas na comparação de modelos."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from recsys.data.interactions import InteractionData
from recsys.models.base import RecommenderModel

DECISION_THRESHOLD = 0.5


def evaluate_model(
    model: RecommenderModel,
    data: InteractionData,
    top_k: int = 10,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, float]:
    """Avalia um modelo treinado com métricas de classificação e de ranking.

    Args:
        model: Recomendador já treinado.
        data: Interações sobre as quais avaliar.
        top_k: Corte usado pelas métricas de ranking.
        threshold: Probabilidade a partir da qual a interação é relevante.

    Returns:
        Mapa de nome da métrica para o valor.
    """
    scores = model.predict_proba(data)
    metrics = classification_metrics(data.labels, scores, threshold)
    metrics.update(ranking_metrics(data.user_indices, data.labels, scores, top_k))
    metrics["n_interactions"] = float(len(data))
    return metrics


def classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float = DECISION_THRESHOLD
) -> dict[str, float]:
    """Avalia as previsões como um problema de classificação binária.

    Args:
        labels: Relevância verdadeira (``0``/``1``).
        scores: Probabilidades previstas.
        threshold: Probabilidade a partir da qual a interação é relevante.

    Returns:
        Mapa com acurácia, precisão, recall, F1, ROC AUC e log loss.
    """
    predictions = (scores >= threshold).astype(int)
    truth = labels.astype(int)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": _roc_auc(truth, scores),
        "log_loss": float(
            log_loss(truth, np.clip(scores, 1e-7, 1 - 1e-7), labels=[0, 1])
        ),
    }


def ranking_metrics(
    user_indices: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    """Média, por usuário, da qualidade do ranking das interações pontuadas.

    As interações de holdout de cada usuário são reordenadas pelo score
    previsto e comparadas com os itens que ele de fato achou relevantes.
    Usuários sem nenhum item relevante ficam de fora, já que qualquer
    ordenação zeraria as métricas.

    Args:
        user_indices: Índice codificado do usuário de cada linha.
        labels: Relevância verdadeira de cada linha.
        scores: Score previsto de cada linha.
        top_k: Corte aplicado ao ranking de cada usuário.

    Returns:
        Mapa com ``precision_at_k``, ``recall_at_k``, ``ndcg_at_k`` e a
        quantidade de usuários usados na média.
    """
    frame = pd.DataFrame(
        {"user": user_indices, "label": labels.astype(float), "score": scores}
    )
    per_user = [
        _user_metrics(group, top_k)
        for _, group in frame.groupby("user", sort=False)
        if group["label"].sum() > 0
    ]
    if not per_user:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "evaluated_users": 0.0,
        }
    summary = pd.DataFrame(per_user).mean()
    return {
        "precision_at_k": float(summary["precision"]),
        "recall_at_k": float(summary["recall"]),
        "ndcg_at_k": float(summary["ndcg"]),
        "evaluated_users": float(len(per_user)),
    }


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Calcula o ROC AUC, tolerando entradas com uma única classe.

    Args:
        labels: Relevância verdadeira.
        scores: Probabilidades previstas.

    Returns:
        O ROC AUC, ou ``0.5`` quando só existe uma classe.
    """
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def _user_metrics(group: pd.DataFrame, top_k: int) -> dict[str, float]:
    """Calcula as métricas de ranking de um único usuário.

    Args:
        group: Linhas de um usuário, com ``label`` e ``score``.
        top_k: Corte do ranking.

    Returns:
        Mapa com a precisão, o recall e o NDCG daquele usuário.
    """
    relevance = group.sort_values("score", ascending=False)["label"].to_numpy()
    cut = relevance[:top_k]
    relevant_total = float(relevance.sum())
    return {
        "precision": float(cut.sum() / max(len(cut), 1)),
        "recall": float(cut.sum() / relevant_total),
        "ndcg": _ndcg(cut, relevant_total, top_k),
    }


def _ndcg(cut: np.ndarray, relevant_total: float, top_k: int) -> float:
    """Calcula o ganho cumulativo descontado normalizado de um ranking.

    Args:
        cut: Relevância binária dos ``top_k`` itens melhor colocados.
        relevant_total: Quantidade de itens relevantes disponíveis ao usuário.
        top_k: Corte do ranking.

    Returns:
        O NDCG, em ``[0, 1]``.
    """
    discounts = 1.0 / np.log2(np.arange(2, len(cut) + 2))
    gain = float((cut * discounts).sum())
    ideal = float(discounts[: int(min(relevant_total, top_k))].sum())
    return gain / ideal if ideal > 0 else 0.0
