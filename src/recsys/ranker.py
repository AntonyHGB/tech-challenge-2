"""Recomendação top-k construída sobre qualquer :class:`RecommenderModel`."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import pandas as pd

from recsys.data.interactions import InteractionData
from recsys.features.builder import scale_frame
from recsys.features.store import FeatureStore
from recsys.models.base import RecommenderModel
from recsys.preprocessing.pipeline import PreprocessingPipeline


@dataclass(frozen=True)
class Recommendation:
    """Uma sugestão do ranking.

    Attributes:
        item_id: Identificador bruto do item recomendado.
        score: Probabilidade de relevância prevista.
    """

    item_id: int
    score: float


class TopKRanker:
    """Ordena o catálogo para um usuário, delegando a pontuação ao modelo.

    O ranqueamento fica de propósito fora de :class:`RecommenderModel`: os
    modelos apenas pontuam interações, enquanto a geração de candidatos, a
    montagem das features e a ordenação vivem aqui (Responsabilidade Única).
    """

    def __init__(
        self,
        model: RecommenderModel,
        store: FeatureStore,
        pipeline: PreprocessingPipeline,
    ) -> None:
        """Liga o ranqueador a um modelo treinado e aos artefatos de features.

        Args:
            model: Recomendador treinado, usado para pontuar candidatos.
            store: Artefatos de features (estatísticas e vocabulários).
            pipeline: Pipeline de pré-processamento já ajustado.
        """
        self._model = model
        self._store = store
        self._pipeline = pipeline
        self._user_encoder = store.user_encoder()
        self._item_encoder = store.item_encoder()

    def recommend(
        self, user_id: int, top_k: int = 10, exclude: Collection[int] = ()
    ) -> list[Recommendation]:
        """Devolve os itens de maior score para um usuário.

        Args:
            user_id: Identificador bruto do usuário, que precisa ser conhecido.
            top_k: Quantidade de itens a devolver.
            exclude: Itens a pular, tipicamente os já vistos pelo usuário.

        Returns:
            Recomendações ordenadas do mais para o menos relevante.

        Raises:
            KeyError: Se o usuário não tiver sido visto no treino.
        """
        if not self._user_encoder.contains(user_id):
            raise KeyError(f"Usuário '{user_id}' desconhecido.")
        candidates = self._candidate_frame(user_id, exclude)
        if candidates.empty:
            return []
        scores = self._model.predict_proba(
            InteractionData.from_frame(candidates, self._store.feature_columns)
        )
        ranked = candidates.assign(score=scores).nlargest(top_k, "score")
        return [
            Recommendation(item_id=int(row.item_id), score=float(row.score))
            for row in ranked.itertuples()
        ]

    def _candidate_frame(self, user_id: int, exclude: Collection[int]) -> pd.DataFrame:
        """Monta o frame de features de cada item candidato.

        Args:
            user_id: Identificador bruto do usuário.
            exclude: Itens deixados de fora do conjunto de candidatos.

        Returns:
            Frame com índices codificados, features escalonadas e rótulo neutro.
        """
        blocked = set(exclude)
        items = [item for item in self._store.item_classes if item not in blocked]
        frame = pd.DataFrame({"user_id": user_id, "item_id": items})
        frame = self._store.statistics.attach(frame)
        frame["label"] = 0.0
        frame["user_index"] = self._user_encoder.transform(frame["user_id"])
        frame["item_index"] = self._item_encoder.transform(frame["item_id"])
        return scale_frame(self._pipeline, frame)
