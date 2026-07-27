"""Empacotamento ``pyfunc`` que torna qualquer recomendador implantável."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from recsys.data.interactions import LABEL_COLUMN, InteractionData
from recsys.models.persistence import load_model

PIP_REQUIREMENTS: list[str] = ["torch", "scikit-learn", "numpy", "pandas", "joblib"]
EXAMPLE_ROWS = 5


class RecommenderPyfunc(mlflow.pyfunc.PythonModel):
    """Serve um recomendador salvo pela interface genérica do MLflow.

    O empacotamento mantém o registry independente de framework: a rede em
    PyTorch e os baselines do Scikit-Learn são registrados, versionados e
    promovidos exatamente do mesmo jeito.
    """

    def load_context(self, context: Any) -> None:
        """Carrega o recomendador serializado junto ao modelo.

        Args:
            context: Contexto do MLflow, que expõe os artefatos registrados.
        """
        self._model = load_model(Path(context.artifacts["model"]))

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Pontua um lote de interações candidatas.

        Args:
            context: Contexto do MLflow (não usado; o modelo já está em memória).
            model_input: Frame com índices codificados e features escalonadas.
            params: Parâmetros de inferência, não utilizados.

        Returns:
            Probabilidade de relevância de cada linha.
        """
        return self._model.predict_proba(InteractionData.from_frame(model_input))


def build_input_example(
    data: InteractionData, feature_columns: Sequence[str]
) -> pd.DataFrame:
    """Monta o exemplo de entrada com que o MLflow infere a assinatura.

    Args:
        data: Interações de onde tirar o exemplo.
        feature_columns: Nomes das colunas de features comportamentais.

    Returns:
        Algumas linhas no mesmo formato da carga usada em produção.
    """
    rows = min(EXAMPLE_ROWS, len(data))
    example = pd.DataFrame(
        {
            "user_index": data.user_indices[:rows],
            "item_index": data.item_indices[:rows],
            LABEL_COLUMN: data.labels[:rows],
        }
    )
    for position, column in enumerate(feature_columns):
        example[column] = data.features[:rows, position]
    return example
