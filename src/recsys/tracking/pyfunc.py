"""MLflow ``pyfunc`` wrapper that makes any recommender deployable."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from recsys.data.interactions import LABEL_COLUMN, InteractionData
from recsys.models.persistence import load_model

PIP_REQUIREMENTS: list[str] = [
    "torch",
    "scikit-learn",
    "numpy",
    "pandas",
    "joblib",
]
EXAMPLE_ROWS = 5


def build_input_example(
    data: InteractionData, feature_columns: Sequence[str]
) -> pd.DataFrame:
    """Build the input example MLflow uses to infer the model signature.

    Args:
        data: Interactions to sample the example from.
        feature_columns: Names of the behavioural feature columns.

    Returns:
        A few rows shaped exactly like the serving payload.
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


class RecommenderPyfunc(mlflow.pyfunc.PythonModel):
    """Serve a persisted recommender through the generic MLflow interface.

    Wrapping the model keeps the registry framework-agnostic: the PyTorch net
    and the Scikit-Learn baselines are logged, versioned and promoted exactly
    the same way.
    """

    def load_context(self, context: Any) -> None:
        """Load the serialised recommender bundled with the model.

        Args:
            context: MLflow context exposing the logged artifacts.
        """
        self._model = load_model(Path(context.artifacts["model"]))

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Score a batch of candidate interactions.

        Args:
            context: MLflow context (unused, the model is already loaded).
            model_input: Frame with encoded indices and scaled features.
            params: Unused inference parameters.

        Returns:
            Relevance probability of each row.
        """
        return self._model.predict_proba(InteractionData.from_frame(model_input))
