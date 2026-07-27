"""Early stopping para o laço de treino em PyTorch."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class EarlyStopping:
    """Interrompe o treino quando a perda de validação para de melhorar.

    Guarda uma cópia dos melhores pesos vistos, para que o modelo final seja o
    que melhor generalizou, e não simplesmente o da última época executada.
    """

    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        """Configura o critério de parada.

        Args:
            patience: Épocas sem melhora toleradas antes de parar. Um valor
                menor ou igual a zero desliga o early stopping.
            min_delta: Redução mínima da perda que conta como melhora.
        """
        self._patience = patience
        self._min_delta = min_delta
        self._best_loss = float("inf")
        self._best_epoch = 0
        self._epochs_without_improvement = 0
        self._best_state: dict[str, torch.Tensor] = {}

    def update(
        self, epoch: int, loss: float, state: Mapping[str, torch.Tensor]
    ) -> bool:
        """Registra o resultado da época e diz se o treino deve parar.

        Args:
            epoch: Número da época, começando em um.
            loss: Perda de validação observada na época.
            state: ``state_dict`` do modelo naquela época.

        Returns:
            ``True`` quando a paciência se esgotou e o treino deve parar.
        """
        if loss < self._best_loss - self._min_delta:
            self._best_loss = loss
            self._best_epoch = epoch
            self._epochs_without_improvement = 0
            self._best_state = {
                key: value.detach().clone() for key, value in state.items()
            }
            return False
        self._epochs_without_improvement += 1
        return self._patience > 0 and self._epochs_without_improvement >= self._patience

    def restore(self, network: nn.Module) -> None:
        """Recarrega os melhores pesos observados, se houver algum.

        Args:
            network: Rede a restaurar.
        """
        if self._best_state:
            network.load_state_dict(self._best_state)

    @property
    def best_loss(self) -> float:
        """Menor perda de validação observada."""
        return self._best_loss

    @property
    def best_epoch(self) -> int:
        """Época que produziu a menor perda de validação."""
        return self._best_epoch
