"""Early stopping helper for the PyTorch training loop."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class EarlyStopping:
    """Stop training once the validation loss stops improving.

    Keeps a copy of the best weights seen so far so the fitted model is the one
    that generalised best, not merely the last epoch executed.
    """

    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        """Configure the stopping criterion.

        Args:
            patience: Number of epochs without improvement tolerated before
                stopping. A non-positive value disables early stopping.
            min_delta: Minimum loss decrease that counts as an improvement.
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
        """Register an epoch result and report whether training should stop.

        Args:
            epoch: One-based epoch number.
            loss: Validation loss observed at ``epoch``.
            state: Model ``state_dict`` of that epoch.

        Returns:
            ``True`` when patience is exhausted and training should stop.
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
        """Load the best observed weights back into ``network``.

        Args:
            network: Network to restore; left untouched if no epoch improved.
        """
        if self._best_state:
            network.load_state_dict(self._best_state)

    @property
    def best_loss(self) -> float:
        """Lowest validation loss observed.

        Returns:
            The best loss, or infinity when no epoch ran.
        """
        return self._best_loss

    @property
    def best_epoch(self) -> int:
        """Epoch that produced the best validation loss.

        Returns:
            One-based epoch number, or ``0`` when no epoch improved.
        """
        return self._best_epoch
