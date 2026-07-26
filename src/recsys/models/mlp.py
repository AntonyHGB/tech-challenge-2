"""Embedding-based MLP recommender implemented with PyTorch."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from recsys.data.interactions import InteractionData
from recsys.models.base import RecommenderModel
from recsys.models.early_stopping import EarlyStopping
from recsys.seeding import set_global_seed


class RecommenderNetwork(nn.Module):
    """Two-tower style network: user/item embeddings plus behavioural features."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_features: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        """Build the embedding tables and the dense tower.

        Args:
            n_users: Size of the user vocabulary.
            n_items: Size of the item vocabulary.
            n_features: Number of behavioural features per interaction.
            embedding_dim: Width of the user and item embeddings.
            hidden_dim: Width of the first hidden layer.
            dropout: Dropout probability applied after each hidden layer.
        """
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.tower = nn.Sequential(
            nn.Linear(2 * embedding_dim + n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 1), 1),
        )

    def forward(
        self, users: torch.Tensor, items: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """Score a batch of interactions.

        Args:
            users: Encoded user indices.
            items: Encoded item indices.
            features: Behavioural features.

        Returns:
            Unnormalised logits, one per interaction.
        """
        merged = torch.cat(
            [self.user_embedding(users), self.item_embedding(items), features], dim=1
        )
        return self.tower(merged).squeeze(-1)


class MLPRecommender(RecommenderModel):
    """Neural recommender trained with binary cross-entropy and early stopping."""

    name: ClassVar[str] = "mlp"

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        early_stopping_patience: int = 5,
        seed: int = 42,
    ) -> None:
        """Store the architecture and optimisation hyper-parameters.

        Args:
            n_users: Size of the user vocabulary.
            n_items: Size of the item vocabulary.
            embedding_dim: Width of the user and item embeddings.
            hidden_dim: Width of the first hidden layer.
            dropout: Dropout probability applied after each hidden layer.
            epochs: Maximum number of training epochs.
            batch_size: Mini-batch size.
            learning_rate: Adam learning rate.
            weight_decay: L2 penalty applied by Adam.
            early_stopping_patience: Epochs without validation improvement
                tolerated before stopping.
            seed: Seed applied before the weights are initialised.
        """
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self._network: RecommenderNetwork | None = None
        self._criterion = nn.BCEWithLogitsLoss()
        self._history: dict[str, list[float]] = {
            "train_loss": [],
            "validation_loss": [],
        }
        self._best_epoch = 0

    def fit(
        self, data: InteractionData, validation: InteractionData | None = None
    ) -> None:
        """Train the network, early stopping on the validation loss.

        Args:
            data: Training interactions.
            validation: Interactions monitored for early stopping. Falls back to
                the training loss when omitted.
        """
        set_global_seed(self.seed)
        self._network = self._build_network(data.n_features)
        optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        stopper = EarlyStopping(self.early_stopping_patience)
        loader = self._build_loader(data)
        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(loader, optimizer)
            monitored = self._monitored_loss(validation, train_loss)
            self._record(train_loss, monitored)
            if stopper.update(epoch, monitored, self._network.state_dict()):
                break
        stopper.restore(self._network)
        self._best_epoch = stopper.best_epoch

    def predict_proba(self, data: InteractionData) -> np.ndarray:
        """Score interactions with the trained network.

        Args:
            data: Interactions to score.

        Returns:
            Relevance probabilities aligned with the input rows.
        """
        network = self._require_network()
        network.eval()
        users, items, features, _ = self._to_tensors(data)
        with torch.no_grad():
            logits = network(users, items, features)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    def hyperparameters(self) -> dict[str, Any]:
        """Report the configuration MLflow logs for this run.

        Returns:
            Mapping of hyper-parameter name to value.
        """
        return {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "early_stopping_patience": self.early_stopping_patience,
            "n_users": self.n_users,
            "n_items": self.n_items,
        }

    def training_history(self) -> dict[str, list[float]]:
        """Return the per-epoch learning curves.

        Returns:
            Mapping with the ``train_loss`` and ``validation_loss`` curves.
        """
        return {key: list(values) for key, values in self._history.items()}

    @property
    def best_epoch(self) -> int:
        """Epoch whose weights were restored by early stopping.

        Returns:
            One-based epoch number, or ``0`` before training.
        """
        return self._best_epoch

    @property
    def epochs_run(self) -> int:
        """Number of epochs actually executed.

        Returns:
            Length of the training curve.
        """
        return len(self._history["train_loss"])

    def _build_network(self, n_features: int) -> RecommenderNetwork:
        """Instantiate the network for the given feature width.

        Args:
            n_features: Number of behavioural features per interaction.

        Returns:
            A freshly initialised network.
        """
        return RecommenderNetwork(
            n_users=self.n_users,
            n_items=self.n_items,
            n_features=n_features,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

    def _build_loader(self, data: InteractionData) -> DataLoader:
        """Wrap the training arrays into a shuffled, reproducible loader.

        Args:
            data: Training interactions.

        Returns:
            A ``DataLoader`` seeded for reproducibility.
        """
        dataset = TensorDataset(*self._to_tensors(data))
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
        )

    def _train_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> float:
        """Run one optimisation pass over the training data.

        Args:
            loader: Batched training data.
            optimizer: Optimiser updating the weights.

        Returns:
            Sample-weighted mean training loss of the epoch.
        """
        network = self._require_network()
        network.train()
        total, seen = 0.0, 0
        for users, items, features, labels in loader:
            optimizer.zero_grad()
            loss = self._criterion(network(users, items, features), labels)
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * labels.shape[0]
            seen += int(labels.shape[0])
        return total / max(seen, 1)

    def _monitored_loss(
        self, validation: InteractionData | None, fallback: float
    ) -> float:
        """Compute the loss early stopping should monitor.

        Args:
            validation: Optional validation interactions.
            fallback: Loss used when no validation split is available.

        Returns:
            The monitored loss value.
        """
        if validation is None or len(validation) == 0:
            return fallback
        network = self._require_network()
        network.eval()
        users, items, features, labels = self._to_tensors(validation)
        with torch.no_grad():
            loss = self._criterion(network(users, items, features), labels)
        return float(loss)

    def _record(self, train_loss: float, validation_loss: float) -> None:
        """Append an epoch result to the learning curves.

        Args:
            train_loss: Training loss of the epoch.
            validation_loss: Monitored loss of the epoch.
        """
        self._history["train_loss"].append(train_loss)
        self._history["validation_loss"].append(validation_loss)

    @staticmethod
    def _to_tensors(
        data: InteractionData,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert an interaction split into PyTorch tensors.

        Args:
            data: Interactions to convert.

        Returns:
            Tuple of user, item, feature and label tensors.
        """
        return (
            torch.from_numpy(data.user_indices).long(),
            torch.from_numpy(data.item_indices).long(),
            torch.from_numpy(data.features).float(),
            torch.from_numpy(data.labels).float(),
        )

    def _require_network(self) -> RecommenderNetwork:
        """Return the trained network, guarding unfitted usage.

        Returns:
            The trained network.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self._network is None:
            raise RuntimeError("MLPRecommender must be fitted before scoring.")
        return self._network
