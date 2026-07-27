"""Recomendador MLP com embeddings, implementado em PyTorch."""

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
    """Rede com embeddings de usuário e item mais as features de navegação."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_features: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        """Monta as tabelas de embedding e a torre densa.

        Args:
            n_users: Tamanho do vocabulário de usuários.
            n_items: Tamanho do vocabulário de itens.
            n_features: Quantidade de features por interação.
            embedding_dim: Dimensão dos embeddings de usuário e item.
            hidden_dim: Largura da primeira camada oculta.
            dropout: Probabilidade de dropout após cada camada oculta.
        """
        super().__init__()
        half = max(hidden_dim // 2, 1)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.tower = nn.Sequential(
            nn.Linear(2 * embedding_dim + n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, half),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(half, 1),
        )

    def forward(
        self, users: torch.Tensor, items: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """Pontua um lote de interações.

        Args:
            users: Índices codificados de usuário.
            items: Índices codificados de item.
            features: Features comportamentais.

        Returns:
            Logits sem normalização, um por interação.
        """
        merged = torch.cat(
            [self.user_embedding(users), self.item_embedding(items), features], dim=1
        )
        return self.tower(merged).squeeze(-1)


class MLPRecommender(RecommenderModel):
    """Recomendador neural treinado com entropia cruzada e early stopping."""

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
        """Guarda os hiperparâmetros de arquitetura e de otimização.

        Args:
            n_users: Tamanho do vocabulário de usuários.
            n_items: Tamanho do vocabulário de itens.
            embedding_dim: Dimensão dos embeddings de usuário e item.
            hidden_dim: Largura da primeira camada oculta.
            dropout: Probabilidade de dropout após cada camada oculta.
            epochs: Número máximo de épocas de treino.
            batch_size: Tamanho do lote.
            learning_rate: Taxa de aprendizado do Adam.
            weight_decay: Penalidade L2 aplicada pelo Adam.
            early_stopping_patience: Épocas sem melhora na validação toleradas
                antes de parar.
            seed: Semente aplicada antes da inicialização dos pesos.
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
        """Treina a rede, parando cedo pela perda de validação.

        Args:
            data: Interações de treino.
            validation: Interações monitoradas pelo early stopping. Sem elas, a
                própria perda de treino é monitorada.
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
            self._history["train_loss"].append(train_loss)
            self._history["validation_loss"].append(monitored)
            if stopper.update(epoch, monitored, self._network.state_dict()):
                break
        stopper.restore(self._network)
        self._best_epoch = stopper.best_epoch

    def predict_proba(self, data: InteractionData) -> np.ndarray:
        """Pontua interações com a rede treinada.

        Args:
            data: Interações a pontuar.

        Returns:
            Probabilidades de relevância alinhadas com as linhas de entrada.
        """
        network = self._require_network()
        network.eval()
        users, items, features, _ = self._to_tensors(data)
        with torch.no_grad():
            logits = network(users, items, features)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    def hyperparameters(self) -> dict[str, Any]:
        """Descreve a configuração registrada no MLflow.

        Returns:
            Mapa de hiperparâmetro para valor.
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
        """Devolve as curvas de aprendizado por época.

        Returns:
            Mapa com as curvas ``train_loss`` e ``validation_loss``.
        """
        return {key: list(values) for key, values in self._history.items()}

    @property
    def best_epoch(self) -> int:
        """Época cujos pesos o early stopping restaurou.

        Returns:
            Número da época (base um), ou ``0`` antes do treino.
        """
        return self._best_epoch

    @property
    def epochs_run(self) -> int:
        """Quantidade de épocas de fato executadas.

        Returns:
            Tamanho da curva de treino.
        """
        return len(self._history["train_loss"])

    def _build_network(self, n_features: int) -> RecommenderNetwork:
        """Instancia a rede para a largura de features informada.

        Args:
            n_features: Quantidade de features por interação.

        Returns:
            Uma rede recém-inicializada.
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
        """Embrulha os arrays de treino em um loader embaralhado e semeado.

        Args:
            data: Interações de treino.

        Returns:
            Um ``DataLoader`` com embaralhamento reprodutível.
        """
        dataset = TensorDataset(*self._to_tensors(data))
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )

    def _train_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer
    ) -> float:
        """Executa uma passagem de otimização sobre os dados de treino.

        Args:
            loader: Dados de treino em lotes.
            optimizer: Otimizador que atualiza os pesos.

        Returns:
            Perda média de treino da época, ponderada por amostra.
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
        """Calcula a perda que o early stopping deve monitorar.

        Args:
            validation: Interações de validação, se houver.
            fallback: Perda usada quando não existe split de validação.

        Returns:
            O valor da perda monitorada.
        """
        if validation is None or len(validation) == 0:
            return fallback
        network = self._require_network()
        network.eval()
        users, items, features, labels = self._to_tensors(validation)
        with torch.no_grad():
            loss = self._criterion(network(users, items, features), labels)
        return float(loss)

    @staticmethod
    def _to_tensors(
        data: InteractionData,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converte um conjunto de interações em tensores do PyTorch.

        Args:
            data: Interações a converter.

        Returns:
            Tupla com os tensores de usuário, item, features e rótulo.
        """
        return (
            torch.from_numpy(data.user_indices).long(),
            torch.from_numpy(data.item_indices).long(),
            torch.from_numpy(data.features).float(),
            torch.from_numpy(data.labels).float(),
        )

    def _require_network(self) -> RecommenderNetwork:
        """Devolve a rede treinada, barrando o uso antes do treino.

        Returns:
            A rede treinada.

        Raises:
            RuntimeError: Se o modelo ainda não tiver sido treinado.
        """
        if self._network is None:
            raise RuntimeError("MLPRecommender exige fit antes de pontuar.")
        return self._network
