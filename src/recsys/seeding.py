"""Semeadura global que mantém cada execução do pipeline reprodutível."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Semeia todos os geradores de números aleatórios usados no pipeline.

    Args:
        seed: Semente compartilhada por ``random``, NumPy e PyTorch, de modo que
            um stage reexecutado com as mesmas entradas produza a mesma saída.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
