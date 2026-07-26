"""Global seeding helpers that keep every pipeline run reproducible."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed every random number generator used across the pipeline.

    Args:
        seed: Seed shared by ``random``, NumPy and PyTorch so that a stage
            re-executed with the same inputs produces the same outputs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
