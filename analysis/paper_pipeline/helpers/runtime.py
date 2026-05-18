from __future__ import annotations

import random

import numpy as np
import torch


def get_device() -> torch.device:
    """Select the best available PyTorch device on the current machine."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for analysis-time reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

