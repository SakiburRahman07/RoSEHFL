"""
Reproducibility Utilities
=========================
Ensures deterministic behaviour across runs by seeding all RNGs
and disabling non-deterministic cuDNN kernels.

Adopted from ShapeFL-Flower's set_seed() for full reproducibility.
"""

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seed for full reproducibility.

    Sets seeds for:
      - Python's ``numpy``
      - PyTorch CPU & CUDA
      - cuDNN (deterministic mode, no benchmarking)

    Args:
        seed: Integer seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
