"""
Byzantine Attack Simulation for RoSE-HFL
========================================
Client-side attack injection used in the Byzantine robustness study.
All attackers take a list of layer ndarrays (``weights``) and optional
batch of (x, y) labels, and return a modified weight list / label
batch.

Attackers
---------
- :class:`LabelFlipAttacker`      - flips labels ``y → (num_classes - 1 - y)``
- :class:`SignFlipAttacker`       - negates all weights before sending
- :class:`GaussianNoiseAttacker`  - additive N(0, σ²) noise on weights

Usage
-----
In the client's ``fit`` method, after local training::

    if self.byzantine_attacker is not None:
        weights = self.byzantine_attacker.apply_to_weights(weights)

For label-flip attacks, the attacker wraps the training DataLoader
and flips the labels at sample time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ═══════════════════════════════════════════════════════════════════════════
#  Attacker interface
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ByzantineAttacker:
    """Base class — subclasses implement one or both hooks."""
    num_classes: int = 10
    seed: int = 42

    def apply_to_labels(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def apply_to_weights(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        return weights


# ═══════════════════════════════════════════════════════════════════════════
#  Attacks
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LabelFlipAttacker(ByzantineAttacker):
    """Flip each label y to (num_classes - 1 - y).

    A standard targeted poisoning attack; trains the model toward
    (systematically) wrong classes without adversarial-noise signatures.
    """
    def apply_to_labels(self, y: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.num_classes - 1, dtype=y.dtype, device=y.device) - y


@dataclass
class SignFlipAttacker(ByzantineAttacker):
    """Negate every weight ndarray (malicious gradient inversion)."""
    def apply_to_weights(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        return [(-w).astype(w.dtype) for w in weights]


@dataclass
class GaussianNoiseAttacker(ByzantineAttacker):
    """Additive N(0, σ²) noise on every weight coordinate."""
    sigma: float = 0.5

    def apply_to_weights(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        rng = np.random.RandomState(self.seed)
        return [
            (w + rng.normal(0.0, self.sigma, size=w.shape)).astype(w.dtype)
            for w in weights
        ]


# ═══════════════════════════════════════════════════════════════════════════
#  DataLoader wrapper for label-flip
# ═══════════════════════════════════════════════════════════════════════════

class FlippedLabelDataset(Dataset):
    """Wrap a dataset and flip every label via the provided attacker."""
    def __init__(self, base: Dataset, attacker: LabelFlipAttacker):
        self.base = base
        self.attacker = attacker

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.base[idx]
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        return x, self.attacker.apply_to_labels(y)


def make_byzantine_attacker(
    mode: str,
    num_classes: int,
    seed: int = 42,
    gaussian_sigma: float = 0.5,
) -> Optional[ByzantineAttacker]:
    """Factory for attack modes; returns None for mode == 'none'."""
    mode = (mode or "none").lower()
    if mode in ("none", "off", ""):
        return None
    if mode == "label_flip":
        return LabelFlipAttacker(num_classes=num_classes, seed=seed)
    if mode == "sign_flip":
        return SignFlipAttacker(num_classes=num_classes, seed=seed)
    if mode in ("gaussian", "gaussian_noise"):
        return GaussianNoiseAttacker(
            num_classes=num_classes, seed=seed, sigma=gaussian_sigma,
        )
    raise ValueError(
        f"make_byzantine_attacker: unknown mode '{mode}'. "
        f"Choose: none, label_flip, sign_flip, gaussian"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Deterministic byzantine node selection
# ═══════════════════════════════════════════════════════════════════════════

def select_byzantine_nodes(
    num_nodes: int, byz_fraction: float, seed: int = 42,
) -> List[int]:
    """Deterministically choose which ``ceil(byz_fraction · N)`` nodes are adversarial."""
    if byz_fraction <= 0:
        return []
    rng = np.random.RandomState(seed)
    k = int(np.ceil(num_nodes * byz_fraction))
    k = min(k, num_nodes)
    return sorted(rng.choice(num_nodes, size=k, replace=False).tolist())
