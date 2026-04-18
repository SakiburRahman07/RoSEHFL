"""
Helpers for working with model state dictionaries.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set

import torch
import torch.nn as nn


def batch_norm_state_keys(model: nn.Module) -> Set[str]:
    """Return all state-dict keys owned by BatchNorm modules."""
    prefixes = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    keys = set()
    for key in model.state_dict().keys():
        for prefix in prefixes:
            if key == prefix or key.startswith(f"{prefix}."):
                keys.add(key)
                break
    return keys


def head_state_keys(model: nn.Module) -> Set[str]:
    """Return state-dict keys for the model's classification head."""
    linear_name = getattr(model, "linear_layer_name", None)
    if not linear_name:
        return set()
    return {
        key for key in model.state_dict().keys()
        if key == linear_name or key.startswith(f"{linear_name}.")
    }


def state_key_indices(
    state_keys: Sequence[str],
    selected_keys: Iterable[str],
) -> List[int]:
    """Return the indices of ``selected_keys`` within ``state_keys``."""
    selected = set(selected_keys)
    return [index for index, key in enumerate(state_keys) if key in selected]
