"""
Model Factory & Utilities
=========================
Centralised factory for creating models by name, plus serialisation helpers.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def get_model(
    model_name: str = "lenet5",
    num_classes: int = 10,
    input_channels: int = 1,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Create and return a model by name."""
    name = model_name.lower()

    if name == "lenet5":
        from .lenet5 import LeNet5
        model = LeNet5(num_classes=num_classes, input_channels=input_channels)
    elif name == "mobilenetv2":
        from .mobilenetv2 import MobileNetV2
        model = MobileNetV2(num_classes=num_classes, input_channels=input_channels)
    elif name == "resnet18":
        from .resnet18 import ResNet18
        model = ResNet18(num_classes=num_classes, input_channels=input_channels)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: lenet5, mobilenetv2, resnet18"
        )

    if device is not None:
        model = model.to(device)

    return model


def get_model_size(model: nn.Module) -> Tuple[int, float]:
    """Return (num_parameters, size_in_mb)."""
    num_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 * 1024)
    return num_params, size_mb
