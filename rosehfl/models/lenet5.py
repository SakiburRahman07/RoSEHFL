"""
LeNet-5 Model for Fashion-MNIST Classification
===============================================
Architecture adapted for Fashion-MNIST (28x28 grayscale, 10 classes).
Reference: LeCun et al., "Gradient-based learning applied to document recognition"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    LeNet-5 for Fashion-MNIST.

    Architecture:
        Input:  1x28x28
        Conv1:  1→6,  5x5 kernel, padding=2  →  6x28x28
        Pool1:  2x2 max pool                 →  6x14x14
        Conv2:  6→16, 5x5 kernel             → 16x10x10
        Pool2:  2x2 max pool                 → 16x5x5
        FC1:    400→120
        FC2:    120→84
        FC3:    84→num_classes (output / linear layer)

    Total parameters: ~61,706
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        super().__init__()
        self.linear_layer_name = "fc3"  # For ShapeFL similarity

        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization — consistent with MobileNetV2 / ResNet18."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_linear_layer_params(self) -> torch.Tensor:
        """Flattened fc3 weights + biases (for ShapeFL similarity)."""
        weight = self.fc3.weight.data.flatten()
        bias = self.fc3.bias.data.flatten()
        return torch.cat([weight, bias])

    def get_linear_layer_size(self) -> int:
        return self.fc3.weight.numel() + self.fc3.bias.numel()
