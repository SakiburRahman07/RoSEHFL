"""
MobileNetV2 for CIFAR-10 Classification
========================================
Adapted for 32×32 colour images (CIFAR-10, 10 classes).
Reference: Sandler et al., "MobileNetV2" (CVPR 2018)
"""

import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, expand_ratio: int):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden = in_channels * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
        ])
        layers.extend([
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 for CIFAR-10 (32×32).

    The classifier layer is named ``classifier`` and is used for
    ShapeFL similarity computation.
    """

    _cfg = [
        (1, 16, 1, 1),
        (6, 24, 2, 1),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.linear_layer_name = "classifier"

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        in_ch = 32
        layers = []
        for t, c, n, s in self._cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(in_ch, c, stride, t))
                in_ch = c
        self.blocks = nn.Sequential(*layers)

        self.last_conv = nn.Sequential(
            nn.Conv2d(in_ch, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.blocks(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_linear_layer_params(self) -> torch.Tensor:
        weight = self.classifier.weight.data.flatten()
        bias = self.classifier.bias.data.flatten()
        return torch.cat([weight, bias])

    def get_linear_layer_size(self) -> int:
        return self.classifier.weight.numel() + self.classifier.bias.numel()
