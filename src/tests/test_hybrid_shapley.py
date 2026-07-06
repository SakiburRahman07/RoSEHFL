from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ._module_loader import load_utils_module


load_utils_module("model_state")
shapley = load_utils_module("shapley")
compute_hybrid_phi = shapley.compute_hybrid_phi
predict_probe_logits = shapley.predict_probe_logits


class TinyHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_layer_name = "fc"
        self.backbone = nn.Linear(2, 2, bias=False)
        self.fc = nn.Linear(2, 2)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.eye(2))
            self.fc.weight.zero_()
            self.fc.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))


def _model_factory() -> TinyHeadModel:
    return TinyHeadModel()


def _weights_with_head(weight: np.ndarray, bias: np.ndarray):
    model = _model_factory()
    state = model.state_dict()
    state["fc.weight"] = torch.tensor(weight, dtype=torch.float32)
    state["fc.bias"] = torch.tensor(bias, dtype=torch.float32)
    return [value.detach().cpu().numpy().copy() for value in state.values()]


def test_compute_hybrid_phi_returns_normalised_signal() -> None:
    features = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [1.5, 0.5], [0.5, 1.5]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    probe_loader = DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=False)

    reference_model = _model_factory()
    reference_weights = [
        value.detach().cpu().numpy().copy()
        for value in reference_model.state_dict().values()
    ]

    client_weights = {
        0: _weights_with_head(
            np.array([[2.5, -1.0], [-2.5, 1.0]], dtype=np.float32),
            np.array([0.2, -0.2], dtype=np.float32),
        ),
        1: _weights_with_head(
            np.array([[1.2, -0.6], [-1.2, 0.6]], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        ),
        2: _weights_with_head(
            np.array([[0.1, 0.2], [-0.1, -0.2]], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        ),
    }
    client_sizes = {0: 30, 1: 20, 2: 10}
    probe_logits = {
        node_id: predict_probe_logits(
            weights=weights,
            model_factory=_model_factory,
            probe_loader=probe_loader,
            device=torch.device("cpu"),
        )
        for node_id, weights in client_weights.items()
    }

    hybrid_phi, info = compute_hybrid_phi(
        client_weights=client_weights,
        client_sizes=client_sizes,
        probe_loader=probe_loader,
        model_factory=_model_factory,
        device=torch.device("cpu"),
        reference_weights=reference_weights,
        probe_logits=probe_logits,
        probe_targets=labels.numpy(),
        T=1,
        K=8,
        seed=7,
    )

    assert set(hybrid_phi.keys()) == set(client_weights.keys())
    assert set(info["probe_accuracy"].keys()) == set(client_weights.keys())
    assert 0.1 <= info["lambda"] <= 0.9
    assert all(1e-6 <= value <= 1.0 for value in hybrid_phi.values())
    assert len({round(value, 6) for value in hybrid_phi.values()}) > 1
    best_accuracy_node = max(info["probe_accuracy"], key=info["probe_accuracy"].get)
    worst_hybrid_node = min(hybrid_phi, key=hybrid_phi.get)
    assert best_accuracy_node != worst_hybrid_node
