"""
ShapeFL and RoSE-HFL Flower clients.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import Context, NDArrays
from torch.utils.data import DataLoader, Subset

from .byzantine import LabelFlipAttacker, make_byzantine_attacker
from .data.data_loader import DATASET_INFO, get_node_dataloader
from .models.factory import get_model
from .utils.model_state import batch_norm_state_keys
from .utils.shapley import add_gaussian_noise, serialize_probe_logits


class ShapeFlClient(fl.client.NumPyClient):
    """Flower NumPyClient for ShapeFL and RoSE-HFL experiments."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu",
        node_id: Optional[int] = None,
        probe_loader: Optional[DataLoader] = None,
        byzantine_attacker=None,
        seed: int = 42,
        class_prior: Optional[np.ndarray] = None,
        local_bn: bool = False,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.node_id = node_id
        self.probe_loader = probe_loader
        self.byzantine_attacker = byzantine_attacker
        self.seed = seed
        self.class_prior = None if class_prior is None else np.asarray(class_prior, dtype=np.float32)
        self.local_bn = bool(local_bn)
        self._state_keys = list(self.model.state_dict().keys())
        self._bn_state_keys = batch_norm_state_keys(self.model)
        self._server_parameters: Optional[List[np.ndarray]] = None

    def get_parameters(self, config=None) -> NDArrays:
        arrays = [value.detach().cpu().numpy() for _, value in self.model.state_dict().items()]
        if not self.local_bn or not self._bn_state_keys or self._server_parameters is None:
            return arrays
        return [
            self._server_parameters[index].copy()
            if key in self._bn_state_keys
            else arrays[index]
            for index, key in enumerate(self._state_keys)
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        self._server_parameters = [np.asarray(value).copy() for value in parameters]
        current_state = OrderedDict(self.model.state_dict())
        for key, value in zip(self._state_keys, parameters):
            if self.local_bn and key in self._bn_state_keys:
                continue
            current_state[key] = torch.tensor(np.asarray(value))
        self.model.load_state_dict(current_state, strict=True)

    def _compute_probe_logits(self) -> np.ndarray:
        if self.probe_loader is None:
            raise RuntimeError("probe logits requested but probe_loader is not configured")
        self.model.eval()
        self.model.to(self.device)
        batches = []
        with torch.no_grad():
            for features, _ in self.probe_loader:
                features = features.to(self.device)
                logits = self.model(features).detach().cpu().numpy().astype(np.float32)
                batches.append(logits)
        if not batches:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(batches, axis=0)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict,
    ) -> Tuple[NDArrays, int, Dict]:
        self.local_bn = bool(config.get("local_bn", self.local_bn))
        self.set_parameters(parameters)

        epochs = int(config.get("epochs", 1))
        lr = float(config.get("lr", 0.001))
        momentum = float(config.get("momentum", 0.0))
        prox_mu = float(config.get("prox_mu", 0.0))
        logit_adjustment_tau = float(config.get("logit_adjustment_tau", 0.0))

        self.model.train()
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        global_parameters = [
            parameter.detach().clone() for parameter in self.model.parameters()
        ]
        class_prior = self.class_prior
        if "class_prior" in config and config["class_prior"] is not None:
            class_prior = np.asarray(config["class_prior"], dtype=np.float32)
        class_prior_tensor = None
        if class_prior is not None:
            class_prior_tensor = torch.as_tensor(
                class_prior,
                dtype=torch.float32,
                device=self.device,
            ).clamp_min(1e-12)

        total_loss, num_batches = 0.0, 0
        for _ in range(epochs):
            for features, labels in self.train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                if isinstance(self.byzantine_attacker, LabelFlipAttacker):
                    labels = self.byzantine_attacker.apply_to_labels(labels)
                optimizer.zero_grad()
                outputs = self.model(features)
                adjusted_outputs = outputs
                if (
                    logit_adjustment_tau > 0.0
                    and class_prior_tensor is not None
                    and class_prior_tensor.numel() == outputs.shape[1]
                ):
                    adjusted_outputs = outputs + (
                        logit_adjustment_tau * class_prior_tensor.log().view(1, -1)
                    )
                loss = criterion(adjusted_outputs, labels)
                if prox_mu > 0.0:
                    prox_term = torch.zeros((), device=self.device)
                    for parameter, global_parameter in zip(
                        self.model.parameters(),
                        global_parameters,
                    ):
                        prox_term = prox_term + torch.sum((parameter - global_parameter) ** 2)
                    loss = loss + 0.5 * prox_mu * prox_term
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        weights = self.get_parameters()
        if self.byzantine_attacker is not None and not isinstance(
            self.byzantine_attacker,
            LabelFlipAttacker,
        ):
            weights = self.byzantine_attacker.apply_to_weights(weights)
            self.set_parameters(weights)

        metrics: Dict[str, object] = {
            "loss": float(total_loss / max(num_batches, 1)),
            "prox_mu": float(prox_mu),
            "logit_adjustment_tau": float(logit_adjustment_tau),
            "node_id": int(self.node_id) if self.node_id is not None else -1,
            "is_byzantine": bool(self.byzantine_attacker is not None),
            "local_bn": bool(self.local_bn),
        }

        if bool(config.get("emit_probe_logits", False)) and self.probe_loader is not None:
            logits = self._compute_probe_logits()
            dp_epsilon = float(config.get("dp_epsilon", 0.0))
            dp_delta = float(config.get("dp_delta", 1e-5))
            if dp_epsilon > 0.0:
                noise_seed = int(config.get("probe_noise_seed", self.seed))
                logits = add_gaussian_noise(
                    logits,
                    epsilon=dp_epsilon,
                    delta=dp_delta,
                    sensitivity=1.0,
                    seed=noise_seed,
                )
            metrics.update(serialize_probe_logits(logits))

        return weights, len(self.train_loader.dataset), metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        self.local_bn = bool(config.get("local_bn", self.local_bn))
        self.set_parameters(parameters)
        self.model.eval()
        self.model.to(self.device)

        correct, total = 0, 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for features, labels in self.test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                predictions = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        return float(avg_loss), total, {"accuracy": float(accuracy)}


def client_fn_factory(
    model_name: str,
    dataset_name: str,
    train_dataset,
    test_dataset,
    partitions: Dict[int, list],
    batch_size: int = 32,
    device: str = "cpu",
    probe_indices: Optional[Iterable[int]] = None,
    node_label_counts: Optional[Dict[int, np.ndarray]] = None,
    byzantine_nodes: Optional[Set[int]] = None,
    byz_mode: str = "none",
    gaussian_sigma: float = 0.5,
    seed: int = 42,
) -> Callable[[str], fl.client.Client]:
    """Return a Flower ``client_fn`` supporting probe logits and Byzantine nodes."""
    ds_info = DATASET_INFO[dataset_name]
    num_classes = ds_info["num_classes"]
    input_channels = ds_info["input_channels"]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    probe_loader = None
    if probe_indices is not None:
        probe_loader = DataLoader(
            Subset(test_dataset, list(probe_indices)),
            batch_size=batch_size,
            shuffle=False,
        )

    byzantine_nodes = set(byzantine_nodes or set())

    def client_fn(context: Context) -> fl.client.Client:
        node_id = int(context.node_config["partition-id"])
        model = get_model(model_name, num_classes, input_channels, device)
        train_loader = get_node_dataloader(
            train_dataset,
            partitions[node_id],
            batch_size,
        )
        attacker = None
        if node_id in byzantine_nodes:
            attacker = make_byzantine_attacker(
                mode=byz_mode,
                num_classes=num_classes,
                seed=seed + node_id,
                gaussian_sigma=gaussian_sigma,
            )
        class_prior = None
        if node_label_counts is not None and node_id in node_label_counts:
            counts = np.asarray(node_label_counts[node_id], dtype=np.float32)
            total = float(counts.sum())
            if total > 0.0:
                class_prior = counts / total
        return ShapeFlClient(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            node_id=node_id,
            probe_loader=probe_loader,
            byzantine_attacker=attacker,
            seed=seed + node_id,
            class_prior=class_prior,
        ).to_client()

    return client_fn
