"""
Shared helpers for RoSE-HFL experiment scripts.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Callable, Dict, Optional

import flwr as fl
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import DataLoader

from rosehfl.client import client_fn_factory
from rosehfl.data.data_loader import (
    DATASET_INFO,
    create_client_eval_partitions,
    create_non_iid_partitions,
    get_partition_label_counts,
    load_data,
)
from rosehfl.models.factory import get_model, get_model_size
from rosehfl.utils.fairness import per_client_accuracy_from_weights, summarise_fairness
from rosehfl.utils.json_utils import NumpyEncoder, save_json
from rosehfl.utils.seed import set_seed
from rosehfl.utils.shapley import build_probe_set
from rosehfl.byzantine import select_byzantine_nodes


def timestamped_dir(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def build_model_factory(
    model_name: str,
    dataset_name: str,
    device: str = "cpu",
) -> Callable[[], torch.nn.Module]:
    ds_info = DATASET_INFO[dataset_name]
    return lambda: get_model(
        model_name,
        ds_info["num_classes"],
        ds_info["input_channels"],
        device,
    )


def prepare_shared_context(
    *,
    model_name: str,
    dataset_name: str,
    num_nodes: int,
    batch_size: int,
    shard_size: int,
    shards_per_node: int,
    classes_per_node: int,
    augment: bool,
    seed: int,
    probe_size: int,
    byz_frac: float = 0.0,
    byz_mode: str = "none",
    gaussian_sigma: float = 0.5,
) -> Dict[str, object]:
    set_seed(seed)

    ds_info = DATASET_INFO[dataset_name]
    server_device = "cuda" if torch.cuda.is_available() else "cpu"
    client_device = "cpu"

    model = get_model(
        model_name,
        ds_info["num_classes"],
        ds_info["input_channels"],
        server_device,
    )
    num_params, size_mb = get_model_size(model)
    initial_ndarrays = [value.cpu().numpy() for _, value in model.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(initial_ndarrays)

    train_dataset, test_dataset = load_data(dataset_name, augment=augment)
    partitions = create_non_iid_partitions(
        train_dataset,
        num_nodes,
        shard_size,
        shards_per_node,
        classes_per_node,
        seed=seed,
    )
    node_label_counts = get_partition_label_counts(
        train_dataset,
        partitions,
        ds_info["num_classes"],
    )
    fairness_partitions = create_client_eval_partitions(
        test_dataset,
        node_label_counts,
        ds_info["num_classes"],
        seed=seed,
    )
    probe_subset = build_probe_set(
        test_dataset=test_dataset,
        probe_size=probe_size,
        num_classes=ds_info["num_classes"],
        seed=seed,
    )
    probe_indices = list(probe_subset.indices)
    probe_loader = DataLoader(probe_subset, batch_size=batch_size, shuffle=False)
    byzantine_nodes = set(select_byzantine_nodes(num_nodes, byz_frac, seed=seed))

    client_fn = client_fn_factory(
        model_name,
        dataset_name,
        train_dataset,
        test_dataset,
        partitions,
        batch_size=batch_size,
        device=client_device,
        probe_indices=probe_indices,
        node_label_counts=node_label_counts,
        byzantine_nodes=byzantine_nodes,
        byz_mode=byz_mode,
        gaussian_sigma=gaussian_sigma,
        seed=seed,
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def evaluate_fn(server_round, parameters_ndarrays, config):
        eval_model = get_model(
            model_name,
            ds_info["num_classes"],
            ds_info["input_channels"],
            server_device,
        )
        state_dict = {
            key: torch.tensor(value)
            for key, value in zip(eval_model.state_dict().keys(), parameters_ndarrays)
        }
        eval_model.load_state_dict(state_dict, strict=True)
        eval_model.to(server_device).eval()

        criterion = torch.nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(server_device), labels.to(server_device)
                outputs = eval_model(features)
                total_loss += criterion(outputs, labels).item() * labels.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        return total_loss / max(total, 1), {"accuracy": correct / max(total, 1)}

    return {
        "ds_info": ds_info,
        "server_device": server_device,
        "client_device": client_device,
        "num_params": num_params,
        "size_mb": size_mb,
        "initial_parameters": initial_parameters,
        "initial_ndarrays": initial_ndarrays,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "partitions": partitions,
        "node_label_counts": node_label_counts,
        "fairness_partitions": fairness_partitions,
        "probe_indices": probe_indices,
        "probe_loader": probe_loader,
        "evaluate_fn": evaluate_fn,
        "client_fn": client_fn,
        "model_factory": build_model_factory(model_name, dataset_name, server_device),
        "byzantine_nodes": byzantine_nodes,
    }


def run_strategy(
    strategy,
    client_fn,
    num_clients: int,
    num_rounds: int,
) -> float:
    start = time.time()
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )
    return time.time() - start


def write_fairness_report(
    *,
    output_dir: str,
    parameters,
    model_factory: Callable[[], torch.nn.Module],
    test_dataset,
    fairness_partitions,
    server_device: str,
    seed: int,
) -> Dict[str, object]:
    per_client = per_client_accuracy_from_weights(
        parameters_ndarrays=parameters_to_ndarrays(parameters),
        model_factory=model_factory,
        dataset=test_dataset,
        partitions=fairness_partitions,
        device=torch.device(server_device),
    )
    report = summarise_fairness(per_client, seed=seed)
    save_json(report, os.path.join(output_dir, "fairness.json"))
    return report


def load_checkpoint_if_available(output_dir: str) -> Optional[Dict[str, object]]:
    checkpoint_path = os.path.join(output_dir, "checkpoint.pkl")
    if not os.path.isfile(checkpoint_path):
        return None
    import pickle

    # SECURITY: Only load checkpoint files you trust. pickle.load can execute
    # arbitrary code. Do not load checkpoints from untrusted sources.
    with open(checkpoint_path, "rb") as handle:
        return pickle.load(handle)


def write_summary_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, cls=NumpyEncoder)
