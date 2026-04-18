#!/usr/bin/env python3
"""
ShapeFL Flower Simulation
=========================
Run the complete ShapeFL pipeline using Flower's simulation framework.
All clients run in a single process — no networking required.

Usage (from project root):
    python -m scripts.run_simulation [--model lenet5] [--dataset fmnist]
    python -m scripts.run_simulation --planning-mode cost_first --gamma 0
"""

import argparse
import math
import os
import json
import time

import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters
from torch.utils.data import DataLoader

from shapefl.models.factory import get_model, get_model_size
from shapefl.data.data_loader import (
    load_data,
    create_non_iid_partitions,
    get_partition_label_counts,
    DATASET_INFO,
)
from shapefl.client import client_fn_factory
from shapefl.strategy import ShapeFlStrategy
from shapefl.utils.seed import set_seed
from shapefl.utils.json_utils import NumpyEncoder


def main():
    parser = argparse.ArgumentParser(
        description="ShapeFL Flower Simulation (Algorithm 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="lenet5",
                        choices=["lenet5", "mobilenetv2", "resnet18"])
    parser.add_argument("--dataset", type=str, default="fmnist",
                        choices=["fmnist", "cifar10", "cifar100"])
    parser.add_argument("--num-nodes", type=int, default=30)

    parser.add_argument("--kappa-p", type=int, default=30)
    parser.add_argument("--kappa-e", type=int, default=1)
    parser.add_argument("--kappa-c", type=int, default=10)
    parser.add_argument("--kappa", type=int, default=50)
    parser.add_argument("--total-local-epochs", type=int, default=None,
                        help="Exact local-epoch budget per node for fixed-budget reproduction runs.")
    parser.add_argument("--gamma", type=float, default=2800.0)
    parser.add_argument("--B-e", type=int, default=None)
    parser.add_argument("--T-max", type=int, default=30)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="SGD momentum (paper uses 0; set 0.9 for faster convergence).")
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--shard-size", type=int, default=15)
    parser.add_argument("--shards-per-node", type=int, default=None)
    parser.add_argument("--classes-per-node", type=int, default=None)

    parser.add_argument("--augment", action="store_true",
                        help="Enable training-time data augmentation for CIFAR datasets.")
    parser.add_argument("--no-augment", action="store_true",
                        help=argparse.SUPPRESS)

    parser.add_argument("--planning-mode", type=str, default="shapefl",
                        choices=["shapefl", "cost_first", "data_first", "share", "random"])
    parser.add_argument("--topology", type=str, default="geant2010",
                        choices=["geant2010", "uunet", "tinet", "viatel", "random"],
                        help="Network topology for communication costs (paper: geant2010/uunet/tinet, robust: viatel).")

    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Auto-generated from config + timestamp if omitted.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # ── Auto-generate unique output directory ─────────────────────────
    if args.output_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            f"results/{args.model}_{args.dataset}_{args.planning_mode}"
            f"_n{args.num_nodes}_k{args.kappa}_{ts}"
        )
    # Warn if output directory already has results
    _existing = os.path.join(args.output_dir, "flower_simulation_results.json")
    if os.path.isfile(_existing):
        print(f"  WARNING: {_existing} already exists and will be overwritten.")

    ds_info = DATASET_INFO[args.dataset]
    if args.shards_per_node is None:
        args.shards_per_node = ds_info["shards_per_node"]
    if args.classes_per_node is None:
        args.classes_per_node = ds_info["classes_per_node"]
    if args.B_e is None:
        args.B_e = max(3, math.ceil(args.num_nodes / 3))

    set_seed(args.seed)

    if args.no_augment:
        args.augment = False

    # Use CPU clients in simulation to avoid multiple worker processes
    # contending for the same CUDA device. Keep server-side evaluation on GPU.
    server_device = "cuda" if torch.cuda.is_available() else "cpu"
    client_device = "cpu"

    model = get_model(
        args.model, ds_info["num_classes"], ds_info["input_channels"], server_device
    )
    num_params, size_mb = get_model_size(model)
    initial_params = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in model.state_dict().items()]
    )

    train_dataset, test_dataset = load_data(args.dataset, augment=args.augment)
    partitions = create_non_iid_partitions(
        train_dataset, args.num_nodes, args.shard_size,
        args.shards_per_node, args.classes_per_node, seed=42,
    )
    node_label_counts = get_partition_label_counts(
        train_dataset, partitions, ds_info["num_classes"],
    )

    # ── Centralized evaluation function (server-side) ─────────────────
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    def evaluate_fn(server_round, parameters_ndarrays, config):
        """Evaluate the global model on the full test set (server-side)."""
        eval_model = get_model(
            args.model,
            ds_info["num_classes"],
            ds_info["input_channels"],
            server_device,
        )
        keys = list(eval_model.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters_ndarrays)}
        eval_model.load_state_dict(state_dict, strict=True)
        eval_model.to(server_device)
        eval_model.eval()

        total_loss, correct, total = 0.0, 0, 0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(server_device), labels.to(server_device)
                outputs = eval_model(images)
                total_loss += criterion(outputs, labels).item() * labels.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, {"accuracy": accuracy}

    total_rounds = 1 + args.kappa * args.kappa_c
    if args.total_local_epochs is not None:
        total_rounds = 1 + math.ceil(args.total_local_epochs / args.kappa_e)
    print("\n" + "=" * 65)
    print("  ShapeFL Flower Simulation")
    print("=" * 65)
    print(f"  Model: {args.model} ({num_params:,} params, {size_mb:.3f} MB)")
    print(f"  Dataset: {args.dataset}")
    print(f"  Nodes: {args.num_nodes}  B_e: {args.B_e}")
    print(f"  κ_p={args.kappa_p}  κ_e={args.kappa_e}  κ_c={args.kappa_c}  κ={args.kappa}")
    if args.total_local_epochs is not None:
        print(f"  Total local epochs per node: {args.total_local_epochs}")
    print(f"  γ={args.gamma}  lr={args.lr}  momentum={args.momentum}  planning={args.planning_mode}")
    print(f"  Flower rounds: {total_rounds} (1 pretrain + {args.kappa}×{args.kappa_c} training)")
    print()

    strategy = ShapeFlStrategy(
        model_name=args.model, dataset_name=args.dataset, num_nodes=args.num_nodes,
        kappa_p=args.kappa_p, kappa_e=args.kappa_e, kappa_c=args.kappa_c,
        kappa=args.kappa, gamma=args.gamma, B_e=args.B_e, T_max=args.T_max,
        lr=args.lr, momentum=args.momentum, initial_parameters=initial_params,
        planning_mode=args.planning_mode,
        topology=args.topology, evaluate_fn=evaluate_fn,
        node_label_counts=node_label_counts,
        total_local_epochs=args.total_local_epochs,
    )

    client_fn = client_fn_factory(
        args.model, args.dataset, train_dataset, test_dataset,
        partitions, args.batch_size, client_device,
    )

    start = time.time()
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=args.num_nodes,
        config=fl.server.ServerConfig(num_rounds=total_rounds),
        strategy=strategy, client_resources={"num_cpus": 1},
    )
    elapsed = time.time() - start

    print("\n" + "=" * 65)
    print("  Simulation Complete")
    print("=" * 65)
    h = strategy.metrics_history
    if h["accuracy"]:
        print(f"  Final accuracy: {h['accuracy'][-1] * 100:.2f}%")
        print(f"  Best accuracy:  {max(h['accuracy']) * 100:.2f}%")
        print(f"  Cumulative cost: {h['cumulative_cost_gb'][-1]:.4f} GB")
    print(f"  Time: {elapsed:.1f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "config": vars(args),
        "model_params": num_params, "model_size_mb": size_mb,
        "planning_mode": args.planning_mode,
        "metrics": {k: [float(v) for v in vals] for k, vals in h.items()},
        "edges": sorted(list(strategy.selected_edges)),
        "edge_nodes": {str(e): sorted(ns) for e, ns in strategy.edge_nodes.items()},
        "per_round_cost_gb": strategy.per_round_cost_gb,
        "time_seconds": elapsed,
    }
    path = os.path.join(args.output_dir, "flower_simulation_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved to {path}")

    # ── Visualization ────────────────────────────────────────────────────
    try:
        from shapefl.utils.visualization import visualize_simulation as viz_sim
        viz_sim(
            metrics={k: [float(v) for v in vals] for k, vals in h.items()},
            config=vars(args),
            edge_nodes={str(e): sorted(ns) for e, ns in strategy.edge_nodes.items()},
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"  [Warning] Visualization failed: {e}")


if __name__ == "__main__":
    main()
