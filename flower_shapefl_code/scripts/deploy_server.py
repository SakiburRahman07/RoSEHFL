#!/usr/bin/env python3
"""
ShapeFL Cloud Server — Flower Deployment
=========================================
Runs the Flower gRPC server. Computing nodes connect as clients.

Usage (from project root):
    python -m scripts.deploy_server --num-nodes 30 --address 0.0.0.0:8080
"""

import argparse
import math

import numpy as np
import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters

from shapefl.models.factory import get_model, get_model_size
from shapefl.data.data_loader import DATASET_INFO
from shapefl.strategy import ShapeFlStrategy


def main():
    parser = argparse.ArgumentParser(
        description="ShapeFL Cloud Server (Flower deployment)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--model", type=str, default="lenet5",
                        choices=["lenet5", "mobilenetv2", "resnet18"])
    parser.add_argument("--dataset", type=str, default="fmnist",
                        choices=["fmnist", "cifar10", "cifar100"])
    parser.add_argument("--num-nodes", type=int, default=30)

    parser.add_argument("--kappa-p", type=int, default=30)
    parser.add_argument("--kappa-e", type=int, default=1)
    parser.add_argument("--kappa-c", type=int, default=10)
    parser.add_argument("--kappa", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=2800.0)
    parser.add_argument("--B-e", type=int, default=None)
    parser.add_argument("--T-max", type=int, default=30)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="SGD momentum (paper uses 0).")
    parser.add_argument("--planning-mode", type=str, default="shapefl",
                        choices=["shapefl", "cost_first", "data_first", "random"])
    parser.add_argument("--topology", type=str, default="geant2010",
                        choices=["geant2010", "uunet", "tinet", "viatel", "random"],
                        help="Network topology for communication costs (paper: geant2010/uunet/tinet, robust: viatel).")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.B_e is None:
        args.B_e = max(3, math.ceil(args.num_nodes / 3))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds_info = DATASET_INFO[args.dataset]
    model = get_model(args.model, ds_info["num_classes"], ds_info["input_channels"], "cpu")
    num_params, size_mb = get_model_size(model)
    initial_params = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in model.state_dict().items()]
    )

    strategy = ShapeFlStrategy(
        model_name=args.model, dataset_name=args.dataset, num_nodes=args.num_nodes,
        kappa_p=args.kappa_p, kappa_e=args.kappa_e, kappa_c=args.kappa_c,
        kappa=args.kappa, gamma=args.gamma, B_e=args.B_e, T_max=args.T_max,
        lr=args.lr, momentum=args.momentum, initial_parameters=initial_params,
        planning_mode=args.planning_mode, topology=args.topology,
    )
    total_rounds = strategy.total_flower_rounds

    print("\n" + "=" * 60)
    print("  ShapeFL Cloud Server")
    print("=" * 60)
    print(f"  Model: {args.model} ({num_params:,} params)")
    print(f"  Nodes expected: {args.num_nodes}")
    print(f"  Flower rounds: {total_rounds}")
    print(f"  Listening on: {args.address}")
    print(f"  Planning: {args.planning_mode}")
    print("\n  Waiting for clients to connect...\n")

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=total_rounds),
        strategy=strategy,
    )

    h = strategy.metrics_history
    if h["accuracy"]:
        print(f"\nFinal accuracy: {h['accuracy'][-1] * 100:.2f}%")
        print(f"Best accuracy:  {max(h['accuracy']) * 100:.2f}%")


if __name__ == "__main__":
    main()
