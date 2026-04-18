#!/usr/bin/env python3
"""
ShapeFL Computing Node — Flower Deployment
============================================
Runs a Flower client on a computing node and connects to the cloud server.

Usage (from project root):
    python -m scripts.deploy_client --node-id 0 --server-address cloud_ip:8080
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
import flwr as fl

from shapefl.models.factory import get_model
from shapefl.data.data_loader import (
    load_data,
    get_node_dataloader,
    create_non_iid_partitions,
    DATASET_INFO,
)
from shapefl.client import ShapeFlClient


def _load_partition(args):
    """Load data partition for this node."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    partitions_file = os.path.join(project_root, "partitions", "partitions.json")

    train_dataset, test_dataset = load_data(
        args.dataset, augment=args.augment,
    )

    if os.path.exists(partitions_file):
        with open(partitions_file, "r") as f:
            partitions = json.load(f)
        partition = partitions.get(str(args.node_id), partitions.get(args.node_id))
    else:
        partitions = create_non_iid_partitions(
            train_dataset, args.num_nodes, args.shard_size,
            args.shards_per_node, args.classes_per_node, seed=args.seed,
        )
        partition = partitions[args.node_id]

    train_loader = get_node_dataloader(train_dataset, partition, args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(
        description="ShapeFL Computing Node (Flower client)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--server-address", type=str, default="localhost:8080")
    parser.add_argument("--model", type=str, default="lenet5",
                        choices=["lenet5", "mobilenetv2", "resnet18"])
    parser.add_argument("--dataset", type=str, default="fmnist",
                        choices=["fmnist", "cifar10", "cifar100"])
    parser.add_argument("--num-nodes", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=15)
    parser.add_argument("--shards-per-node", type=int, default=None,
                        help="Shards per node (default: from dataset config).")
    parser.add_argument("--classes-per-node", type=int, default=None,
                        help="Classes per node (default: from dataset config).")
    parser.add_argument("--augment", action="store_true",
                        help="Enable training-time data augmentation for CIFAR datasets.")
    parser.add_argument("--no-augment", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.no_augment:
        args.augment = False

    ds_info = DATASET_INFO[args.dataset]
    if args.shards_per_node is None:
        args.shards_per_node = ds_info["shards_per_node"]
    if args.classes_per_node is None:
        args.classes_per_node = ds_info["classes_per_node"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n  Node {args.node_id} starting  |  server={args.server_address}  |  device={device}")

    ds_info = DATASET_INFO[args.dataset]
    model = get_model(args.model, ds_info["num_classes"], ds_info["input_channels"], device)

    train_loader, test_loader = _load_partition(args)
    print(f"  Partition loaded: {len(train_loader.dataset)} train samples")

    client = ShapeFlClient(model, train_loader, test_loader, device)

    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
    )

    print(f"  Node {args.node_id} finished")


if __name__ == "__main__":
    main()
