#!/usr/bin/env python3
"""
RoSEHFL Computing Node - Flower Deployment ( Based on The original ShapeFL Client )
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

from rosehfl.models.factory import get_model
from rosehfl.data.data_loader import (
    load_data,
    get_node_dataloader,
    create_non_iid_partitions,
    get_partition_label_counts,
    DATASET_INFO,
)
from rosehfl.client import FlClient
from rosehfl.utils.shapley import build_probe_set


def _load_partition_context(args):
    """Load the local train/eval partition and probe context for this node."""
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
    label_counts = get_partition_label_counts(
        train_dataset,
        {int(node_id): indices for node_id, indices in partitions.items()},
        DATASET_INFO[args.dataset]["num_classes"],
    )
    class_prior = None
    if args.node_id in label_counts:
        counts = label_counts[args.node_id].astype("float32")
        total = float(counts.sum())
        if total > 0.0:
            class_prior = counts / total

    probe_subset = build_probe_set(
        test_dataset=test_dataset,
        probe_size=args.probe_size,
        num_classes=DATASET_INFO[args.dataset]["num_classes"],
        seed=args.seed,
    )
    probe_loader = DataLoader(probe_subset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, probe_loader, class_prior


def main():
    parser = argparse.ArgumentParser(
        description="RoSEHFL Computing Node (Flower client)",
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
    parser.add_argument("--probe-size", type=int, default=1000)
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

    train_loader, test_loader, probe_loader, class_prior = _load_partition_context(args)
    print(f"  Partition loaded: {len(train_loader.dataset)} train samples")

    client = FlClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        node_id=args.node_id,
        probe_loader=probe_loader,
        seed=args.seed + args.node_id,
        class_prior=class_prior,
    )

    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(),
    )

    print(f"  Node {args.node_id} finished")


if __name__ == "__main__":
    main()
