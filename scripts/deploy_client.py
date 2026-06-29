#!/usr/bin/env python3
"""
RoSEHFL Computing Node — Flower Deployment Client
==================================================
Runs a Flower client on a computing node and connects to the cloud server.
Loads only its assigned data partition from pre-generated partition files.

Usage:
    python -m scripts.deploy_client --node-id 0 --server-address cloud_ip:8080
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Subset
import flwr as fl

from rosehfl.models.factory import get_model
from rosehfl.data.data_loader import load_data, DATASET_INFO
from rosehfl.client import FlClient
from rosehfl.utils.shapley import build_probe_set
from scripts._deploy_common import setup_logging


def _load_node_partition(args, project_root: str):
    """Load this node's train partition and the shared probe set."""
    partitions_dir = os.path.join(project_root, "partitions")

    train_dataset, test_dataset = load_data(args.dataset, augment=args.augment)
    ds_info = DATASET_INFO[args.dataset]

    # Load this node's partition indices
    node_file = os.path.join(partitions_dir, f"node_{args.node_id}_indices.json")
    if os.path.isfile(node_file):
        with open(node_file) as f:
            indices = json.load(f)
    else:
        # Fallback: create all partitions in-memory (single-machine mode)
        from rosehfl.data.data_loader import create_non_iid_partitions
        partitions = create_non_iid_partitions(
            train_dataset, args.num_nodes, args.shard_size,
            args.shards_per_node, args.classes_per_node, seed=args.seed,
        )
        indices = partitions[args.node_id]

    train_loader = DataLoader(
        Subset(train_dataset, indices), batch_size=args.batch_size, shuffle=True,
    )

    # Load shared probe set
    probe_file = os.path.join(partitions_dir, "probe_indices.json")
    if os.path.isfile(probe_file):
        with open(probe_file) as f:
            probe_indices = json.load(f)
    else:
        probe_subset = build_probe_set(
            test_dataset=test_dataset,
            probe_size=args.probe_size,
            num_classes=ds_info["num_classes"],
            seed=args.seed,
        )
        probe_indices = list(probe_subset.indices)

    probe_loader = DataLoader(
        Subset(test_dataset, probe_indices), batch_size=args.batch_size, shuffle=False,
    )

    # Compute class prior from this node's labels only
    class_prior = None
    labels = [train_dataset[idx][1] for idx in indices]
    if labels:
        import numpy as np
        counts = np.zeros(ds_info["num_classes"], dtype=np.float32)
        for label in labels:
            counts[int(label)] += 1.0
        total = float(counts.sum())
        if total > 0.0:
            class_prior = counts / total

    return train_loader, probe_loader, class_prior, test_dataset


def main() -> None:
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
    parser.add_argument("--shards-per-node", type=int, default=None)
    parser.add_argument("--classes-per-node", type=int, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-augment", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Connection retry attempts before giving up.")
    parser.add_argument("--retry-delay", type=float, default=10.0,
                        help="Seconds to wait between retries.")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for log files (default: ./logs).")

    args = parser.parse_args()

    if args.no_augment:
        args.augment = False

    ds_info = DATASET_INFO[args.dataset]
    if args.shards_per_node is None:
        args.shards_per_node = ds_info["shards_per_node"]
    if args.classes_per_node is None:
        args.classes_per_node = ds_info["classes_per_node"]

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = args.log_dir or os.path.join(project_root, "logs")
    logger = setup_logging(log_dir, "client", args.node_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Node {args.node_id} starting | server={args.server_address} | device={device}")

    model = get_model(args.model, ds_info["num_classes"], ds_info["input_channels"], device)
    train_loader, probe_loader, class_prior, test_dataset = _load_node_partition(args, project_root)
    logger.info(f"Partition loaded: {len(train_loader.dataset)} train samples")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

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

    for attempt in range(1, args.max_retries + 1):
        try:
            logger.info(f"Connecting to server (attempt {attempt}/{args.max_retries})")
            fl.client.start_client(
                server_address=args.server_address,
                client=client.to_client(),
            )
            logger.info(f"Node {args.node_id} finished")
            return
        except Exception as e:
            logger.error(f"Connection failed (attempt {attempt}): {e}")
            if attempt < args.max_retries:
                time.sleep(args.retry_delay * attempt)
            else:
                logger.error("Max retries reached, giving up")
                raise


if __name__ == "__main__":
    main()
