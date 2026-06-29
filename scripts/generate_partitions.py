#!/usr/bin/env python3
"""
Pre-partition dataset for deployment.

Generates per-node index files so each client can load only its
assigned data partition without needing the full partition mapping.

Usage:
    python -m scripts.generate_partitions \
      --dataset cifar10 --num-nodes 30 --seed 42 \
      --output-dir partitions/
"""

from __future__ import annotations

import argparse
import json
import os

from rosehfl.data.data_loader import (
    DATASET_INFO,
    create_non_iid_partitions,
    load_data,
)
from rosehfl.utils.seed import set_seed
from rosehfl.utils.shapley import build_probe_set


def generate_partition_files(
    dataset_name: str,
    num_nodes: int,
    shard_size: int,
    shards_per_node: int,
    classes_per_node: int,
    probe_size: int,
    seed: int,
    output_dir: str,
) -> None:
    """Generate and write per-node partition files."""
    set_seed(seed)
    ds_info = DATASET_INFO[dataset_name]

    train_dataset, test_dataset = load_data(dataset_name, augment=False)

    partitions = create_non_iid_partitions(
        train_dataset,
        num_nodes,
        shard_size,
        shards_per_node,
        classes_per_node,
        seed=seed,
    )

    probe_subset = build_probe_set(
        test_dataset=test_dataset,
        probe_size=probe_size,
        num_classes=ds_info["num_classes"],
        seed=seed,
    )
    probe_indices = list(probe_subset.indices)

    os.makedirs(output_dir, exist_ok=True)

    # Write full mapping
    partitions_json = {
        str(node_id): list(indices) for node_id, indices in partitions.items()
    }
    with open(os.path.join(output_dir, "partitions.json"), "w") as f:
        json.dump(partitions_json, f)

    # Write per-node files
    for node_id, indices in partitions.items():
        path = os.path.join(output_dir, f"node_{node_id}_indices.json")
        with open(path, "w") as f:
            json.dump(list(indices), f)

    # Write probe indices
    with open(os.path.join(output_dir, "probe_indices.json"), "w") as f:
        json.dump(probe_indices, f)

    # Write metadata
    metadata = {
        "dataset": dataset_name,
        "num_nodes": num_nodes,
        "num_classes": ds_info["num_classes"],
        "shard_size": shard_size,
        "shards_per_node": shards_per_node,
        "classes_per_node": classes_per_node,
        "probe_size": probe_size,
        "seed": seed,
        "train_dataset_size": len(train_dataset),
        "test_dataset_size": len(test_dataset),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Partitions written to {output_dir}")
    print(f"  {num_nodes} nodes, {len(train_dataset)} train samples")
    for node_id in range(num_nodes):
        print(f"  Node {node_id}: {len(partitions[node_id])} samples")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-partition dataset for deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="fmnist",
                        choices=["fmnist", "cifar10", "cifar100"])
    parser.add_argument("--num-nodes", type=int, default=30)
    parser.add_argument("--shard-size", type=int, default=15)
    parser.add_argument("--shards-per-node", type=int, default=None)
    parser.add_argument("--classes-per-node", type=int, default=None)
    parser.add_argument("--probe-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="partitions/")

    args = parser.parse_args()

    ds_info = DATASET_INFO[args.dataset]
    if args.shards_per_node is None:
        args.shards_per_node = ds_info["shards_per_node"]
    if args.classes_per_node is None:
        args.classes_per_node = ds_info["classes_per_node"]

    generate_partition_files(
        dataset_name=args.dataset,
        num_nodes=args.num_nodes,
        shard_size=args.shard_size,
        shards_per_node=args.shards_per_node,
        classes_per_node=args.classes_per_node,
        probe_size=args.probe_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
