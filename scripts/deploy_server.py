#!/usr/bin/env python3
"""
RoSEHFL Cloud Server — Flower Deployment
=========================================
Runs the Flower gRPC server. Computing nodes connect as clients.

Usage (from project root):
    python -m scripts.deploy_server --num-nodes 30 --address 0.0.0.0:8080
"""

import argparse
import math
import os

import flwr as fl

from ._rose_common import prepare_shared_context, timestamped_dir
from ._strategy_factory import build_strategy, default_target_accuracy


def main():
    parser = argparse.ArgumentParser(
        description="RoSEHFL/ShapeFL Flower deployment server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--address", type=str, default="0.0.0.0:8080")
    parser.add_argument(
        "--method",
        type=str,
        default="rose_q1s",
        choices=["shapefl", "rose", "roseplusplus", "rose_q1", "rose_q1s", "rose_effective"],
    )
    parser.add_argument("--model", type=str, default="lenet5",
                        choices=["lenet5", "mobilenetv2", "resnet18"])
    parser.add_argument("--dataset", type=str, default="fmnist",
                        choices=["fmnist", "cifar10", "cifar100"])
    parser.add_argument("--num-nodes", type=int, default=30)

    parser.add_argument("--kappa-p", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--kappa-e", type=int, default=1)
    parser.add_argument("--kappa-c", type=int, default=10)
    parser.add_argument("--kappa", type=int, default=50)
    parser.add_argument("--total-local-epochs", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=2800.0)
    parser.add_argument("--gamma-max", type=float, default=2800.0)
    parser.add_argument("--gamma-anneal", type=str, default="cosine", choices=["fixed", "linear", "cosine", "adaptive"])
    parser.add_argument("--B-e", type=int, default=None)
    parser.add_argument("--T-max", type=int, default=30)
    parser.add_argument("--target-accuracy", type=float, default=None)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="SGD momentum (paper uses 0).")
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=15)
    parser.add_argument("--shards-per-node", type=int, default=None)
    parser.add_argument("--classes-per-node", type=int, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-augment", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--topology", type=str, default="geant2010",
                        choices=["geant2010", "uunet", "tinet", "viatel", "random"],
                        help="Network topology for communication costs (paper: geant2010/uunet/tinet, robust: viatel).")
    parser.add_argument("--planning-signal", type=str, default="shapley", choices=["shapley", "cosine", "hybrid"])
    parser.add_argument("--shapley-T", type=int, default=4)
    parser.add_argument("--shapley-K", type=int, default=6)
    parser.add_argument("--probe-size", type=int, default=1000)
    parser.add_argument("--dp-epsilon", type=float, default=0.0)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--compression-keep-ratio-min", type=float, default=0.05)
    parser.add_argument("--compression-keep-ratio-max", type=float, default=0.25)
    parser.add_argument("--compression-eta", type=float, default=1.0)
    parser.add_argument("--compression-target-deficit", type=float, default=0.25)
    parser.add_argument("--disable-edge-to-cloud-compression", action="store_true")
    parser.add_argument("--edge-min-members", type=int, default=2)
    parser.add_argument("--edge-underfill-penalty", type=float, default=None)
    parser.add_argument("--agg-rule", type=str, default="trust", choices=["trust", "uniform", "median", "trimmed_mean", "krum"])
    parser.add_argument("--agg-trim-ratio", type=float, default=0.2)
    parser.add_argument("--krum-f", type=int, default=1)
    parser.add_argument("--drift-delta", type=float, default=1e-3)
    parser.add_argument("--drift-lambda", type=float, default=0.5)
    parser.add_argument("--max-replans", type=int, default=8)
    parser.add_argument("--disable-drift", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.no_augment:
        args.augment = False

    if args.gamma_max == 2800.0 and args.gamma != 2800.0:
        args.gamma_max = args.gamma

    from rosehfl.data.data_loader import DATASET_INFO

    ds_info = DATASET_INFO[args.dataset]
    if args.shards_per_node is None:
        args.shards_per_node = ds_info["shards_per_node"]
    if args.classes_per_node is None:
        args.classes_per_node = ds_info["classes_per_node"]
    if args.B_e is None:
        args.B_e = max(3, math.ceil(args.num_nodes / 3))
    if args.target_accuracy is None:
        args.target_accuracy = default_target_accuracy(args.dataset)
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "results",
            timestamped_dir(f"deploy_{args.method}_{args.model}_{args.dataset}"),
        )
    os.makedirs(args.output_dir, exist_ok=True)

    shared = prepare_shared_context(
        model_name=args.model,
        dataset_name=args.dataset,
        num_nodes=args.num_nodes,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        shards_per_node=args.shards_per_node,
        classes_per_node=args.classes_per_node,
        augment=args.augment,
        seed=args.seed,
        probe_size=args.probe_size,
    )

    strategy = build_strategy(args.method, args, shared, args.output_dir)
    if args.resume and hasattr(strategy, "load_checkpoint_state"):
        from ._rose_common import load_checkpoint_if_available

        checkpoint = load_checkpoint_if_available(args.output_dir)
        if checkpoint is not None:
            strategy.load_checkpoint_state(checkpoint)
    total_rounds = strategy.total_flower_rounds

    print("\n" + "=" * 60)
    print("  RoSEHFL Cloud Server")
    print("=" * 60)
    print(f"  Method: {args.method}")
    print(f"  Model: {args.model} ({shared['num_params']:,} params)")
    print(f"  Nodes expected: {args.num_nodes}")
    print(f"  Flower rounds: {total_rounds}")
    print(f"  Listening on: {args.address}")
    print(f"  Output dir: {args.output_dir}")
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
