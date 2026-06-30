#!/usr/bin/env python3
"""
RoSEHFL Cloud Server — Flower Deployment
=========================================
Runs the Flower gRPC server. Computing nodes connect as clients.
Supports all strategies, all topologies, analytical/delayed cost modes,
checkpoint+resume, and full result persistence.

Usage:
    python -m scripts.deploy_server \
      --strategy rose_q1s --model mobilenetv2 --dataset cifar10 \
      --num-nodes 30 --topology geant2010 --address 0.0.0.0:8080
"""

from __future__ import annotations

import argparse
import os
import time

import flwr as fl

from rosehfl.data.data_loader import DATASET_INFO
from rosehfl.utils.seed import set_seed

try:
    from ._cli_args import add_common_experiment_args, add_single_strategy_arg
    from ._deploy_common import setup_logging, write_deploy_config
    from ._experiment_bundle import (
        ensure_dataset_defaults,
        finalise_strategy_run,
        load_json,
        write_json,
    )
    from ._rose_common import load_checkpoint_if_available, prepare_shared_context, timestamped_dir
    from ._strategy_factory import build_strategy, default_target_accuracy
except ImportError:
    from scripts._cli_args import add_common_experiment_args, add_single_strategy_arg
    from scripts._deploy_common import setup_logging, write_deploy_config
    from scripts._experiment_bundle import (
        ensure_dataset_defaults,
        finalise_strategy_run,
        load_json,
        write_json,
    )
    from scripts._rose_common import load_checkpoint_if_available, prepare_shared_context, timestamped_dir
    from scripts._strategy_factory import build_strategy, default_target_accuracy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RoSEHFL/ShapeFL Flower deployment server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_experiment_args(parser)
    add_single_strategy_arg(parser)
    parser.add_argument("--address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--cost-mode", type=str, default="analytical",
                        choices=["analytical", "delayed"],
                        help="Communication cost mode: analytical (compute only) or delayed (compute + artificial sleep).")
    parser.add_argument("--lan-bandwidth-mbps", type=float, default=100.0,
                        help="Assumed LAN bandwidth for delay calculation (delayed mode only).")
    parser.add_argument("--delay-scale", type=float, default=1.0,
                        help="Multiplier for artificial delays (delayed mode only).")
    parser.add_argument("--include-fairness", action="store_true",
                        help="Compute and write fairness report after training.")
    parser.add_argument("--client-wait-timeout", type=float, default=300.0,
                        help="Seconds to wait for all clients to connect before starting.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.no_augment:
        args.augment = False

    ensure_dataset_defaults(args)
    set_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "results",
            timestamped_dir(f"deploy_{args.strategy}_{args.model}_{args.dataset}"),
        )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    logger = setup_logging(args.output_dir, "server")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Model: {args.model}, Dataset: {args.dataset}")
    logger.info(f"Nodes expected: {args.num_nodes}")
    logger.info(f"Topology: {args.topology}")
    logger.info(f"Cost mode: {args.cost_mode}")
    logger.info(f"Output dir: {args.output_dir}")

    write_deploy_config(
        args.output_dir,
        strategy=args.strategy,
        model=args.model,
        dataset=args.dataset,
        num_nodes=args.num_nodes,
        topology=args.topology,
        cost_mode=args.cost_mode,
        lan_bandwidth_mbps=args.lan_bandwidth_mbps if args.cost_mode == "delayed" else None,
        delay_scale=args.delay_scale if args.cost_mode == "delayed" else None,
    )

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

    strategy = build_strategy(args.strategy, args, shared, args.output_dir)
    strategy.output_dir = args.output_dir

    if args.cost_mode == "delayed":
        strategy.cost_mode = "delayed"
        strategy.lan_bandwidth_mbps = args.lan_bandwidth_mbps
        strategy.delay_scale = args.delay_scale
        logger.info(f"Delay mode: bandwidth={args.lan_bandwidth_mbps} Mbps, scale={args.delay_scale}")

    # Set partition hash for checkpoint integrity verification
    if hasattr(strategy, "set_partitions") and "partitions" in shared:
        strategy.set_partitions(shared["partitions"], seed=args.seed)

    if args.resume and hasattr(strategy, "load_checkpoint_state"):
        checkpoint = load_checkpoint_if_available(args.output_dir)
        if checkpoint is not None:
            strategy.load_checkpoint_state(checkpoint)
            logger.info(f"Resumed from checkpoint: {strategy.completed_flower_rounds} rounds completed")

    total_rounds = getattr(strategy, "remaining_flower_rounds", None)
    if total_rounds is None:
        total_rounds = int(strategy.total_flower_rounds)
    logger.info(f"Flower rounds to run: {total_rounds}")

    logger.info(f"Listening on {args.address}")
    logger.info("Waiting for clients to connect...")

    start_time = time.time()
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=total_rounds),
        strategy=strategy,
    )
    elapsed = time.time() - start_time

    logger.info("Training complete. Finalising results...")
    finalise_strategy_run(
        output_dir=args.output_dir,
        strategy_name=args.strategy,
        args=args,
        strategy=strategy,
        shared=shared,
        elapsed_seconds=elapsed,
        include_fairness=args.include_fairness,
    )

    h = strategy.metrics_history
    if h.get("accuracy"):
        logger.info(f"Final accuracy: {h['accuracy'][-1] * 100:.2f}%")
        logger.info(f"Best accuracy:  {max(h['accuracy']) * 100:.2f}%")


if __name__ == "__main__":
    main()
