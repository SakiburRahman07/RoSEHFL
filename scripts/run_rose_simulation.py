#!/usr/bin/env python3
"""
RoSE-HFL Flower simulation entry point.
"""

from __future__ import annotations

import argparse
import math
import os

import torch
from rosehfl.utils.seed import set_seed

from ._rose_common import (
    load_checkpoint_if_available,
    prepare_shared_context,
    run_strategy,
    timestamped_dir,
    write_fairness_report,
    write_summary_json,
)
from ._strategy_factory import build_strategy, default_target_accuracy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RoSE-HFL Flower Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="lenet5", choices=["lenet5", "mobilenetv2", "resnet18"])
    parser.add_argument("--dataset", type=str, default="fmnist", choices=["fmnist", "cifar10", "cifar100"])
    parser.add_argument("--num-nodes", type=int, default=30)
    parser.add_argument(
        "--method",
        type=str,
        default="rose",
        choices=["rose", "roseplusplus", "rose_q1", "rose_q1s", "rose_effective"],
    )

    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--kappa-e", type=int, default=1)
    parser.add_argument("--kappa-c", type=int, default=10)
    parser.add_argument("--kappa", type=int, default=50)
    parser.add_argument("--total-local-epochs", type=int, default=None)
    parser.add_argument("--gamma-max", type=float, default=2800.0)
    parser.add_argument("--gamma-anneal", type=str, default="cosine", choices=["fixed", "linear", "cosine", "adaptive"])
    parser.add_argument("--B-e", type=int, default=None)
    parser.add_argument("--T-max", type=int, default=30)
    parser.add_argument("--target-accuracy", type=float, default=None)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--shard-size", type=int, default=15)
    parser.add_argument("--shards-per-node", type=int, default=None)
    parser.add_argument("--classes-per-node", type=int, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--no-augment", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--topology", type=str, default="geant2010", choices=["geant2010", "uunet", "tinet", "viatel", "random"])
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

    parser.add_argument("--drift-delta", type=float, default=1e-3)
    parser.add_argument("--drift-lambda", type=float, default=0.5)
    parser.add_argument("--max-replans", type=int, default=8)
    parser.add_argument("--disable-drift", action="store_true")

    parser.add_argument("--agg-rule", type=str, default="trust", choices=["trust", "uniform", "median", "trimmed_mean", "krum"])
    parser.add_argument("--agg-trim-ratio", type=float, default=0.2)
    parser.add_argument("--krum-f", type=int, default=1)

    parser.add_argument("--byz-frac", type=float, default=0.0)
    parser.add_argument("--byz-mode", type=str, default="none", choices=["none", "label_flip", "sign_flip", "gaussian"])
    parser.add_argument("--gaussian-sigma", type=float, default=0.5)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run_rose_experiment(args) -> dict:
    if args.no_augment:
        args.augment = False

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "results",
            timestamped_dir(f"rose_{args.model}_{args.dataset}"),
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
        byz_frac=args.byz_frac,
        byz_mode=args.byz_mode,
        gaussian_sigma=args.gaussian_sigma,
    )

    if args.B_e is None:
        args.B_e = max(3, math.ceil(args.num_nodes / 3))
    if args.target_accuracy is None:
        args.target_accuracy = default_target_accuracy(args.dataset)
    strategy = build_strategy(args.method, args, shared, args.output_dir)

    if args.resume:
        checkpoint = load_checkpoint_if_available(args.output_dir)
        if checkpoint is not None:
            strategy.load_checkpoint_state(checkpoint)

    write_summary_json(os.path.join(args.output_dir, "config.json"), vars(args))

    remaining_rounds = strategy.remaining_flower_rounds
    elapsed = 0.0
    if remaining_rounds > 0:
        elapsed = run_strategy(
            strategy=strategy,
            client_fn=shared["client_fn"],
            num_clients=args.num_nodes,
            num_rounds=remaining_rounds,
        )

    fairness_report = write_fairness_report(
        output_dir=args.output_dir,
        parameters=strategy.global_parameters,
        model_factory=shared["model_factory"],
        test_dataset=shared["test_dataset"],
        fairness_partitions=shared["fairness_partitions"],
        server_device=shared["server_device"],
        seed=args.seed,
    )
    strategy._persist_artifacts(completed=True)

    summary = {
        "output_dir": args.output_dir,
        "elapsed_seconds": elapsed,
        "final_accuracy": strategy.metrics_history["accuracy"][-1] if strategy.metrics_history["accuracy"] else None,
        "best_accuracy": max(strategy.metrics_history["accuracy"]) if strategy.metrics_history["accuracy"] else None,
        "final_cumulative_cost_gb": strategy.metrics_history["cumulative_cost_gb"][-1] if strategy.metrics_history["cumulative_cost_gb"] else 0.0,
        "final_paper_cumulative_cost_gb": (
            strategy.metrics_history["paper_cumulative_cost_gb"][-1]
            if strategy.metrics_history.get("paper_cumulative_cost_gb")
            else 0.0
        ),
        "final_effective_cumulative_cost_gb": (
            strategy.metrics_history["effective_cumulative_cost_gb"][-1]
            if strategy.metrics_history.get("effective_cumulative_cost_gb")
            else strategy.metrics_history["cumulative_cost_gb"][-1]
            if strategy.metrics_history["cumulative_cost_gb"]
            else 0.0
        ),
        "total_model_payload_bytes": int(sum(strategy.metrics_history.get("model_payload_bytes", []))),
        "total_probe_payload_bytes": int(sum(strategy.metrics_history.get("probe_payload_bytes", []))),
        "replan_count": strategy.replan_count,
        "fairness": fairness_report,
    }
    write_summary_json(os.path.join(args.output_dir, "summary.json"), summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    ds_info = prepare_defaults(args)
    summary = run_rose_experiment(args)
    print("\nRoSE-HFL simulation complete")
    print(f"  Output dir: {summary['output_dir']}")
    if summary["final_accuracy"] is not None:
        print(f"  Final accuracy: {summary['final_accuracy'] * 100:.2f}%")
        print(f"  Best accuracy:  {summary['best_accuracy'] * 100:.2f}%")
    print(f"  Replans: {summary['replan_count']}")
    print(f"  Time: {summary['elapsed_seconds']:.1f}s")


def prepare_defaults(args):
    ds_info = prepare_dataset_defaults(args)
    set_seed(args.seed)
    return ds_info


def prepare_dataset_defaults(args):
    from rosehfl.data.data_loader import DATASET_INFO

    ds_info = DATASET_INFO[args.dataset]
    if args.shards_per_node is None:
        args.shards_per_node = ds_info["shards_per_node"]
    if args.classes_per_node is None:
        args.classes_per_node = ds_info["classes_per_node"]
    return ds_info


if __name__ == "__main__":
    main()
