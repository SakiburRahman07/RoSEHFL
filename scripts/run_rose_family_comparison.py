#!/usr/bin/env python3
"""
Compare ShapeFL and all supported RoSE-family variants across paper and effective costs.
"""

from __future__ import annotations

import argparse
import math
import os

from shapefl.utils.seed import set_seed

from ._rose_common import (
    prepare_shared_context,
    run_strategy,
    timestamped_dir,
    write_summary_json,
)
from .run_rose_comparison import build_strategy, build_summary


DEFAULT_STRATEGIES = [
    "shapefl",
    "rose",
    "roseplusplus",
    "rose_q1",
    "rose_q1s",
    "rose_median",
    "rose_trimmed_mean",
    "rose_krum",
]

ALL_STRATEGIES = DEFAULT_STRATEGIES + ["rose_effective"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare ShapeFL and all RoSE-family variants",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="lenet5", choices=["lenet5", "mobilenetv2", "resnet18"])
    parser.add_argument("--dataset", type=str, default="fmnist", choices=["fmnist", "cifar10", "cifar100"])
    parser.add_argument("--num-nodes", type=int, default=30)

    parser.add_argument("--kappa-p", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--kappa-e", type=int, default=1)
    parser.add_argument("--kappa-c", type=int, default=10)
    parser.add_argument("--kappa", type=int, default=50)
    parser.add_argument("--total-local-epochs", type=int, default=None)
    parser.add_argument("--gamma-max", type=float, default=2800.0)
    parser.add_argument("--B-e", type=int, default=None)
    parser.add_argument("--T-max", type=int, default=30)
    parser.add_argument("--target-accuracy", type=float, default=None)
    parser.add_argument("--budget-gb", type=float, default=None)

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
    parser.add_argument("--shapley-T", type=int, default=4)
    parser.add_argument("--shapley-K", type=int, default=6)
    parser.add_argument("--probe-size", type=int, default=1000)
    parser.add_argument("--gamma-anneal", type=str, default="cosine", choices=["fixed", "linear", "cosine"])
    parser.add_argument("--compression-keep-ratio-min", type=float, default=0.05)
    parser.add_argument("--compression-keep-ratio-max", type=float, default=0.25)
    parser.add_argument("--compression-eta", type=float, default=1.0)
    parser.add_argument("--compression-target-deficit", type=float, default=0.25)
    parser.add_argument("--disable-edge-to-cloud-compression", action="store_true")
    parser.add_argument("--edge-min-members", type=int, default=2)
    parser.add_argument("--edge-underfill-penalty", type=float, default=None)

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        choices=ALL_STRATEGIES,
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _metric_last(metrics: dict, field: str):
    values = metrics.get(field, [])
    return values[-1] if values else None


def build_final_metric_snapshot(all_results: dict) -> dict:
    snapshot = {}
    for strategy_name, result in all_results.items():
        metrics = result["metrics"]
        snapshot[strategy_name] = {
            "final_accuracy": _metric_last(metrics, "accuracy"),
            "final_loss": _metric_last(metrics, "loss"),
            "final_paper_per_round_cost_gb": _metric_last(metrics, "paper_per_round_cost_gb")
            or _metric_last(metrics, "per_round_cost_gb"),
            "final_paper_cumulative_cost_gb": _metric_last(metrics, "paper_cumulative_cost_gb")
            or _metric_last(metrics, "cumulative_cost_gb"),
            "final_effective_per_round_cost_gb": _metric_last(metrics, "effective_per_round_cost_gb")
            or _metric_last(metrics, "per_round_cost_gb"),
            "final_effective_cumulative_cost_gb": _metric_last(metrics, "effective_cumulative_cost_gb")
            or _metric_last(metrics, "cumulative_cost_gb"),
            "total_model_payload_bytes": int(sum(metrics.get("model_payload_bytes", []))),
            "total_probe_payload_bytes": int(sum(metrics.get("probe_payload_bytes", []))),
            "best_accuracy": max(metrics["accuracy"]) if metrics.get("accuracy") else None,
            "cloud_rounds_recorded": len(metrics.get("cloud_round", [])),
            "elapsed_seconds": result["elapsed_seconds"],
        }
    return snapshot


def _print_summary(title: str, summary: dict, common_budget_gb: float) -> None:
    print(f"\n{title} summary at common budget {common_budget_gb:.6f} GB")
    for strategy_name, values in summary.items():
        final_accuracy = values["final_accuracy"]
        budget_accuracy = values["accuracy_at_common_budget"]
        per_round_cost_gb = values["per_round_cost_gb"]
        if final_accuracy is not None and budget_accuracy is not None and per_round_cost_gb is not None:
            line = (
                f"  {strategy_name:<18}"
                f"final_acc={final_accuracy:.4f} "
                f"acc_at_budget={budget_accuracy:.4f} "
                f"per_round_cost={per_round_cost_gb:.6f} GB"
            )
        else:
            line = (
                f"  {strategy_name:<18}"
                f"final_acc={final_accuracy} "
                f"acc_at_budget={budget_accuracy} "
                f"per_round_cost={per_round_cost_gb}"
            )
        print(line)


def main() -> None:
    args = build_parser().parse_args()
    if args.no_augment:
        args.augment = False

    from shapefl.data.data_loader import DATASET_INFO

    ds_info = DATASET_INFO[args.dataset]
    if args.shards_per_node is None:
        args.shards_per_node = ds_info["shards_per_node"]
    if args.classes_per_node is None:
        args.classes_per_node = ds_info["classes_per_node"]
    if args.B_e is None:
        args.B_e = max(3, math.ceil(args.num_nodes / 3))
    if args.target_accuracy is None:
        args.target_accuracy = {
            "fmnist": 0.70,
            "cifar10": 0.40,
            "cifar100": 0.20,
        }[args.dataset]
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "results",
            timestamped_dir(f"rose_family_comparison_{args.model}_{args.dataset}"),
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

    all_results = {}
    for strategy_name in args.strategies:
        set_seed(args.seed)
        strategy_output_dir = os.path.join(args.output_dir, strategy_name)
        os.makedirs(strategy_output_dir, exist_ok=True)
        strategy = build_strategy(strategy_name, args, shared, strategy_output_dir)
        elapsed = run_strategy(
            strategy=strategy,
            client_fn=shared["client_fn"],
            num_clients=args.num_nodes,
            num_rounds=strategy.total_flower_rounds,
        )
        all_results[strategy_name] = {
            "metrics": strategy.metrics_history,
            "elapsed_seconds": elapsed,
        }

    paper_common_budget, paper_summary = build_summary(
        all_results,
        mode="paper",
        target_accuracy=args.target_accuracy,
        budget_gb=args.budget_gb,
    )
    effective_common_budget, effective_summary = build_summary(
        all_results,
        mode="matched",
        target_accuracy=args.target_accuracy,
        budget_gb=args.budget_gb,
    )
    final_metrics = build_final_metric_snapshot(all_results)

    payload = {
        "config": vars(args),
        "paper_common_budget_gb": paper_common_budget,
        "effective_common_budget_gb": effective_common_budget,
        "paper_summary": paper_summary,
        "effective_summary": effective_summary,
        "final_metrics": final_metrics,
        "per_round_metrics": {name: result["metrics"] for name, result in all_results.items()},
    }
    output_path = os.path.join(args.output_dir, "rose_family_comparison_results.json")
    write_summary_json(output_path, payload)

    _print_summary("Paper-cost", paper_summary, paper_common_budget)
    _print_summary("Effective-cost", effective_summary, effective_common_budget)
    print(f"\nSaved RoSE-family comparison to {output_path}")


if __name__ == "__main__":
    main()
