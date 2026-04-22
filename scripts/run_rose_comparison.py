#!/usr/bin/env python3
"""
RoSE-HFL comparison runner across RoSE and baseline strategies.
"""

from __future__ import annotations

import argparse
import math
import os

from rosehfl.utils.seed import set_seed

from ._rose_common import (
    prepare_shared_context,
    run_strategy,
    timestamped_dir,
    write_summary_json,
)
from ._strategy_factory import build_strategy, default_target_accuracy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RoSE-HFL strategy comparison",
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
    parser.add_argument("--q-fedavg-q", type=float, default=2.0)
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
    parser.add_argument("--comparison-mode", type=str, default="matched", choices=["paper", "matched"])
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
        default=["rose", "rose_q1", "rose_q1s", "shapefl", "fedavg", "fedprox", "gtg_shapley", "q_fedavg"],
        choices=[
            "rose",
            "roseplusplus",
            "rose_q1",
            "rose_q1s",
            "rose_effective",
            "rose_median",
            "rose_trimmed_mean",
            "rose_krum",
            "shapefl",
            "fedavg",
            "fedprox",
            "gtg_shapley",
            "q_fedavg",
        ],
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _cost_series(metrics: dict, mode: str, field: str) -> list[float]:
    if mode == "paper":
        values = metrics.get(f"paper_{field}")
        if values:
            return [float(value) for value in values]
    if mode == "matched":
        values = metrics.get(f"effective_{field}")
        if values:
            return [float(value) for value in values]
    legacy = metrics.get(field, [])
    return [float(value) for value in legacy]


def accuracy_at_budget(metrics: dict, budget_gb: float, *, mode: str) -> float | None:
    for cost, accuracy in zip(_cost_series(metrics, mode, "cumulative_cost_gb"), metrics.get("accuracy", [])):
        if cost >= budget_gb:
            return float(accuracy)
    return None


def cost_to_target(metrics: dict, target_accuracy: float, *, mode: str) -> float | None:
    for cost, accuracy in zip(_cost_series(metrics, mode, "cumulative_cost_gb"), metrics.get("accuracy", [])):
        if accuracy >= target_accuracy:
            return float(cost)
    return None


def per_round_cost(metrics: dict, *, mode: str) -> float | None:
    series = _cost_series(metrics, mode, "per_round_cost_gb")
    if not series:
        return None
    return float(series[-1])


def build_summary(all_results: dict, *, mode: str, target_accuracy: float, budget_gb: float | None) -> tuple[float, dict]:
    common_budget = budget_gb
    if common_budget is None:
        final_costs = [
            _cost_series(result["metrics"], mode, "cumulative_cost_gb")[-1]
            for result in all_results.values()
            if _cost_series(result["metrics"], mode, "cumulative_cost_gb")
        ]
        common_budget = min(final_costs) if final_costs else 0.0

    summary = {}
    for strategy_name, result in all_results.items():
        metrics = result["metrics"]
        summary[strategy_name] = {
            "final_accuracy": metrics["accuracy"][-1] if metrics["accuracy"] else None,
            "best_accuracy": max(metrics["accuracy"]) if metrics["accuracy"] else None,
            "cost_to_target_gb": cost_to_target(metrics, target_accuracy, mode=mode),
            "accuracy_at_common_budget": accuracy_at_budget(metrics, common_budget, mode=mode),
            "per_round_cost_gb": per_round_cost(metrics, mode=mode),
            "elapsed_seconds": result["elapsed_seconds"],
            "cost_mode": mode,
        }
    return float(common_budget), summary


def main() -> None:
    args = build_parser().parse_args()
    if args.no_augment:
        args.augment = False

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
        args.output_dir = os.path.join("results", timestamped_dir(f"rose_comparison_{args.model}_{args.dataset}"))
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
    matched_common_budget, matched_summary = build_summary(
        all_results,
        mode="matched",
        target_accuracy=args.target_accuracy,
        budget_gb=args.budget_gb,
    )
    selected_common_budget = matched_common_budget if args.comparison_mode == "matched" else paper_common_budget
    selected_summary = matched_summary if args.comparison_mode == "matched" else paper_summary

    payload = {
        "config": vars(args),
        "comparison_mode": args.comparison_mode,
        "common_budget_gb": selected_common_budget,
        "paper_common_budget_gb": paper_common_budget,
        "matched_common_budget_gb": matched_common_budget,
        "summary": selected_summary,
        "paper_summary": paper_summary,
        "matched_summary": matched_summary,
        "per_round_metrics": {name: result["metrics"] for name, result in all_results.items()},
    }
    write_summary_json(os.path.join(args.output_dir, "rose_comparison_results.json"), payload)
    print(f"Comparison complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
