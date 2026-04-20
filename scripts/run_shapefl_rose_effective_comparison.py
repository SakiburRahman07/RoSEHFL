#!/usr/bin/env python3
"""
Compare ShapeFL and a RoSE effective-cost variant using real effective-cost accounting only.
"""

from __future__ import annotations

import argparse
import math
import os

from shapefl.strategy import RoSEHFLStrategy, ShapeFlStrategy
from shapefl.utils.seed import set_seed

from ._rose_common import (
    prepare_shared_context,
    run_strategy,
    timestamped_dir,
    write_summary_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ShapeFL vs RoSE-effective comparison with strict effective-cost accounting",
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
    parser.add_argument("--shapley-T", type=int, default=1)
    parser.add_argument("--shapley-K", type=int, default=64)
    parser.add_argument("--probe-size", type=int, default=1000)
    parser.add_argument("--compression-keep-ratio-min", type=float, default=0.05)
    parser.add_argument("--compression-keep-ratio-max", type=float, default=0.25)
    parser.add_argument("--compression-eta", type=float, default=1.0)
    parser.add_argument("--compression-target-deficit", type=float, default=0.25)
    parser.add_argument("--disable-edge-to-cloud-compression", action="store_true")
    parser.add_argument("--edge-min-members", type=int, default=2)
    parser.add_argument("--edge-underfill-penalty", type=float, default=None)
    parser.add_argument(
        "--rose-strategy",
        type=str,
        default="rose_q1s",
        choices=["rose_q1s", "rose_effective"],
    )

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _require_effective_series(metrics: dict, field: str) -> list[float]:
    series_name = f"effective_{field}"
    values = metrics.get(series_name)
    if not values:
        raise RuntimeError(
            f"Missing real {series_name} metrics. "
            "This comparison does not fall back to paper cost."
        )
    return [float(value) for value in values]


def accuracy_at_effective_budget(metrics: dict, budget_gb: float) -> float | None:
    for cost, accuracy in zip(
        _require_effective_series(metrics, "cumulative_cost_gb"),
        metrics.get("accuracy", []),
    ):
        if cost >= budget_gb:
            return float(accuracy)
    return None


def effective_cost_to_target(metrics: dict, target_accuracy: float) -> float | None:
    for cost, accuracy in zip(
        _require_effective_series(metrics, "cumulative_cost_gb"),
        metrics.get("accuracy", []),
    ):
        if float(accuracy) >= float(target_accuracy):
            return float(cost)
    return None


def effective_per_round_cost(metrics: dict) -> float | None:
    series = _require_effective_series(metrics, "per_round_cost_gb")
    if not series:
        return None
    return float(series[-1])


def build_strategy(name: str, args, shared, output_dir: str):
    if name == "shapefl":
        return ShapeFlStrategy(
            model_name=args.model,
            dataset_name=args.dataset,
            num_nodes=args.num_nodes,
            kappa_p=args.kappa_p,
            kappa_e=args.kappa_e,
            kappa_c=args.kappa_c,
            kappa=args.kappa,
            gamma=args.gamma_max,
            B_e=args.B_e,
            T_max=args.T_max,
            lr=args.lr,
            momentum=args.momentum,
            initial_parameters=shared["initial_parameters"],
            planning_mode="shapefl",
            topology=args.topology,
            evaluate_fn=shared["evaluate_fn"],
            node_label_counts=shared["node_label_counts"],
            total_local_epochs=args.total_local_epochs,
        )

    if name == "rose_q1s":
        edge_underfill_penalty = (
            -1.0
            if args.edge_underfill_penalty is None
            else float(args.edge_underfill_penalty)
        )
        return RoSEHFLStrategy(
            model_name=args.model,
            dataset_name=args.dataset,
            num_nodes=args.num_nodes,
            warmup_epochs=3,
            kappa_e=args.kappa_e,
            kappa_c=args.kappa_c,
            kappa=args.kappa,
            gamma_max=args.gamma_max,
            gamma_min=1400.0,
            gamma_anneal="adaptive",
            B_e=args.B_e,
            T_max=args.T_max,
            lr=args.lr,
            momentum=args.momentum,
            initial_parameters=shared["initial_parameters"],
            evaluate_fn=shared["evaluate_fn"],
            topology=args.topology,
            node_label_counts=shared["node_label_counts"],
            total_local_epochs=args.total_local_epochs,
            probe_loader=shared["probe_loader"],
            model_factory=shared["model_factory"],
            server_device=shared["server_device"],
            output_dir=output_dir,
            seed=args.seed,
            shapley_T=2,
            shapley_K=64,
            planning_signal="hybrid",
            probe_size=args.probe_size,
            agg_rule="trust",
            trust_use_shrinkage=True,
            adaptive_gamma_eta=0.5,
            adaptive_gamma_target=0.25,
            warm_start_replan=True,
            warm_start_threshold=0.05,
            replan_cost_increase_tolerance=0.1,
            compression_enabled=True,
            compression_keep_ratio_min=0.15,
            compression_keep_ratio_max=0.30,
            compression_eta=args.compression_eta,
            compression_target_deficit=args.compression_target_deficit,
            compress_edge_to_cloud=not args.disable_edge_to_cloud_compression,
            edge_min_members=max(args.edge_min_members, 3),
            edge_underfill_penalty=edge_underfill_penalty,
            local_objective_prox_mu=args.fedprox_mu,
            logit_adjustment_tau=1.0,
            local_bn=True,
            edge_swa_k=3,
            planning_objective="effective",
            target_accuracy=args.target_accuracy,
            accuracy_guard_tolerance=0.02,
            effective_planning_start_cloud_round=3,
            late_phase_start_fraction=0.8,
            effective_accuracy_delta=0.01,
            probe_emit_mode="cycle_start",
            client_compression_start_cloud_round=3,
            edge_compression_start_cloud_round=4,
            server_optimizer="fedadam",
            server_lr=0.03,
            server_beta1=0.9,
            server_beta2=0.99,
            server_tau=1e-3,
            hard_edge_min_members=3,
        )

    if name == "rose_effective":
        edge_underfill_penalty = (
            -1.0
            if args.edge_underfill_penalty is None
            else float(args.edge_underfill_penalty)
        )
        return RoSEHFLStrategy(
            model_name=args.model,
            dataset_name=args.dataset,
            num_nodes=args.num_nodes,
            warmup_epochs=args.warmup_epochs,
            kappa_e=args.kappa_e,
            kappa_c=args.kappa_c,
            kappa=args.kappa,
            gamma_max=args.gamma_max,
            gamma_min=1400.0,
            gamma_anneal="adaptive",
            B_e=args.B_e,
            T_max=args.T_max,
            lr=args.lr,
            momentum=args.momentum,
            initial_parameters=shared["initial_parameters"],
            evaluate_fn=shared["evaluate_fn"],
            topology=args.topology,
            node_label_counts=shared["node_label_counts"],
            total_local_epochs=args.total_local_epochs,
            probe_loader=shared["probe_loader"],
            model_factory=shared["model_factory"],
            server_device=shared["server_device"],
            output_dir=output_dir,
            seed=args.seed,
            shapley_T=args.shapley_T,
            shapley_K=args.shapley_K,
            planning_signal="hybrid",
            probe_size=args.probe_size,
            agg_rule="trust",
            trust_use_shrinkage=True,
            adaptive_gamma_eta=0.5,
            adaptive_gamma_target=0.25,
            warm_start_replan=True,
            warm_start_threshold=0.05,
            replan_cost_increase_tolerance=0.1,
            compression_enabled=True,
            compression_keep_ratio_min=args.compression_keep_ratio_min,
            compression_keep_ratio_max=args.compression_keep_ratio_max,
            compression_eta=args.compression_eta,
            compression_target_deficit=args.compression_target_deficit,
            compress_edge_to_cloud=not args.disable_edge_to_cloud_compression,
            edge_min_members=args.edge_min_members,
            edge_underfill_penalty=edge_underfill_penalty,
            local_objective_prox_mu=args.fedprox_mu,
            logit_adjustment_tau=1.0,
            local_bn=True,
            edge_swa_k=3,
            planning_objective="effective",
            target_accuracy=args.target_accuracy,
            accuracy_guard_tolerance=0.02,
        )

    raise ValueError(f"Unknown strategy: {name}")


def build_effective_summary(
    all_results: dict,
    *,
    target_accuracy: float,
    budget_gb: float | None,
) -> tuple[float, dict]:
    common_budget = budget_gb
    if common_budget is None:
        final_costs = [
            _require_effective_series(result["metrics"], "cumulative_cost_gb")[-1]
            for result in all_results.values()
        ]
        common_budget = min(final_costs) if final_costs else 0.0

    summary = {}
    for strategy_name, result in all_results.items():
        metrics = result["metrics"]
        effective_cumulative = _require_effective_series(metrics, "cumulative_cost_gb")
        summary[strategy_name] = {
            "final_accuracy": metrics["accuracy"][-1] if metrics["accuracy"] else None,
            "best_accuracy": max(metrics["accuracy"]) if metrics["accuracy"] else None,
            "final_effective_cumulative_cost_gb": effective_cumulative[-1] if effective_cumulative else None,
            "effective_cost_to_target_gb": effective_cost_to_target(metrics, target_accuracy),
            "accuracy_at_common_effective_budget": accuracy_at_effective_budget(metrics, common_budget),
            "effective_per_round_cost_gb": effective_per_round_cost(metrics),
            "total_model_payload_bytes": int(sum(metrics.get("model_payload_bytes", []))),
            "total_probe_payload_bytes": int(sum(metrics.get("probe_payload_bytes", []))),
            "elapsed_seconds": result["elapsed_seconds"],
            "cost_mode": "effective_only",
        }
    return float(common_budget), summary


def run_effective_comparison(args) -> dict:
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
            timestamped_dir(f"shapefl_vs_{args.rose_strategy}_{args.model}_{args.dataset}"),
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
    strategy_names = ["shapefl", args.rose_strategy]
    for strategy_name in strategy_names:
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
        _require_effective_series(strategy.metrics_history, "cumulative_cost_gb")
        _require_effective_series(strategy.metrics_history, "per_round_cost_gb")
        all_results[strategy_name] = {
            "metrics": strategy.metrics_history,
            "elapsed_seconds": elapsed,
        }

    common_budget, summary = build_effective_summary(
        all_results,
        target_accuracy=args.target_accuracy,
        budget_gb=args.budget_gb,
    )

    payload = {
        "config": vars(args),
        "strategies": strategy_names,
        "comparison_mode": "effective_only",
        "common_effective_budget_gb": common_budget,
        "summary": summary,
        "per_round_metrics": {name: result["metrics"] for name, result in all_results.items()},
    }
    result_filename = f"shapefl_vs_{args.rose_strategy}_results.json"
    write_summary_json(
        os.path.join(args.output_dir, result_filename),
        payload,
    )
    for strategy_name in strategy_names:
        result = summary[strategy_name]
        final_accuracy = result["final_accuracy"]
        final_cost = result["final_effective_cumulative_cost_gb"]
        target_cost = result["effective_cost_to_target_gb"]
        final_accuracy_str = "n/a" if final_accuracy is None else f"{float(final_accuracy):.4f}"
        final_cost_str = "n/a" if final_cost is None else f"{float(final_cost):.6f}"
        target_cost_str = "not_reached" if target_cost is None else f"{float(target_cost):.6f}"
        print(
            f"{strategy_name}: "
            f"final_acc={final_accuracy_str} "
            f"final_effective_cost_gb={final_cost_str} "
            f"target_cost_gb={target_cost_str}"
        )
    print(f"Effective-cost comparison complete. Results saved to {args.output_dir}")
    return payload


def main() -> None:
    args = build_parser().parse_args()
    run_effective_comparison(args)


if __name__ == "__main__":
    main()
