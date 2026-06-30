#!/usr/bin/env python3
"""
Byzantine robustness sweep for RoSE-HFL.
"""

from __future__ import annotations

import argparse
import math
import os

from rosehfl.strategy import RoSEHFLStrategy, ShapeFlStrategy

from ._rose_common import prepare_shared_context, run_strategy, timestamped_dir, write_summary_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RoSE-HFL Byzantine sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="mobilenetv2", choices=["lenet5", "mobilenetv2", "resnet18"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["fmnist", "cifar10", "cifar100"])
    parser.add_argument("--num-nodes", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--kappa-p", type=int, default=30)
    parser.add_argument("--kappa-e", type=int, default=1)
    parser.add_argument("--kappa-c", type=int, default=10)
    parser.add_argument("--kappa", type=int, default=20)
    parser.add_argument("--gamma-max", type=float, default=2800.0)
    parser.add_argument("--B-e", type=int, default=None)
    parser.add_argument("--T-max", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=15)
    parser.add_argument("--shards-per-node", type=int, default=None)
    parser.add_argument("--classes-per-node", type=int, default=None)
    parser.add_argument("--topology", type=str, default="geant2010", choices=["geant2010", "uunet", "tinet", "viatel", "random"])
    parser.add_argument("--probe-size", type=int, default=1000)
    parser.add_argument("--shapley-T", type=int, default=4)
    parser.add_argument("--shapley-K", type=int, default=6)
    parser.add_argument("--byz-fractions", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--attacks", nargs="+", type=str, default=["label_flip", "sign_flip", "gaussian"])
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["rose", "roseplusplus", "full_minus_c4", "median", "trimmed_mean", "krum"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from rosehfl.data.data_loader import DATASET_INFO

    ds_info = DATASET_INFO[args.dataset]
    if args.shards_per_node is None:
        args.shards_per_node = ds_info["shards_per_node"]
    if args.classes_per_node is None:
        args.classes_per_node = ds_info["classes_per_node"]
    if args.B_e is None:
        args.B_e = max(3, math.ceil(args.num_nodes / 3))
    if args.output_dir is None:
        args.output_dir = os.path.join("results", timestamped_dir(f"rose_byz_{args.model}_{args.dataset}"))
    os.makedirs(args.output_dir, exist_ok=True)

    summary = {}
    for fraction in args.byz_fractions:
        for attack in args.attacks:
            shared = prepare_shared_context(
                model_name=args.model,
                dataset_name=args.dataset,
                num_nodes=args.num_nodes,
                batch_size=args.batch_size,
                shard_size=args.shard_size,
                shards_per_node=args.shards_per_node,
                classes_per_node=args.classes_per_node,
                augment=False,
                seed=args.seed,
                probe_size=args.probe_size,
                byz_frac=fraction,
                byz_mode=attack,
            )
            combo_key = f"frac_{fraction:g}__{attack}"
            summary[combo_key] = {}
            for strategy_name in args.strategies:
                strategy_output_dir = os.path.join(args.output_dir, combo_key, strategy_name)
                os.makedirs(strategy_output_dir, exist_ok=True)
                if strategy_name == "rose":
                    strategy = RoSEHFLStrategy(
                        model_name=args.model,
                        dataset_name=args.dataset,
                        num_nodes=args.num_nodes,
                        warmup_epochs=args.warmup_epochs,
                        kappa_e=args.kappa_e,
                        kappa_c=args.kappa_c,
                        kappa=args.kappa,
                        gamma_max=args.gamma_max,
                        B_e=args.B_e,
                        T_max=args.T_max,
                        lr=args.lr,
                        momentum=args.momentum,
                        initial_parameters=shared["initial_parameters"],
                        evaluate_fn=shared["evaluate_fn"],
                        topology=args.topology,
                        node_label_counts=shared["node_label_counts"],
                        probe_loader=shared["probe_loader"],
                        model_factory=shared["model_factory"],
                        server_device=shared["server_device"],
                        output_dir=strategy_output_dir,
                        seed=args.seed,
                        shapley_T=args.shapley_T,
                        shapley_K=args.shapley_K,
                        agg_rule="trust",
                        trust_use_shrinkage=True,
                    )
                elif strategy_name == "roseplusplus":
                    strategy = RoSEHFLStrategy(
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
                        probe_loader=shared["probe_loader"],
                        model_factory=shared["model_factory"],
                        server_device=shared["server_device"],
                        output_dir=strategy_output_dir,
                        seed=args.seed,
                        shapley_T=1,
                        shapley_K=64,
                        planning_signal="hybrid",
                        agg_rule="trust",
                        trust_use_shrinkage=True,
                        adaptive_gamma_eta=0.5,
                        adaptive_gamma_target=0.25,
                        warm_start_replan=True,
                        warm_start_threshold=0.05,
                        replan_cost_increase_tolerance=0.1,
                        local_objective_prox_mu=0.01,
                        logit_adjustment_tau=1.0,
                        local_bn=True,
                        edge_swa_k=3,
                    )
                elif strategy_name == "full_minus_c4":
                    strategy = RoSEHFLStrategy(
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
                        probe_loader=shared["probe_loader"],
                        model_factory=shared["model_factory"],
                        server_device=shared["server_device"],
                        output_dir=strategy_output_dir,
                        seed=args.seed,
                        shapley_T=1,
                        shapley_K=64,
                        planning_signal="hybrid",
                        agg_rule="trust_legacy",
                        trust_use_shrinkage=False,
                        adaptive_gamma_eta=0.5,
                        adaptive_gamma_target=0.25,
                        warm_start_replan=True,
                        warm_start_threshold=0.05,
                        replan_cost_increase_tolerance=0.1,
                        local_objective_prox_mu=0.01,
                        logit_adjustment_tau=1.0,
                        local_bn=True,
                        edge_swa_k=3,
                    )
                elif strategy_name in {"median", "trimmed_mean", "krum"}:
                    strategy = RoSEHFLStrategy(
                        model_name=args.model,
                        dataset_name=args.dataset,
                        num_nodes=args.num_nodes,
                        warmup_epochs=args.warmup_epochs,
                        kappa_e=args.kappa_e,
                        kappa_c=args.kappa_c,
                        kappa=args.kappa,
                        gamma_max=args.gamma_max,
                        B_e=args.B_e,
                        T_max=args.T_max,
                        lr=args.lr,
                        momentum=args.momentum,
                        initial_parameters=shared["initial_parameters"],
                        evaluate_fn=shared["evaluate_fn"],
                        topology=args.topology,
                        node_label_counts=shared["node_label_counts"],
                        probe_loader=shared["probe_loader"],
                        model_factory=shared["model_factory"],
                        server_device=shared["server_device"],
                        output_dir=strategy_output_dir,
                        seed=args.seed,
                        shapley_T=args.shapley_T,
                        shapley_K=args.shapley_K,
                        agg_rule=strategy_name,
                    )
                else:
                    strategy = ShapeFlStrategy(
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
                    )

                elapsed = run_strategy(
                    strategy=strategy,
                    client_fn=shared["client_fn"],
                    num_clients=args.num_nodes,
                    num_rounds=strategy.total_flower_rounds,
                )
                metrics = strategy.metrics_history
                summary[combo_key][strategy_name] = {
                    "final_accuracy": metrics["accuracy"][-1] if metrics["accuracy"] else None,
                    "best_accuracy": max(metrics["accuracy"]) if metrics["accuracy"] else None,
                    "elapsed_seconds": elapsed,
                }

    write_summary_json(os.path.join(args.output_dir, "byzantine_sweep_results.json"), {"config": vars(args), "summary": summary})
    print(f"Byzantine sweep complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
