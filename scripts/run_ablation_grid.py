#!/usr/bin/env python3
"""
RoSE++ ablation runner.
"""

from __future__ import annotations

import argparse
import math
import os

from rosehfl.strategy import RoSEHFLStrategy

from ._rose_common import prepare_shared_context, run_strategy, timestamped_dir, write_summary_json


ABLATION_CELLS = [
    ("rose_baseline", False, False, False, False, False),
    ("plus_c1", True, False, False, False, False),
    ("plus_c1_c2", True, True, False, False, False),
    ("plus_c1_c2_c3", True, True, True, False, False),
    ("plus_c1_c2_c3_c4", True, True, True, True, False),
    ("roseplusplus_full", True, True, True, True, True),
    ("full_minus_c3", True, True, False, True, True),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RoSE++ ablation grid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="mobilenetv2", choices=["lenet5", "mobilenetv2", "resnet18"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["fmnist", "cifar10", "cifar100"])
    parser.add_argument("--num-nodes", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=1)
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
        args.output_dir = os.path.join("results", timestamped_dir(f"rose_ablation_{args.model}_{args.dataset}"))
    os.makedirs(args.output_dir, exist_ok=True)

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
    )

    summary = {}
    for cell_name, use_c1, use_c2, use_c3, use_c4, use_c5 in ABLATION_CELLS:
        output_dir = os.path.join(args.output_dir, cell_name)
        os.makedirs(output_dir, exist_ok=True)
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
            gamma_anneal="adaptive" if use_c2 else "cosine",
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
            output_dir=output_dir,
            seed=args.seed,
            shapley_T=1 if use_c1 else args.shapley_T,
            shapley_K=64 if use_c1 else args.shapley_K,
            planning_signal="hybrid" if use_c1 else "shapley",
            agg_rule="trust" if use_c4 else "trust_legacy",
            trust_use_shrinkage=use_c4,
            adaptive_gamma_eta=0.5,
            adaptive_gamma_target=0.25,
            warm_start_replan=use_c2,
            warm_start_threshold=0.05 if use_c2 else 0.0,
            replan_cost_increase_tolerance=0.1,
            local_objective_prox_mu=0.01 if use_c3 else 0.0,
            logit_adjustment_tau=1.0 if use_c3 else 0.0,
            local_bn=use_c5,
            edge_swa_k=3 if use_c5 else 1,
        )

        elapsed = run_strategy(
            strategy=strategy,
            client_fn=shared["client_fn"],
            num_clients=args.num_nodes,
            num_rounds=strategy.total_flower_rounds,
        )
        metrics = strategy.metrics_history
        summary[cell_name] = {
            "final_accuracy": metrics["accuracy"][-1] if metrics["accuracy"] else None,
            "best_accuracy": max(metrics["accuracy"]) if metrics["accuracy"] else None,
            "elapsed_seconds": elapsed,
            "c1_hybrid_phi": use_c1,
            "c2_adaptive_gamma": use_c2,
            "c3_balanced_softmax_fedprox": use_c3,
            "c4_shrinkage_trust": use_c4,
            "c5_local_bn_swa": use_c5,
        }

    write_summary_json(
        os.path.join(args.output_dir, "ablation_results.json"),
        {"config": vars(args), "summary": summary},
    )
    print(f"Ablation grid complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
