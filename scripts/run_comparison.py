#!/usr/bin/env python3
"""
ShapeFL Flower Strategy Comparison
===================================
Compare ShapeFL, SHARE, Cost First, Data First, Random, FedAvg, and FedProx strategies.
Reproduces paper Fig. 11.

Usage (from project root):
    python -m scripts.run_comparison
    python -m scripts.run_comparison --kappa 30 --target-accuracy 0.60
"""

import argparse
import math
import os
import json
import time
import logging

import numpy as np
import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters
from torch.utils.data import DataLoader

from shapefl.models.factory import get_model, get_model_size
from shapefl.data.data_loader import (
    load_data,
    create_non_iid_partitions,
    get_partition_label_counts,
    DATASET_INFO,
)
from shapefl.client import client_fn_factory
from shapefl.strategy import (
    ShapeFlStrategy,
    FedAvgFlatStrategy,
    FedProxFlatStrategy,
    generate_communication_costs,
)
from shapefl.utils.seed import set_seed
from shapefl.utils.json_utils import NumpyEncoder


def run_one_strategy(name, strategy, client_fn, num_nodes, seed):
    set_seed(seed)
    num_rounds = strategy.total_flower_rounds
    print(f"\n{'=' * 60}")
    print(f"  STRATEGY: {name}  ({num_rounds} Flower rounds)")
    print(f"{'=' * 60}")
    start = time.time()
    fl.simulation.start_simulation(
        client_fn=client_fn, num_clients=num_nodes,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy, client_resources={"num_cpus": 1},
    )
    return strategy.metrics_history, time.time() - start


def main():
    parser = argparse.ArgumentParser(
        description="ShapeFL Flower Strategy Comparison (Fig. 11)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="lenet5",
                        choices=["lenet5", "mobilenetv2", "resnet18"])
    parser.add_argument("--dataset", type=str, default="fmnist",
                        choices=["fmnist", "cifar10", "cifar100"])
    parser.add_argument("--num-nodes", type=int, default=30)

    parser.add_argument("--kappa-p", type=int, default=30)
    parser.add_argument("--kappa-e", type=int, default=1)
    parser.add_argument("--kappa-c", type=int, default=10)
    parser.add_argument("--kappa", type=int, default=50)
    parser.add_argument("--total-local-epochs", type=int, default=None,
                        help="Exact local-epoch budget per node for fixed-budget reproduction runs.")
    parser.add_argument("--gamma", type=float, default=2800.0)
    parser.add_argument("--B-e", type=int, default=None)
    parser.add_argument("--T-max", type=int, default=30)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="SGD momentum (paper uses 0; set 0.9 for faster convergence).")
    parser.add_argument("--fedprox-mu", type=float, default=0.01,
                        help="FedProx proximal coefficient mu. Tune this if reproducing the paper baseline.")
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--shard-size", type=int, default=15)
    parser.add_argument("--shards-per-node", type=int, default=None)
    parser.add_argument("--classes-per-node", type=int, default=None)

    parser.add_argument("--augment", action="store_true",
                        help="Enable training-time data augmentation for CIFAR datasets.")
    parser.add_argument("--no-augment", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--target-accuracy", type=float, default=0.70)
    parser.add_argument(
        "--strategies", type=str, nargs="+",
        default=["shapefl", "cost_first", "data_first", "random", "fedavg"],
        choices=["shapefl", "share", "cost_first", "data_first", "random", "fedavg", "fedprox"],
    )
    parser.add_argument("--topology", type=str, default="geant2010",
                        choices=["geant2010", "uunet", "tinet", "viatel", "random"],
                        help="Network topology for communication costs (paper: geant2010/uunet/tinet, robust: viatel).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory. Auto-generated from config + timestamp if omitted.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.no_augment:
        args.augment = False

    # ── Auto-generate unique output directory ─────────────────────────
    if args.output_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            f"results/comparison_{args.model}_{args.dataset}"
            f"_n{args.num_nodes}_k{args.kappa}_{ts}"
        )
    # Warn if output directory already has results
    _existing = os.path.join(args.output_dir, "flower_comparison_results.json")
    if os.path.isfile(_existing):
        print(f"  WARNING: {_existing} already exists and will be overwritten.")

    ds_info = DATASET_INFO[args.dataset]
    if args.shards_per_node is None:
        args.shards_per_node = ds_info["shards_per_node"]
    if args.classes_per_node is None:
        args.classes_per_node = ds_info["classes_per_node"]
    if args.B_e is None:
        args.B_e = max(3, math.ceil(args.num_nodes / 3))

    set_seed(args.seed)
    # Server-side evaluation can use GPU; simulation clients must use CPU
    # because Ray spawns many parallel worker processes that would all
    # compete for the same CUDA device and trigger CUDA busy errors.
    server_device = "cuda" if torch.cuda.is_available() else "cpu"
    client_device = "cpu"
    device = server_device  # kept for backwards-compat references below

    model = get_model(args.model, ds_info["num_classes"], ds_info["input_channels"], server_device)
    num_params, size_mb = get_model_size(model)
    initial_ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    initial_params = ndarrays_to_parameters(initial_ndarrays)

    train_dataset, test_dataset = load_data(args.dataset, augment=args.augment)
    partitions = create_non_iid_partitions(
        train_dataset, args.num_nodes, args.shard_size,
        args.shards_per_node, args.classes_per_node, seed=42,
    )
    node_label_counts = get_partition_label_counts(
        train_dataset, partitions, ds_info["num_classes"],
    )
    client_fn = client_fn_factory(
        args.model, args.dataset, train_dataset, test_dataset,
        partitions, args.batch_size, client_device,
    )

    # ── Centralized evaluation function (server-side) ─────────────────
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    def evaluate_fn(server_round, parameters_ndarrays, config):
        """Evaluate the global model on the full test set (server-side)."""
        eval_model = get_model(
            args.model, ds_info["num_classes"], ds_info["input_channels"], device,
        )
        keys = list(eval_model.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters_ndarrays)}
        eval_model.load_state_dict(state_dict, strict=True)
        eval_model.to(device)
        eval_model.eval()

        total_loss, correct, total = 0.0, 0, 0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = eval_model(images)
                total_loss += criterion(outputs, labels).item() * labels.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, {"accuracy": accuracy}

    # Use full state_dict size (includes BatchNorm buffers) — must match
    # the S_m used inside ShapeFlStrategy._aggregate_pretrain so that
    # FedAvg and HFL strategies compute communication costs consistently.
    model_size_bytes = sum(w.nbytes for w in initial_ndarrays)
    c_ne, c_ec = generate_communication_costs(args.num_nodes, model_size_bytes, topology=args.topology)

    print("\n" + "=" * 70)
    print("  ShapeFL Flower Strategy Comparison")
    print("=" * 70)
    print(f"  Model: {args.model} ({num_params:,} params)")
    print(f"  Dataset: {args.dataset}  Nodes: {args.num_nodes}")
    print(f"  κ_e={args.kappa_e} κ_c={args.kappa_c} κ={args.kappa}")
    if args.total_local_epochs is not None:
        print(f"  Total local epochs per node: {args.total_local_epochs}")
    print(f"  Strategies: {args.strategies}")

    STRATEGY_DISPLAY = {
        "shapefl": "ShapeFL", "share": "SHARE", "cost_first": "Cost First",
        "data_first": "Data First", "random": "Random",
        "fedavg": "FedAvg", "fedprox": "FedProx",
    }

    strategies = {}
    for s in args.strategies:
        if s == "fedavg":
            strat = FedAvgFlatStrategy(
                num_nodes=args.num_nodes, kappa=args.kappa,
                local_epochs=args.kappa_c * args.kappa_e,
                lr=args.lr, momentum=args.momentum,
                total_local_epochs=args.total_local_epochs,
                initial_parameters=initial_params,
                evaluate_fn=evaluate_fn,
            )
            strat.set_comm_costs(c_ec)
        elif s == "fedprox":
            strat = FedProxFlatStrategy(
                num_nodes=args.num_nodes, kappa=args.kappa,
                local_epochs=args.kappa_c * args.kappa_e,
                lr=args.lr, momentum=args.momentum, prox_mu=args.fedprox_mu,
                total_local_epochs=args.total_local_epochs,
                initial_parameters=initial_params,
                evaluate_fn=evaluate_fn,
            )
            strat.set_comm_costs(c_ec)
        else:
            strat = ShapeFlStrategy(
                model_name=args.model, dataset_name=args.dataset,
                num_nodes=args.num_nodes, kappa_p=args.kappa_p,
                kappa_e=args.kappa_e, kappa_c=args.kappa_c, kappa=args.kappa,
                gamma=args.gamma, B_e=args.B_e, T_max=args.T_max, lr=args.lr,
                momentum=args.momentum,
                initial_parameters=initial_params, planning_mode=s,
                topology=args.topology, evaluate_fn=evaluate_fn,
                node_label_counts=node_label_counts,
                total_local_epochs=args.total_local_epochs,
            )
        strategies[STRATEGY_DISPLAY[s]] = strat

    all_results = {}
    for name, strat in strategies.items():
        metrics, elapsed = run_one_strategy(name, strat, client_fn, args.num_nodes, args.seed)
        all_results[name] = {"metrics": metrics, "time": elapsed}

    target = args.target_accuracy
    print("\n" + "=" * 70)
    print("  COMPARISON RESULTS")
    print("=" * 70)
    print(
        f"\n{'Strategy':<14} {'Final Acc':>10} {'Best Acc':>10} "
        f"{'Per-Round':>14} "
        f"{'Cost@' + str(int(target * 100)) + '%':>14} "
        f"{'Rounds@' + str(int(target * 100)) + '%':>12} {'Time':>8}"
    )
    print("-" * 84)

    summary = {}
    for name, res in all_results.items():
        h = res["metrics"]
        if not h["accuracy"]:
            print(f"{name:<14}  (no evaluation data)")
            continue

        final_acc = h["accuracy"][-1]
        best_acc = max(h["accuracy"])
        per_round = h["per_round_cost_gb"][0] if h["per_round_cost_gb"] else 0

        cost_at_target, rounds_at_target = None, None
        for i, acc in enumerate(h["accuracy"]):
            if acc >= target:
                cost_at_target = h["cumulative_cost_gb"][i]
                rounds_at_target = h["cloud_round"][i]
                break

        cost_str = f"{cost_at_target:.4f} GB" if cost_at_target else "NOT REACHED"
        rnd_str = str(rounds_at_target) if rounds_at_target else "-"
        t = res["time"]

        print(
            f"{name:<14} {final_acc * 100:>9.2f}% {best_acc * 100:>9.2f}% "
            f"{per_round:>12.6f}GB {cost_str:>14} "
            f"{rnd_str:>12} {t:>7.1f}s"
        )
        summary[name] = {
            "final_accuracy": final_acc, "best_accuracy": best_acc,
            "per_round_cost_gb": per_round, "cost_to_target_gb": cost_at_target,
            "rounds_to_target": rounds_at_target, "time_seconds": t,
        }

    if "ShapeFL" in summary and summary["ShapeFL"].get("cost_to_target_gb"):
        sc = summary["ShapeFL"]["cost_to_target_gb"]
        print(f"\n--- Cost savings (ShapeFL vs baselines to reach {target * 100:.0f}%) ---")
        for name, s in summary.items():
            if name == "ShapeFL":
                continue
            if s.get("cost_to_target_gb"):
                saving = (1 - sc / s["cost_to_target_gb"]) * 100
                print(f"  vs {name:<12}: {saving:+.1f}% {'(saved)' if saving > 0 else '(worse)'}")
            else:
                print(f"  vs {name:<12}: baseline did NOT reach {target * 100:.0f}%")

    os.makedirs(args.output_dir, exist_ok=True)

    output = {
        "config": {k: v for k, v in vars(args).items() if k != "strategies"},
        "strategies_run": args.strategies,
        "summary": summary,
        "per_round_metrics": {n: r["metrics"] for n, r in all_results.items()},
    }
    path = os.path.join(args.output_dir, "flower_comparison_results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {path}")

    # ── Visualization ────────────────────────────────────────────────────
    try:
        from shapefl.utils.visualization import visualize_comparison as viz_cmp
        all_metrics_for_viz = {n: r["metrics"] for n, r in all_results.items()}
        viz_cmp(
            all_metrics=all_metrics_for_viz,
            summary=summary,
            config={k: v for k, v in vars(args).items() if k != "strategies"},
            target_accuracy=target,
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"[Warning] Visualization failed: {e}")


if __name__ == "__main__":
    main()
