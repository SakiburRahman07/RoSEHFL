"""
Shared CLI argument helpers for experiment scripts.
"""

from __future__ import annotations

import argparse

from ._strategy_factory import SUPPORTED_STRATEGIES


def add_common_experiment_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--gamma-anneal", type=str, default="cosine", choices=["fixed", "linear", "cosine", "adaptive"])
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

    parser.add_argument("--min-fit-clients", type=int, default=None,
                        help="Minimum clients required per round (default: all)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)


def add_single_strategy_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--strategy", type=str, default="rose", choices=SUPPORTED_STRATEGIES)


def add_multi_strategy_arg(parser: argparse.ArgumentParser, default: list[str] | None = None) -> None:
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=default or ["shapefl", "rose_q1s"],
        choices=SUPPORTED_STRATEGIES,
    )
