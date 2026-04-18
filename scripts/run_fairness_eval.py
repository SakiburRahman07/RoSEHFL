#!/usr/bin/env python3
"""
Recompute fairness reports for existing RoSE/ShapeFL run directories.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

from flwr.common import ndarrays_to_parameters

from ._rose_common import prepare_shared_context, write_fairness_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute fairness metrics for completed runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_dirs", nargs="+", help="One or more completed run directories")
    parser.add_argument("--seed", type=int, default=None, help="Override the seed from config.json")
    return parser


def load_config(run_dir: str) -> dict:
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_checkpoint(run_dir: str) -> dict:
    checkpoint_path = os.path.join(run_dir, "checkpoint.pkl")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint.pkl in {run_dir}")
    with open(checkpoint_path, "rb") as handle:
        return pickle.load(handle)


def main() -> None:
    args = build_parser().parse_args()
    for run_dir in args.run_dirs:
        config = load_config(run_dir)
        checkpoint = load_checkpoint(run_dir)
        seed = args.seed if args.seed is not None else int(config.get("seed", 42))

        shared = prepare_shared_context(
            model_name=config["model"],
            dataset_name=config["dataset"],
            num_nodes=int(config["num_nodes"]),
            batch_size=int(config.get("batch_size", 32)),
            shard_size=int(config.get("shard_size", 15)),
            shards_per_node=int(config["shards_per_node"]),
            classes_per_node=int(config["classes_per_node"]),
            augment=bool(config.get("augment", False)),
            seed=seed,
            probe_size=int(config.get("probe_size", 1000)),
        )

        parameters = ndarrays_to_parameters(checkpoint["global_parameters"])
        report = write_fairness_report(
            output_dir=run_dir,
            parameters=parameters,
            model_factory=shared["model_factory"],
            test_dataset=shared["test_dataset"],
            fairness_partitions=shared["fairness_partitions"],
            server_device=shared["server_device"],
            seed=seed,
        )
        print(f"Fairness report written to {os.path.join(run_dir, 'fairness.json')}")
        print(f"  Mean accuracy: {report['mean_accuracy']:.4f}")


if __name__ == "__main__":
    main()

