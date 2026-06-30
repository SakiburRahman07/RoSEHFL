#!/usr/bin/env python3
"""
Unified single-strategy simulation runner.
"""

from __future__ import annotations

import argparse
import os

from rosehfl.utils.seed import set_seed

try:
    from ._cli_args import add_common_experiment_args, add_single_strategy_arg
    from ._experiment_bundle import (
        RUN_STATUS_FILENAME,
        default_output_dir,
        ensure_dataset_defaults,
        initialise_run_bundle,
        run_strategy_bundle,
        update_run_status,
        write_json,
    )
    from ._rose_common import prepare_shared_context
    from ._strategy_factory import build_strategy
except ImportError:
    from scripts._cli_args import add_common_experiment_args, add_single_strategy_arg
    from scripts._experiment_bundle import (
        RUN_STATUS_FILENAME,
        default_output_dir,
        ensure_dataset_defaults,
        initialise_run_bundle,
        run_strategy_bundle,
        update_run_status,
        write_json,
    )
    from scripts._rose_common import prepare_shared_context
    from scripts._strategy_factory import build_strategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified strategy simulation runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_experiment_args(parser)
    add_single_strategy_arg(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.no_augment:
        args.augment = False

    ensure_dataset_defaults(args)
    set_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = default_output_dir("simulation", args, [args.strategy])
    os.makedirs(args.output_dir, exist_ok=True)
    initialise_run_bundle(args.output_dir, "simulation", args, [args.strategy])
    update_run_status(args.output_dir, active_strategy=args.strategy, completed=False, artifacts_generated=False)

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

    strategy = build_strategy(args.strategy, args, shared, args.output_dir)
    result = run_strategy_bundle(
        strategy_name=args.strategy,
        args=args,
        shared=shared,
        strategy=strategy,
        output_dir=args.output_dir,
        resume=args.resume,
        include_fairness=True,
    )

    payload = {
        "config": vars(args),
        "strategy_name": args.strategy,
        "summary": result["summary"],
        "metrics": result["metrics"],
        "edge_nodes": {
            str(edge_id): sorted(nodes)
            for edge_id, nodes in getattr(strategy, "edge_nodes", {}).items()
        },
    }
    write_json(os.path.join(args.output_dir, "simulation_results.json"), payload)
    update_run_status(
        args.output_dir,
        completed=True,
        artifacts_generated=True,
        active_strategy=None,
        completed_strategies=[args.strategy],
        strategy_runs={
            args.strategy: {
                "state": "completed",
                "output_dir": args.output_dir,
                "completed": True,
            }
        },
    )

    try:
        from rosehfl.utils.visualization import visualize_simulation

        visualize_simulation(
            metrics=result["metrics"],
            config=vars(args),
            edge_nodes=payload["edge_nodes"],
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"[Warning] Simulation visualization failed: {exc}")

    summary = result["summary"]
    print("Simulation complete")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Run status: {os.path.join(args.output_dir, RUN_STATUS_FILENAME)}")
    if summary.get("final_accuracy") is not None:
        print(f"  Final accuracy: {float(summary['final_accuracy']) * 100.0:.2f}%")
        print(f"  Best accuracy:  {float(summary['best_accuracy']) * 100.0:.2f}%")


if __name__ == "__main__":
    main()
