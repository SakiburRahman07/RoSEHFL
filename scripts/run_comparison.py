#!/usr/bin/env python3
"""
Unified multi-strategy comparison runner.
"""

from __future__ import annotations

import argparse
import os

from rosehfl.utils.seed import set_seed

try:
    from ._cli_args import add_common_experiment_args, add_multi_strategy_arg
    from ._comparison_plots import generate_comparison_package
    from ._experiment_bundle import (
        RUN_STATUS_FILENAME,
        build_comparison_payload,
        default_output_dir,
        ensure_dataset_defaults,
        initialise_run_bundle,
        load_json,
        load_strategy_result,
        run_strategy_bundle,
        strategy_dir,
        update_run_status,
        write_json,
    )
    from ._rose_common import prepare_shared_context
    from ._strategy_factory import build_strategy
except ImportError:
    from scripts._cli_args import add_common_experiment_args, add_multi_strategy_arg
    from scripts._comparison_plots import generate_comparison_package
    from scripts._experiment_bundle import (
        RUN_STATUS_FILENAME,
        build_comparison_payload,
        default_output_dir,
        ensure_dataset_defaults,
        initialise_run_bundle,
        load_json,
        load_strategy_result,
        run_strategy_bundle,
        strategy_dir,
        update_run_status,
        write_json,
    )
    from scripts._rose_common import prepare_shared_context
    from scripts._strategy_factory import build_strategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified strategy comparison runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_experiment_args(parser)
    add_multi_strategy_arg(parser, default=["shapefl", "rose_q1s"])
    parser.add_argument("--comparison-mode", type=str, default="effective", choices=["paper", "effective"])
    parser.add_argument("--plot-title", type=str, default=None)
    parser.add_argument("--plot-dpi", type=int, default=180)
    return parser


def _update_strategy_status(output_dir: str, strategy_name: str, *, state: str, strategy_output_dir: str, completed: bool | None = None) -> None:
    status = load_json(os.path.join(output_dir, RUN_STATUS_FILENAME), default={})
    strategy_runs = dict(status.get("strategy_runs", {}))
    entry = dict(strategy_runs.get(strategy_name, {}))
    entry["state"] = state
    entry["output_dir"] = strategy_output_dir
    if completed is not None:
        entry["completed"] = bool(completed)
    strategy_runs[strategy_name] = entry
    completed_strategies = list(status.get("completed_strategies", []))
    if completed and strategy_name not in completed_strategies:
        completed_strategies.append(strategy_name)
    update_run_status(
        output_dir,
        active_strategy=None if completed else strategy_name,
        strategy_runs=strategy_runs,
        completed_strategies=completed_strategies,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.no_augment:
        args.augment = False

    ensure_dataset_defaults(args)
    set_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = default_output_dir("comparison", args, list(args.strategies))
    os.makedirs(args.output_dir, exist_ok=True)
    initialise_run_bundle(args.output_dir, "comparison", args, list(args.strategies))

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

    all_results: dict[str, dict] = {}
    status = load_json(os.path.join(args.output_dir, RUN_STATUS_FILENAME), default={})
    completed = set(status.get("completed_strategies", [])) if args.resume else set()

    for strategy_name in args.strategies:
        strategy_output_dir = strategy_dir(args.output_dir, strategy_name)
        os.makedirs(strategy_output_dir, exist_ok=True)

        if strategy_name in completed:
            cached = load_strategy_result(strategy_output_dir)
            if cached is not None:
                all_results[strategy_name] = {
                    "metrics": cached["metrics"],
                    "elapsed_seconds": cached["elapsed_seconds"],
                }
                continue

        _update_strategy_status(
            args.output_dir,
            strategy_name,
            state="running",
            strategy_output_dir=strategy_output_dir,
            completed=False,
        )
        set_seed(args.seed)
        strategy = build_strategy(strategy_name, args, shared, strategy_output_dir)
        result = run_strategy_bundle(
            strategy_name=strategy_name,
            args=args,
            shared=shared,
            strategy=strategy,
            output_dir=strategy_output_dir,
            resume=args.resume,
            include_fairness=False,
        )
        all_results[strategy_name] = {
            "metrics": result["metrics"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
        _update_strategy_status(
            args.output_dir,
            strategy_name,
            state="completed",
            strategy_output_dir=strategy_output_dir,
            completed=True,
        )

    payload = build_comparison_payload(
        all_results,
        args=args,
        strategy_names=list(args.strategies),
        strategy_dirs={name: strategy_dir(args.output_dir, name) for name in args.strategies},
    )
    comparison_json_path = os.path.join(args.output_dir, "comparison_results.json")
    write_json(comparison_json_path, payload)
    update_run_status(args.output_dir, comparison_results_ready=True, active_strategy=None)

    generate_comparison_package(
        comparison_json_path,
        output_dir=args.output_dir,
        title=args.plot_title or "Strategy Comparison",
        dpi=args.plot_dpi,
    )
    update_run_status(
        args.output_dir,
        completed=True,
        comparison_results_ready=True,
        artifacts_generated=True,
        active_strategy=None,
    )

    print(f"Comparison complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
