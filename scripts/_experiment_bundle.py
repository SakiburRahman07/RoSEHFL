"""
Shared experiment packaging, resume, and summary helpers.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rosehfl.data.data_loader import DATASET_INFO
from rosehfl.utils.json_utils import NumpyEncoder

from ._rose_common import (
    load_checkpoint_if_available,
    run_strategy,
    timestamped_dir,
    write_fairness_report,
    write_summary_json,
)
from ._strategy_factory import default_target_accuracy, display_name


RUN_STATUS_FILENAME = "run_status.json"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip().lower())
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "run"


def _strategy_slug(strategy_names: list[str], max_len: int = 72) -> str:
    raw = "-".join(_safe_token(name) for name in strategy_names)
    if len(raw) <= max_len:
        return raw
    return raw[:max_len].rstrip("-_")


def default_output_dir(run_type: str, args, strategy_names: list[str]) -> str:
    prefix = "sim" if run_type == "simulation" else "cmp"
    strategy_part = _strategy_slug(strategy_names)
    label = (
        f"{prefix}_{strategy_part}_{_safe_token(args.model)}_{_safe_token(args.dataset)}"
        f"_{_safe_token(args.topology)}_n{int(args.num_nodes)}_seed{int(args.seed)}"
    )
    return os.path.join("results", timestamped_dir(label))


def ensure_dataset_defaults(args) -> dict[str, Any]:
    ds_info = DATASET_INFO[args.dataset]
    if getattr(args, "shards_per_node", None) is None:
        args.shards_per_node = ds_info["shards_per_node"]
    if getattr(args, "classes_per_node", None) is None:
        args.classes_per_node = ds_info["classes_per_node"]
    if getattr(args, "B_e", None) is None:
        args.B_e = max(3, -(-int(args.num_nodes) // 3))
    if getattr(args, "target_accuracy", None) is None:
        args.target_accuracy = default_target_accuracy(args.dataset)
    return ds_info


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, cls=NumpyEncoder)


def build_metadata(run_type: str, args, strategy_names: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_type": run_type,
        "created_at": utc_timestamp(),
        "command": [sys.executable, *sys.argv],
        "strategy_names": list(strategy_names),
        "config": vars(args),
    }


def initialise_run_bundle(output_dir: str, run_type: str, args, strategy_names: list[str]) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    metadata = build_metadata(run_type, args, strategy_names)
    metadata_path = os.path.join(output_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        write_json(metadata_path, metadata)
    status_path = os.path.join(output_dir, RUN_STATUS_FILENAME)
    status = load_json(status_path)
    if status is None:
        status = {
            "run_type": run_type,
            "created_at": utc_timestamp(),
            "updated_at": utc_timestamp(),
            "completed": False,
            "active_strategy": strategy_names[0] if run_type == "simulation" and strategy_names else None,
            "strategy_names": list(strategy_names),
            "completed_strategies": [],
            "strategy_runs": {},
            "artifacts_generated": False,
            "comparison_results_ready": False,
        }
        write_json(status_path, status)
    return metadata


def update_run_status(output_dir: str, **changes) -> dict[str, Any]:
    status_path = os.path.join(output_dir, RUN_STATUS_FILENAME)
    status = load_json(status_path, default={})
    status.update(changes)
    status["updated_at"] = utc_timestamp()
    write_json(status_path, status)
    return status


def _cost_series(metrics: dict[str, Any], mode: str, field: str) -> list[float]:
    if mode == "paper":
        values = metrics.get(f"paper_{field}")
        if values:
            return [float(value) for value in values]
    if mode == "effective":
        values = metrics.get(f"effective_{field}")
        if values:
            return [float(value) for value in values]
    legacy = metrics.get(field, [])
    return [float(value) for value in legacy]


def accuracy_at_budget(metrics: dict[str, Any], budget_gb: float, *, mode: str) -> float | None:
    for cost, accuracy in zip(_cost_series(metrics, mode, "cumulative_cost_gb"), metrics.get("accuracy", [])):
        if float(cost) >= float(budget_gb):
            return float(accuracy)
    return float(metrics["accuracy"][-1]) if metrics.get("accuracy") else None


def cost_to_target(metrics: dict[str, Any], target_accuracy: float | None, *, mode: str) -> float | None:
    if target_accuracy is None:
        return None
    for cost, accuracy in zip(_cost_series(metrics, mode, "cumulative_cost_gb"), metrics.get("accuracy", [])):
        if float(accuracy) >= float(target_accuracy):
            return float(cost)
    return None


def round_to_target(metrics: dict[str, Any], target_accuracy: float | None) -> int | None:
    if target_accuracy is None:
        return None
    for round_id, accuracy in zip(metrics.get("cloud_round", []), metrics.get("accuracy", [])):
        if float(accuracy) >= float(target_accuracy):
            return int(round_id)
    return None


def per_round_cost(metrics: dict[str, Any], *, mode: str) -> float | None:
    series = _cost_series(metrics, mode, "per_round_cost_gb")
    if not series:
        return None
    return float(series[-1])


def final_metric(metrics: dict[str, Any], field: str) -> float | None:
    values = metrics.get(field, [])
    if not values:
        return None
    return float(values[-1])


def build_strategy_summary(
    strategy_name: str,
    metrics: dict[str, Any],
    *,
    elapsed_seconds: float,
    target_accuracy: float | None,
    common_budgets: dict[str, float] | None = None,
    compare_accuracy: float | None = None,
) -> dict[str, Any]:
    common_budgets = common_budgets or {}
    summary = {
        "strategy": strategy_name,
        "display_name": display_name(strategy_name),
        "elapsed_seconds": float(elapsed_seconds),
        "final_accuracy": final_metric(metrics, "accuracy"),
        "best_accuracy": max([float(value) for value in metrics.get("accuracy", [])], default=None),
        "final_loss": final_metric(metrics, "loss"),
        "final_paper_cost_gb": final_metric(metrics, "paper_cumulative_cost_gb") or final_metric(metrics, "cumulative_cost_gb"),
        "final_effective_cost_gb": final_metric(metrics, "effective_cumulative_cost_gb") or final_metric(metrics, "cumulative_cost_gb"),
        "paper_per_round_cost_gb": per_round_cost(metrics, mode="paper"),
        "effective_per_round_cost_gb": per_round_cost(metrics, mode="effective"),
        "paper_cost_to_target_gb": cost_to_target(metrics, target_accuracy, mode="paper"),
        "effective_cost_to_target_gb": cost_to_target(metrics, target_accuracy, mode="effective"),
        "rounds_to_target": round_to_target(metrics, target_accuracy),
        "paper_accuracy_at_common_budget": (
            accuracy_at_budget(metrics, common_budgets["paper"], mode="paper")
            if "paper" in common_budgets
            else None
        ),
        "effective_accuracy_at_common_budget": (
            accuracy_at_budget(metrics, common_budgets["effective"], mode="effective")
            if "effective" in common_budgets
            else None
        ),
        "total_model_payload_bytes": int(sum(metrics.get("model_payload_bytes", []))),
        "total_probe_payload_bytes": int(sum(metrics.get("probe_payload_bytes", []))),
    }
    if summary["final_paper_cost_gb"] is not None and summary["final_effective_cost_gb"] is not None and summary["final_paper_cost_gb"] > 0:
        raw_delta = float(summary["final_paper_cost_gb"]) - float(summary["final_effective_cost_gb"])
        pct_delta = (1.0 - (float(summary["final_effective_cost_gb"]) / float(summary["final_paper_cost_gb"]))) * 100.0
        summary["cost_savings_gb"] = 0.0 if abs(raw_delta) < 1e-12 else raw_delta
        summary["cost_savings_pct"] = 0.0 if abs(pct_delta) < 1e-12 else pct_delta
    else:
        summary["cost_savings_gb"] = None
        summary["cost_savings_pct"] = None
    if compare_accuracy is not None:
        summary["effective_cost_to_reference_accuracy_gb"] = cost_to_target(metrics, compare_accuracy, mode="effective")
        summary["paper_cost_to_reference_accuracy_gb"] = cost_to_target(metrics, compare_accuracy, mode="paper")
    else:
        summary["effective_cost_to_reference_accuracy_gb"] = None
        summary["paper_cost_to_reference_accuracy_gb"] = None
    return summary


def build_mode_summary(
    all_results: dict[str, dict[str, Any]],
    *,
    mode: str,
    target_accuracy: float | None,
    budget_gb: float | None,
) -> tuple[float, dict[str, dict[str, Any]]]:
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
            "final_accuracy": final_metric(metrics, "accuracy"),
            "best_accuracy": max([float(value) for value in metrics.get("accuracy", [])], default=None),
            "cost_to_target_gb": cost_to_target(metrics, target_accuracy, mode=mode),
            "rounds_to_target": round_to_target(metrics, target_accuracy),
            "accuracy_at_common_budget": accuracy_at_budget(metrics, common_budget, mode=mode),
            "per_round_cost_gb": per_round_cost(metrics, mode=mode),
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "cost_mode": mode,
        }
    return float(common_budget), summary


def build_comparison_payload(
    all_results: dict[str, dict[str, Any]],
    *,
    args,
    strategy_names: list[str],
    strategy_dirs: dict[str, str] | None = None,
) -> dict[str, Any]:
    paper_common_budget, paper_summary = build_mode_summary(
        all_results,
        mode="paper",
        target_accuracy=args.target_accuracy,
        budget_gb=getattr(args, "budget_gb", None),
    )
    effective_common_budget, effective_summary = build_mode_summary(
        all_results,
        mode="effective",
        target_accuracy=args.target_accuracy,
        budget_gb=getattr(args, "budget_gb", None),
    )
    comparison_mode = getattr(args, "comparison_mode", "effective")
    selected_common_budget = effective_common_budget if comparison_mode == "effective" else paper_common_budget
    selected_summary = effective_summary if comparison_mode == "effective" else paper_summary

    reference_accuracy = None
    if "shapefl" in all_results:
        reference_accuracy = final_metric(all_results["shapefl"]["metrics"], "accuracy")

    strategy_summaries = {}
    for strategy_name, result in all_results.items():
        strategy_summaries[strategy_name] = build_strategy_summary(
            strategy_name,
            result["metrics"],
            elapsed_seconds=result["elapsed_seconds"],
            target_accuracy=args.target_accuracy,
            common_budgets={"paper": paper_common_budget, "effective": effective_common_budget},
            compare_accuracy=reference_accuracy,
        )

    return {
        "config": vars(args),
        "comparison_mode": comparison_mode,
        "strategy_names": list(strategy_names),
        "strategy_dirs": dict(strategy_dirs or {}),
        "common_budget_gb": selected_common_budget,
        "paper_common_budget_gb": paper_common_budget,
        "effective_common_budget_gb": effective_common_budget,
        "summary": selected_summary,
        "paper_summary": paper_summary,
        "effective_summary": effective_summary,
        "strategy_summaries": strategy_summaries,
        "reference_accuracy": reference_accuracy,
        "per_round_metrics": {name: result["metrics"] for name, result in all_results.items()},
    }


def attach_strategy_output_dir(strategy, output_dir: str) -> None:
    strategy.output_dir = output_dir


def load_strategy_result(strategy_dir: str) -> dict[str, Any] | None:
    metrics = load_json(os.path.join(strategy_dir, "metrics.json"))
    summary = load_json(os.path.join(strategy_dir, "summary.json"))
    if metrics is None or summary is None:
        return None
    return {
        "metrics": metrics,
        "elapsed_seconds": float(summary.get("elapsed_seconds", 0.0)),
        "summary": summary,
    }


def finalise_strategy_run(
    *,
    output_dir: str,
    strategy_name: str,
    args,
    strategy,
    shared: dict[str, Any],
    elapsed_seconds: float,
    include_fairness: bool,
) -> dict[str, Any]:
    if include_fairness:
        fairness_report = write_fairness_report(
            output_dir=output_dir,
            parameters=strategy.global_parameters,
            model_factory=shared["model_factory"],
            test_dataset=shared["test_dataset"],
            fairness_partitions=shared["fairness_partitions"],
            server_device=shared["server_device"],
            seed=args.seed,
        )
    else:
        fairness_report = None

    if hasattr(strategy, "_persist_artifacts"):
        strategy._persist_artifacts(completed=True)

    metrics = strategy.metrics_history
    summary = build_strategy_summary(
        strategy_name,
        metrics,
        elapsed_seconds=elapsed_seconds,
        target_accuracy=args.target_accuracy,
    )
    if fairness_report is not None:
        summary["fairness"] = fairness_report

    metadata = {
        "strategy_name": strategy_name,
        "display_name": display_name(strategy_name),
        "completed_at": utc_timestamp(),
        "config": vars(args),
    }
    write_json(os.path.join(output_dir, "metadata.json"), metadata)
    write_json(os.path.join(output_dir, "summary.json"), summary)
    return {
        "metrics": metrics,
        "elapsed_seconds": float(elapsed_seconds),
        "summary": summary,
    }


def run_strategy_bundle(
    *,
    strategy_name: str,
    args,
    shared: dict[str, Any],
    strategy,
    output_dir: str,
    resume: bool,
    include_fairness: bool = False,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    attach_strategy_output_dir(strategy, output_dir)

    # Set partition hash for checkpoint integrity verification
    if hasattr(strategy, "set_partitions") and "partitions" in shared:
        strategy.set_partitions(shared["partitions"], seed=getattr(args, "seed", None))

    prior_summary = load_json(os.path.join(output_dir, "summary.json"), default={}) or {}

    if resume and hasattr(strategy, "load_checkpoint_state"):
        checkpoint = load_checkpoint_if_available(output_dir)
        if checkpoint is not None:
            strategy.load_checkpoint_state(checkpoint)

    strategy_metadata_path = os.path.join(output_dir, "metadata.json")
    if not os.path.isfile(strategy_metadata_path):
        write_json(
            strategy_metadata_path,
            {
                "strategy_name": strategy_name,
                "display_name": display_name(strategy_name),
                "config": vars(args),
                "started_at": utc_timestamp(),
            },
        )

    remaining_rounds = getattr(strategy, "remaining_flower_rounds", None)
    if remaining_rounds is None:
        completed_rounds = int(getattr(strategy, "completed_flower_rounds", 0))
        remaining_rounds = max(0, int(strategy.total_flower_rounds) - completed_rounds)

    elapsed_seconds = 0.0
    if remaining_rounds > 0:
        elapsed_seconds = run_strategy(
            strategy=strategy,
            client_fn=shared["client_fn"],
            num_clients=args.num_nodes,
            num_rounds=int(remaining_rounds),
        )
    elapsed_seconds += float(prior_summary.get("elapsed_seconds", 0.0))

    return finalise_strategy_run(
        output_dir=output_dir,
        strategy_name=strategy_name,
        args=args,
        strategy=strategy,
        shared=shared,
        elapsed_seconds=elapsed_seconds,
        include_fairness=include_fairness,
    )


def strategy_dir(root_output_dir: str, strategy_name: str) -> str:
    return os.path.join(root_output_dir, "strategies", strategy_name)
