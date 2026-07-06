#!/usr/bin/env python3
"""Build a comparison package from two lab run directories.

Creates the same comparison_results.json and figures as run_comparison.py,
but sourced from real deployment data instead of simulation.

The lab JSON files have baseline/communication cost fields swapped.
This script corrects them before building the comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from scripts._comparison_plots import generate_comparison_package

LABS = {
    "shapefl": "results/final/lab_shapefl_fmnist_12pi_30k",
    "rosehfl": "results/final/lab_rosehfl_fmnist_12pi_30k",
}
OUTPUT_DIR = "results/final/lab_cmp_fmnist_12pi_30k"

METRIC_SWAP = [
    ("baseline_per_round_cost_gb", "communication_per_round_cost_gb"),
    ("baseline_cumulative_cost_gb", "communication_cumulative_cost_gb"),
]

SUMMARY_SWAP = [
    ("final_baseline_cost_gb", "final_communication_cost_gb"),
    ("baseline_per_round_cost_gb", "communication_per_round_cost_gb"),
    ("baseline_cost_to_target_gb", "communication_cost_to_target_gb"),
    ("baseline_cost_to_reference_accuracy_gb", "communication_cost_to_reference_accuracy_gb"),
]


def swap_fields(d: dict, swaps: list[tuple[str, str]]) -> dict:
    for a, b in swaps:
        if a in d and b in d:
            d[a], d[b] = d[b], d[a]
    return d


def recompute_savings(s: dict) -> dict:
    bl = s.get("final_baseline_cost_gb")
    cm = s.get("final_communication_cost_gb")
    if bl is not None and cm is not None:
        bl, cm = float(bl), float(cm)
        s["cost_savings_gb"] = 0.0 if abs(bl - cm) < 1e-12 else bl - cm
        s["cost_savings_pct"] = 0.0 if abs(bl) < 1e-12 else (1.0 - cm / bl) * 100.0
    return s


def load_lab(lab_dir: str) -> dict:
    with open(os.path.join(lab_dir, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(lab_dir, "summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(lab_dir, "deploy_config.json")) as f:
        deploy_config = json.load(f)

    # Fix swapped labels in per-round metrics only.
    # The summary.json may or may not have swapped labels depending on
    # strategy version.  Correct by checking: baseline (theoretical) should
    # be >= communication (measured with compression).
    metrics = swap_fields(metrics, METRIC_SWAP)

    bl = summary.get("final_baseline_cost_gb")
    cm = summary.get("final_communication_cost_gb")
    if bl is not None and cm is not None and float(bl) < float(cm):
        summary = swap_fields(summary, SUMMARY_SWAP)
        summary = recompute_savings(summary)

    return {
        "metrics": metrics,
        "elapsed_seconds": float(summary.get("elapsed_seconds", 0.0)),
        "summary": summary,
        "deploy_config": deploy_config,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    labs = {}
    for name, lab_dir in LABS.items():
        labs[name] = load_lab(lab_dir)

    # --- Build comparison payload manually (mirrors build_comparison_payload) ---
    strategy_names = list(labs.keys())
    first_deploy = list(labs.values())[0]["deploy_config"]
    target_accuracy = 0.75

    # common_budget = min of final communication costs
    final_comm_costs = []
    for lab in labs.values():
        cm = lab["summary"].get("final_communication_cost_gb")
        if cm is not None:
            final_comm_costs.append(float(cm))
    communication_common_budget = min(final_comm_costs) if final_comm_costs else 0.0

    # common_budget = min of final baseline costs
    final_bl_costs = []
    for lab in labs.values():
        bl = lab["summary"].get("final_baseline_cost_gb")
        if bl is not None:
            final_bl_costs.append(float(bl))
    baseline_common_budget = min(final_bl_costs) if final_bl_costs else 0.0

    # reference_accuracy = shapefl's final accuracy
    reference_accuracy = None
    if "shapefl" in labs:
        reference_accuracy = labs["shapefl"]["summary"].get("final_accuracy")

    # Per-strategy summaries
    strategy_summaries = {}
    for name, lab in labs.items():
        s = dict(lab["summary"])  # copy
        s["strategy"] = name
        # Ensure required fields
        s.setdefault("final_accuracy", lab["metrics"].get("accuracy", [None])[-1])
        s.setdefault("best_accuracy", max(lab["metrics"].get("accuracy", [0]), default=None))
        s.setdefault("final_loss", lab["metrics"].get("loss", [None])[-1])
        s.setdefault("elapsed_seconds", lab["elapsed_seconds"])
        strategy_summaries[name] = s

    # summary (per-strategy at common budget)
    summary = {}
    for name, lab in labs.items():
        acc = lab["metrics"].get("accuracy", [])
        cum = lab["metrics"].get("communication_cumulative_cost_gb", [])
        acc_at_budget = None
        for cost, a in zip(cum, acc):
            if float(cost) >= communication_common_budget:
                acc_at_budget = float(a)
                break
        if acc_at_budget is None and acc:
            acc_at_budget = float(acc[-1])

        summary[name] = {
            "final_accuracy": lab["summary"].get("final_accuracy"),
            "best_accuracy": lab["summary"].get("best_accuracy"),
            "cost_to_target_gb": lab["summary"].get("communication_cost_to_target_gb"),
            "rounds_to_target": lab["summary"].get("rounds_to_target"),
            "accuracy_at_common_budget": acc_at_budget,
            "per_round_cost_gb": lab["summary"].get("communication_per_round_cost_gb"),
            "elapsed_seconds": lab["elapsed_seconds"],
            "cost_mode": "effective",
        }

    payload = {
        "config": {
            "model": first_deploy.get("model", "lenet5"),
            "dataset": first_deploy.get("dataset", "fmnist"),
            "num_nodes": first_deploy.get("num_nodes", 12),
            "topology": first_deploy.get("topology", "geant2010"),
            "source": "lab_deployment",
            "machine": first_deploy.get("machine", "unknown"),
        },
        "comparison_mode": "effective",
        "strategy_names": strategy_names,
        "strategy_dirs": {name: os.path.abspath(LABS[name]) for name in strategy_names},
        "common_budget_gb": communication_common_budget,
        "baseline_common_budget_gb": baseline_common_budget,
        "communication_common_budget_gb": communication_common_budget,
        "summary": summary,
        "baseline_summary": summary,
        "communication_summary": summary,
        "strategy_summaries": strategy_summaries,
        "reference_accuracy": reference_accuracy,
        "per_round_metrics": {name: labs[name]["metrics"] for name in strategy_names},
    }

    comparison_json_path = os.path.join(args.output_dir, "comparison_results.json")
    with open(comparison_json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Written: {comparison_json_path}")

    # Generate figures
    generate_comparison_package(
        comparison_json_path,
        output_dir=args.output_dir,
        title="Lab Deployment: RoSE-HFL vs ShapeFL (12 Pi, 30 rounds)",
        dpi=args.dpi,
    )
    print(f"All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
