#!/usr/bin/env python3
"""
Visualize saved results
========================================
Re-generate all plots and HTML reports from an existing JSON results file
without re-running the training.

Usage:
  uv run python -m scripts.visualize_results results/flower_comparison_results.json
  uv run python -m scripts.visualize_results results/flower_simulation_results.json
  uv run python -m scripts.visualize_results results/       # auto-detect JSON
"""

import argparse
import json
import os
import sys

from rosehfl.utils.visualization import (
    visualize_simulation,
    visualize_comparison,
)


def find_json(path: str) -> str:
    """If path is a directory, find the first JSON file in it."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in ["flower_comparison_results.json", "flower_simulation_results.json",
                      "comparison_results.json", "simulation_results.json", "metrics.json"]:
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return candidate
        for f in os.listdir(path):
            if f.endswith(".json"):
                return os.path.join(path, f)
    raise FileNotFoundError(f"No JSON results found at: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize results from a JSON file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path", type=str, help="Path to results JSON file or directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for plots (default: same as JSON)")
    args = parser.parse_args()

    json_path = find_json(args.path)
    output_dir = args.output_dir or os.path.dirname(json_path)

    print(f"Loading: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    # Detect if it's a comparison, legacy single simulation, or RoSE metrics directory
    if "per_round_metrics" in data and "summary" in data:
        # Comparison results
        config = data.get("config", {})
        summary = data["summary"]
        all_metrics = data["per_round_metrics"]
        target = config.get("target_accuracy", 0.70)

        print(f"Detected: COMPARISON run ({len(summary)} strategies)")
        visualize_comparison(all_metrics, summary, config, target, output_dir)

    elif "metrics" in data:
        # Single simulation results
        config = data.get("config", {})
        metrics = data["metrics"]

        edge_nodes = data.get("edge_nodes", {})

        print("Detected: SINGLE SIMULATION")
        visualize_simulation(metrics, config, edge_nodes, output_dir)

    elif "cloud_round" in data and "accuracy" in data:
        config_path = os.path.join(os.path.dirname(json_path), "config.json")
        plan_path = os.path.join(os.path.dirname(json_path), "plan.json")
        config = {}
        edge_nodes = {}
        if os.path.isfile(config_path):
            with open(config_path) as fh:
                config = json.load(fh)
        if os.path.isfile(plan_path):
            with open(plan_path) as fh:
                plan = json.load(fh)
                edge_nodes = plan.get("edge_nodes", {})

        print("Detected: ROSE RUN DIRECTORY")
        visualize_simulation(data, config, edge_nodes, output_dir)

    else:
        print("ERROR: Unrecognized JSON structure. Expected 'per_round_metrics' or 'metrics' key.")
        sys.exit(1)

    print(f"All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
