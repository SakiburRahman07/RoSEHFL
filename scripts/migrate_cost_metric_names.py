#!/usr/bin/env python3
"""One-time migration: rename paper/effective cost keys to baseline/communication.

Only renames dictionary keys — never modifies numerical values.
Safe to run multiple times (idempotent).
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

KEY_RENAMES = {
    "paper_per_round_cost_gb": "baseline_per_round_cost_gb",
    "paper_cumulative_cost_gb": "baseline_cumulative_cost_gb",
    "effective_per_round_cost_gb": "communication_per_round_cost_gb",
    "effective_cumulative_cost_gb": "communication_cumulative_cost_gb",
    "final_paper_cost_gb": "final_baseline_cost_gb",
    "final_effective_cost_gb": "final_communication_cost_gb",
    "paper_cost_to_target_gb": "baseline_cost_to_target_gb",
    "effective_cost_to_target_gb": "communication_cost_to_target_gb",
    "effective_accuracy_at_common_budget": "communication_accuracy_at_common_budget",
    "effective_cost_to_reference_accuracy_gb": "communication_cost_to_reference_accuracy_gb",
    "paper_cost_to_reference_accuracy_gb": "baseline_cost_to_reference_accuracy_gb",
    "effective_common_budget_gb": "communication_common_budget_gb",
    "common_effective_budget_gb": "common_communication_budget_gb",
    "paper_common_budget_gb": "baseline_common_budget_gb",
    "effective_summary": "communication_summary",
    "paper_summary": "baseline_summary",
    "reported_paper_per_round_cost_gb": "reported_baseline_per_round_cost_gb",
    "reported_effective_per_round_cost_gb": "reported_communication_per_round_cost_gb",
    "current_cycle_effective_cost_gb": "current_cycle_communication_cost_gb",
}


def rename_keys(obj):
    """Recursively rename keys in dicts/lists. Returns (new_obj, count)."""
    if isinstance(obj, dict):
        new_dict = {}
        count = 0
        for k, v in obj.items():
            new_key = KEY_RENAMES.get(k, k)
            if new_key != k:
                count += 1
            new_v, sub_count = rename_keys(v)
            count += sub_count
            new_dict[new_key] = new_v
        return new_dict, count
    elif isinstance(obj, list):
        new_list = []
        count = 0
        for item in obj:
            new_item, sub_count = rename_keys(item)
            count += sub_count
            new_list.append(new_item)
        return new_list, count
    else:
        return obj, 0


def migrate_json(path: Path) -> int:
    """Rename keys in a JSON file. Returns number of keys renamed."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    new_data, count = rename_keys(data)
    if count > 0:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2)
    return count


def migrate_pickle(path: Path) -> int:
    """Rename keys in a pickle checkpoint file. Returns number of keys renamed."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    new_data, count = rename_keys(data)
    if count > 0:
        with open(path, "wb") as f:
            pickle.dump(new_data, f)
    return count


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/final")
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        sys.exit(1)

    total_json = 0
    total_pkl = 0
    total_keys = 0

    for path in sorted(results_dir.rglob("*")):
        if path.is_file() and path.suffix == ".json":
            count = migrate_json(path)
            if count > 0:
                print(f"  JSON [{count:3d} keys] {path}")
                total_json += 1
                total_keys += count
        elif path.is_file() and path.suffix == ".pkl":
            count = migrate_pickle(path)
            if count > 0:
                print(f"  PKL  [{count:3d} keys] {path}")
                total_pkl += 1
                total_keys += count

    print(f"\nDone: {total_json} JSON files, {total_pkl} PKL files, {total_keys} total keys renamed")


if __name__ == "__main__":
    main()
