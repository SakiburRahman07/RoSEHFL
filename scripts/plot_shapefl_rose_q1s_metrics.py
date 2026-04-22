#!/usr/bin/env python3
"""
Generate ShapeFL vs RoSE-Q1S comparison figures from saved comparison JSON.

This script is intended for the `shapefl_vs_rose_q1s_results.json` payload
written by `scripts/run_shapefl_rose_effective_comparison.py`. Unlike the older
hardcoded plotting script, it derives every figure from the actual saved
per-round metrics and summary fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DISPLAY_NAMES = {
    "shapefl": "ShapeFL",
    "rose_q1s": "RoSE-Q1S",
}
COLORS = {
    "shapefl": "#1f77b4",
    "rose_q1s": "#d95f02",
}
MARKERS = {
    "shapefl": "o",
    "rose_q1s": "s",
}

DEFAULT_RESULT_JSON = Path("results/fig11_rose/fmnist_geant2010/shapefl_vs_rose_q1s_results.json")


def _display_name(strategy_name: str) -> str:
    return DISPLAY_NAMES.get(strategy_name, strategy_name.replace("_", " ").title())


def _ordered_strategy_names(names: list[str]) -> list[str]:
    preferred = ["shapefl", "rose_q1s"]
    ordered = [name for name in preferred if name in names]
    ordered.extend(name for name in names if name not in ordered)
    return ordered


def _as_float_list(values: list[Any] | None) -> list[float]:
    return [float(value) for value in (values or [])]


def _as_int_list(values: list[Any] | None) -> list[int]:
    return [int(value) for value in (values or [])]


def _first_round_at_or_above(values: list[float], threshold: float) -> int | None:
    for idx, value in enumerate(values):
        if float(value) >= float(threshold):
            return idx + 1
    return None


def _accuracy_at_budget(accuracies: list[float], effective_costs: list[float], budget_gb: float) -> float | None:
    for cost, accuracy in zip(effective_costs, accuracies):
        if float(cost) >= float(budget_gb):
            return float(accuracy)
    return accuracies[-1] if accuracies else None


def _cost_to_target(accuracies: list[float], effective_costs: list[float], target_accuracy: float | None) -> float | None:
    if target_accuracy is None:
        return None
    round_idx = _first_round_at_or_above(accuracies, target_accuracy)
    if round_idx is None:
        return None
    return float(effective_costs[round_idx - 1])


def _thresholds(target_accuracy: float | None, strategies: list[dict[str, Any]]) -> list[float]:
    max_accuracy = max(
        max(strategy["metrics"]["accuracy"])
        for strategy in strategies
        if strategy["metrics"]["accuracy"]
    )
    if target_accuracy is None:
        floor = max(0.10, math.floor((max_accuracy - 0.20) * 20.0) / 20.0)
        candidates = [floor, floor + 0.10, floor + 0.15, floor + 0.20, floor + 0.22, floor + 0.24]
    else:
        candidates = [
            target_accuracy - 0.20,
            target_accuracy - 0.10,
            target_accuracy - 0.05,
            target_accuracy,
            target_accuracy + 0.02,
            target_accuracy + 0.04,
        ]
    result = []
    for threshold in candidates:
        threshold = round(float(threshold), 2)
        if threshold <= 0.0 or threshold > max_accuracy + 1e-9:
            continue
        if threshold not in result:
            result.append(threshold)
    return result


def _rolling_std(values: list[float], window: int = 5) -> list[float]:
    rolling = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_values = values[start : idx + 1]
        if len(window_values) < 2:
            rolling.append(0.0)
            continue
        rolling.append(statistics.stdev(window_values))
    return rolling


def _clean_near_zero(value: float, eps: float = 1e-12) -> float:
    return 0.0 if abs(value) < eps else float(value)


def _annotation_kwargs(strategy_name: str) -> dict[str, Any]:
    return {
        "color": COLORS.get(strategy_name, "#333333"),
        "fontsize": 9,
        "fontweight": "bold",
    }


def load_dataset(result_json: Path) -> dict[str, Any]:
    payload = json.loads(result_json.read_text(encoding="utf-8"))

    if "strategies" in payload and "per_round_metrics" not in payload:
        payload["source_type"] = payload.get("source_type", "normalized_json")
        payload["source_path"] = str(result_json)
        return payload

    if "per_round_metrics" not in payload:
        raise ValueError(
            f"{result_json} is not a supported comparison payload: missing 'per_round_metrics'."
        )

    config = payload.get("config", {})
    target_accuracy = config.get("target_accuracy")
    if target_accuracy is not None:
        target_accuracy = float(target_accuracy)

    summary_payload = payload.get("summary", {})
    per_round_payload = payload.get("per_round_metrics", {})
    common_budget = payload.get("common_effective_budget_gb")
    if common_budget is None:
        final_costs = []
        for metrics in per_round_payload.values():
            effective = _as_float_list(metrics.get("effective_cumulative_cost_gb"))
            if effective:
                final_costs.append(effective[-1])
        common_budget = min(final_costs) if final_costs else None

    strategies: dict[str, Any] = {}
    for strategy_name in _ordered_strategy_names(list(per_round_payload.keys())):
        metrics = per_round_payload[strategy_name]
        summary = summary_payload.get(strategy_name, {})

        rounds = _as_int_list(metrics.get("cloud_round"))
        accuracies = _as_float_list(metrics.get("accuracy"))
        losses = _as_float_list(metrics.get("loss"))
        paper_cumulative = _as_float_list(metrics.get("paper_cumulative_cost_gb"))
        effective_cumulative = _as_float_list(metrics.get("effective_cumulative_cost_gb"))
        paper_per_round = _as_float_list(metrics.get("paper_per_round_cost_gb"))
        effective_per_round = _as_float_list(metrics.get("effective_per_round_cost_gb"))
        model_payload = _as_float_list(metrics.get("model_payload_bytes"))
        probe_payload = _as_float_list(metrics.get("probe_payload_bytes"))

        if not rounds or not accuracies or not effective_cumulative:
            raise ValueError(
                f"{result_json} strategy '{strategy_name}' is missing required round/accuracy/effective cost data."
            )

        if not paper_cumulative:
            paper_cumulative = _as_float_list(metrics.get("cumulative_cost_gb"))
        if not paper_per_round:
            paper_per_round = _as_float_list(metrics.get("per_round_cost_gb"))

        inferred_rounds_to_target = _first_round_at_or_above(accuracies, target_accuracy) if target_accuracy is not None else None
        inferred_cost_to_target = _cost_to_target(accuracies, effective_cumulative, target_accuracy)
        inferred_accuracy_at_budget = (
            _accuracy_at_budget(accuracies, effective_cumulative, float(common_budget))
            if common_budget is not None
            else None
        )

        cost_savings_gb = (
            paper_cumulative[-1] - effective_cumulative[-1]
            if paper_cumulative and effective_cumulative
            else None
        )
        cost_savings_pct = (
            (1.0 - (effective_cumulative[-1] / paper_cumulative[-1])) * 100.0
            if paper_cumulative and effective_cumulative and paper_cumulative[-1] > 0.0
            else 0.0
        )

        strategies[strategy_name] = {
            "name": strategy_name,
            "display_name": _display_name(strategy_name),
            "metrics": {
                "cloud_round": rounds,
                "accuracy": accuracies,
                "loss": losses,
                "paper_cumulative_cost_gb": paper_cumulative,
                "effective_cumulative_cost_gb": effective_cumulative,
                "paper_per_round_cost_gb": paper_per_round,
                "effective_per_round_cost_gb": effective_per_round,
                "model_payload_bytes": model_payload,
                "probe_payload_bytes": probe_payload,
            },
            "summary": {
                "final_accuracy": float(summary.get("final_accuracy", accuracies[-1])),
                "best_accuracy": float(summary.get("best_accuracy", max(accuracies))),
                "accuracy_at_common_budget": (
                    float(summary["accuracy_at_common_effective_budget"])
                    if summary.get("accuracy_at_common_effective_budget") is not None
                    else inferred_accuracy_at_budget
                ),
                "effective_cost_to_target_gb": (
                    float(summary["effective_cost_to_target_gb"])
                    if summary.get("effective_cost_to_target_gb") is not None
                    else inferred_cost_to_target
                ),
                "rounds_to_target": inferred_rounds_to_target,
                "cost_to_shapefl_final_accuracy_gb": (
                    float(summary["cost_to_shapefl_final_accuracy_gb"])
                    if summary.get("cost_to_shapefl_final_accuracy_gb") is not None
                    else None
                ),
                "final_paper_cost_gb": float(paper_cumulative[-1]) if paper_cumulative else None,
                "final_effective_cost_gb": float(
                    summary.get("final_effective_cumulative_cost_gb", effective_cumulative[-1])
                ),
                "effective_per_round_cost_gb": (
                    float(summary["effective_per_round_cost_gb"])
                    if summary.get("effective_per_round_cost_gb") is not None
                    else (float(effective_per_round[-1]) if effective_per_round else None)
                ),
                "elapsed_seconds": (
                    float(summary["elapsed_seconds"])
                    if summary.get("elapsed_seconds") is not None
                    else None
                ),
                "total_model_payload_bytes": int(
                    summary.get("total_model_payload_bytes", round(sum(model_payload)))
                ),
                "total_probe_payload_bytes": int(
                    summary.get("total_probe_payload_bytes", round(sum(probe_payload)))
                ),
                "final_loss": float(losses[-1]) if losses else None,
                "best_round": rounds[accuracies.index(max(accuracies))],
                "cost_savings_gb": (
                    _clean_near_zero(cost_savings_gb) if cost_savings_gb is not None else None
                ),
                "cost_savings_pct": _clean_near_zero(cost_savings_pct),
            },
        }

    return {
        "source_type": "comparison_json",
        "source_path": str(result_json),
        "config": config,
        "comparison_mode": payload.get("comparison_mode"),
        "common_effective_budget_gb": common_budget,
        "target_accuracy": target_accuracy,
        "shapefl_final_accuracy": payload.get("shapefl_final_accuracy"),
        "strategies": strategies,
    }


def _pick_strategies(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    strategies = dataset["strategies"]
    return [strategies[name] for name in _ordered_strategy_names(list(strategies.keys()))]


def plot_overview(dataset: dict[str, Any], output_path: Path, title: str, dpi: int) -> None:
    strategies = _pick_strategies(dataset)
    common_budget = dataset.get("common_effective_budget_gb")
    target_accuracy = dataset.get("target_accuracy")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_acc_round, ax_loss, ax_cost_round, ax_acc_cost = axes.flatten()

    for strategy in strategies:
        name = strategy["name"]
        label = strategy["display_name"]
        metrics = strategy["metrics"]
        summary = strategy["summary"]
        rounds = metrics["cloud_round"]
        accuracy_pct = [value * 100.0 for value in metrics["accuracy"]]
        loss = metrics["loss"]
        effective_costs = metrics["effective_cumulative_cost_gb"]

        color = COLORS.get(name, "#333333")
        marker = MARKERS.get(name, "o")
        mark_every = max(1, len(rounds) // 12)

        ax_acc_round.plot(rounds, accuracy_pct, color=color, marker=marker, linewidth=2.0, markersize=4, markevery=mark_every, label=label)
        ax_loss.plot(rounds, loss, color=color, marker=marker, linewidth=2.0, markersize=4, markevery=mark_every, label=label)
        ax_cost_round.plot(rounds, effective_costs, color=color, marker=marker, linewidth=2.0, markersize=4, markevery=mark_every, label=label)
        ax_acc_cost.plot(effective_costs, accuracy_pct, color=color, marker=marker, linewidth=2.0, markersize=4, markevery=mark_every, label=label)

        final_round = rounds[-1]
        final_cost = effective_costs[-1]
        final_acc = summary["final_accuracy"] * 100.0
        ax_acc_round.scatter([final_round], [final_acc], color=color, edgecolors="black", s=70, zorder=5)
        ax_acc_cost.scatter([final_cost], [final_acc], color=color, edgecolors="black", s=70, zorder=5)
        ax_acc_round.annotate(f"R{final_round}", (final_round, final_acc), xytext=(6, 6), textcoords="offset points", **_annotation_kwargs(name))
        ax_acc_cost.annotate(f"{final_cost:.2f} GB", (final_cost, final_acc), xytext=(6, 6), textcoords="offset points", **_annotation_kwargs(name))

    if common_budget is not None:
        ax_cost_round.axhline(y=float(common_budget), color="#6c757d", linestyle="--", linewidth=1.2, label=f"Common budget: {float(common_budget):.2f} GB")
        ax_acc_cost.axvline(x=float(common_budget), color="#6c757d", linestyle="--", linewidth=1.2, label=f"Common budget: {float(common_budget):.2f} GB")

    if target_accuracy is not None:
        target_pct = float(target_accuracy) * 100.0
        ax_acc_round.axhline(y=target_pct, color="#444444", linestyle=":", linewidth=1.1, label=f"Target: {target_pct:.0f}%")
        ax_acc_cost.axhline(y=target_pct, color="#444444", linestyle=":", linewidth=1.1, label=f"Target: {target_pct:.0f}%")

    ax_acc_round.set_title("Accuracy vs Cloud Round")
    ax_acc_round.set_xlabel("Cloud Round")
    ax_acc_round.set_ylabel("Accuracy (%)")
    ax_acc_round.grid(True, alpha=0.25)
    ax_acc_round.legend()

    ax_loss.set_title("Loss vs Cloud Round")
    ax_loss.set_xlabel("Cloud Round")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.25)
    ax_loss.legend()

    ax_cost_round.set_title("Effective Cost vs Cloud Round")
    ax_cost_round.set_xlabel("Cloud Round")
    ax_cost_round.set_ylabel("Effective Cumulative Cost (GB)")
    ax_cost_round.grid(True, alpha=0.25)
    ax_cost_round.legend()

    ax_acc_cost.set_title("Accuracy vs Effective Cost")
    ax_acc_cost.set_xlabel("Effective Cumulative Cost (GB)")
    ax_acc_cost.set_ylabel("Accuracy (%)")
    ax_acc_cost.grid(True, alpha=0.25)
    ax_acc_cost.legend()

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_summary(dataset: dict[str, Any], output_path: Path, dpi: int) -> None:
    strategies = _pick_strategies(dataset)
    labels = [strategy["display_name"] for strategy in strategies]
    indices = list(range(len(strategies)))
    width = 0.34

    final_acc = [(strategy["summary"]["final_accuracy"] or 0.0) * 100.0 for strategy in strategies]
    budget_acc = [((strategy["summary"].get("accuracy_at_common_budget") or 0.0) * 100.0) for strategy in strategies]
    final_cost = [strategy["summary"].get("final_effective_cost_gb") or 0.0 for strategy in strategies]
    cost_to_target = [strategy["summary"].get("effective_cost_to_target_gb") for strategy in strategies]
    cost_to_target_bars = [value if value is not None else 0.0 for value in cost_to_target]
    target_accuracy = dataset.get("target_accuracy")
    target_label = f"Cost to {float(target_accuracy) * 100.0:.0f}%" if target_accuracy is not None else "Cost to Target"

    fig, (ax_acc, ax_cost) = plt.subplots(1, 2, figsize=(14, 5.5))

    bars_budget_acc = ax_acc.bar(
        [idx - width / 2 for idx in indices],
        budget_acc,
        width=width,
        color=[COLORS.get(s["name"], "#333333") for s in strategies],
        alpha=0.45,
        label="Accuracy @ Common Budget",
    )
    bars_final_acc = ax_acc.bar(
        [idx + width / 2 for idx in indices],
        final_acc,
        width=width,
        color=[COLORS.get(s["name"], "#333333") for s in strategies],
        alpha=0.9,
        label="Final Accuracy",
    )
    ax_acc.set_xticks(indices)
    ax_acc.set_xticklabels(labels)
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy Summary")
    ax_acc.grid(axis="y", alpha=0.25)
    ax_acc.legend()

    bars_cost_to_target = ax_cost.bar(
        [idx - width / 2 for idx in indices],
        cost_to_target_bars,
        width=width,
        color=[COLORS.get(s["name"], "#333333") for s in strategies],
        alpha=0.45,
        label=target_label,
    )
    bars_final_cost = ax_cost.bar(
        [idx + width / 2 for idx in indices],
        final_cost,
        width=width,
        color=[COLORS.get(s["name"], "#333333") for s in strategies],
        alpha=0.9,
        label="Final Effective Cost",
    )
    ax_cost.set_xticks(indices)
    ax_cost.set_xticklabels(labels)
    ax_cost.set_ylabel("Effective Cost (GB)")
    ax_cost.set_title("Cost Summary")
    ax_cost.grid(axis="y", alpha=0.25)
    ax_cost.legend()

    for bars in (bars_budget_acc, bars_final_acc):
        for bar in bars:
            ax_acc.annotate(
                f"{bar.get_height():.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    for bar, original in zip(bars_cost_to_target, cost_to_target):
        label = "N/R" if original is None else f"{original:.2f}"
        y = bar.get_height() if original is not None else 0.02
        ax_cost.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2, y),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars_final_cost:
        ax_cost.annotate(
            f"{bar.get_height():.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(dataset: dict[str, Any], output_path: Path) -> None:
    strategies = _pick_strategies(dataset)
    fieldnames = [
        "strategy",
        "display_name",
        "final_accuracy",
        "best_accuracy",
        "best_round",
        "accuracy_at_common_budget",
        "final_loss",
        "final_paper_cost_gb",
        "final_effective_cost_gb",
        "cost_savings_gb",
        "cost_savings_pct",
        "effective_per_round_cost_gb",
        "effective_cost_to_target_gb",
        "rounds_to_target",
        "cost_to_shapefl_final_accuracy_gb",
        "total_model_payload_bytes",
        "total_probe_payload_bytes",
        "elapsed_seconds",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for strategy in strategies:
            writer.writerow(
                {
                    "strategy": strategy["name"],
                    "display_name": strategy["display_name"],
                    **strategy["summary"],
                }
            )


def write_normalized_json(dataset: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2)


def plot_communication_efficiency(dataset: dict[str, Any], output_path: Path, dpi: int) -> None:
    strategies = _pick_strategies(dataset)
    labels = [s["display_name"] for s in strategies]
    paper_costs = [s["summary"]["final_paper_cost_gb"] or 0.0 for s in strategies]
    effective_costs = [s["summary"]["final_effective_cost_gb"] or 0.0 for s in strategies]
    savings_pct = [
        (1.0 - effective / paper) * 100.0 if paper > 0.0 else 0.0
        for paper, effective in zip(paper_costs, effective_costs)
    ]

    x = list(range(len(strategies)))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_paper = ax.bar(
        [idx - width / 2 for idx in x],
        paper_costs,
        width=width,
        color=["#5b9bd5", "#ed7d31"][: len(strategies)],
        alpha=0.85,
        label="Raw / Paper Cost",
        edgecolor="white",
        linewidth=1.2,
    )
    bars_effective = ax.bar(
        [idx + width / 2 for idx in x],
        effective_costs,
        width=width,
        color=["#2e75b6", "#c55a11"][: len(strategies)],
        alpha=0.95,
        label="Effective Cost",
        edgecolor="white",
        linewidth=1.2,
    )

    for bar, value in zip(bars_paper, paper_costs):
        ax.annotate(
            f"{value:.2f} GB",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    for bar, value, savings in zip(bars_effective, effective_costs, savings_pct):
        text = f"{value:.2f} GB"
        if savings > 0.5:
            text += f"\n({savings:.1f}% saved)"
        ax.annotate(
            text,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Communication Cost (GB)", fontsize=12)
    ax.set_title("Communication Efficiency: Raw Cost vs Effective Cost", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_cost_savings_over_rounds(dataset: dict[str, Any], output_path: Path, dpi: int) -> None:
    strategies = _pick_strategies(dataset)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    for strategy in strategies:
        name = strategy["name"]
        rounds = strategy["metrics"]["cloud_round"]
        paper = strategy["metrics"]["paper_cumulative_cost_gb"]
        effective = strategy["metrics"]["effective_cumulative_cost_gb"]
        savings_gb = [paper_cost - effective_cost for paper_cost, effective_cost in zip(paper, effective)]
        savings_pct = [
            (1.0 - (effective_cost / paper_cost)) * 100.0 if paper_cost > 0.0 else 0.0
            for paper_cost, effective_cost in zip(paper, effective)
        ]
        color = COLORS.get(name, "#333333")

        ax1.plot(
            rounds,
            savings_gb,
            color=color,
            linewidth=2.5,
            label=f"{strategy['display_name']} (GB saved)",
            marker=MARKERS.get(name, "o"),
            markersize=4,
            markevery=max(1, len(rounds) // 10),
        )
        ax2.plot(rounds, savings_pct, color=color, linewidth=1.5, linestyle="--", alpha=0.6)

    ax1.set_xlabel("Cloud Round", fontsize=12)
    ax1.set_ylabel("Cumulative Savings (GB)", fontsize=12, color="#333333")
    ax2.set_ylabel("Savings (%)", fontsize=12, color="#888888")
    ax1.set_title("Cost Savings Over Training Rounds", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=11, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_cost_zoomed(dataset: dict[str, Any], output_path: Path, dpi: int) -> None:
    strategies = _pick_strategies(dataset)
    target_accuracy = dataset.get("target_accuracy")
    all_acc_pct = [acc * 100.0 for strategy in strategies for acc in strategy["metrics"]["accuracy"]]
    low = max(0.0, min(all_acc_pct) - 5.0)
    high = min(100.0, max(all_acc_pct) + 2.0)

    fig, ax = plt.subplots(figsize=(12, 7))
    for strategy in strategies:
        name = strategy["name"]
        effective = strategy["metrics"]["effective_cumulative_cost_gb"]
        acc_pct = [a * 100.0 for a in strategy["metrics"]["accuracy"]]
        color = COLORS.get(name, "#333333")
        marker = MARKERS.get(name, "o")

        ax.plot(
            effective,
            acc_pct,
            color=color,
            marker=marker,
            linewidth=2.5,
            markersize=5,
            markevery=max(1, len(effective) // 15),
            label=strategy["display_name"],
        )
        ax.annotate(
            f"{acc_pct[-1]:.2f}%\n@ {effective[-1]:.2f} GB",
            (effective[-1], acc_pct[-1]),
            xytext=(-80, -25),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )

    if target_accuracy is not None:
        target_pct = float(target_accuracy) * 100.0
        ax.axhline(y=target_pct, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)

    ax.set_ylim(low, high)
    ax.set_xlabel("Effective Cumulative Cost (GB)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Effective Cost", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=12, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_speed(dataset: dict[str, Any], output_path: Path, dpi: int) -> None:
    strategies = _pick_strategies(dataset)
    thresholds = _thresholds(dataset.get("target_accuracy"), strategies)
    threshold_labels = [f"{threshold * 100.0:.0f}%" for threshold in thresholds]

    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.35
    x = list(range(len(thresholds)))

    for strategy_idx, strategy in enumerate(strategies):
        name = strategy["name"]
        accuracies = strategy["metrics"]["accuracy"]
        rounds_to_target = []
        for threshold in thresholds:
            round_idx = _first_round_at_or_above(accuracies, threshold)
            rounds_to_target.append(round_idx if round_idx is not None else 0)

        offset = (strategy_idx - (len(strategies) - 1) / 2) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            rounds_to_target,
            width=width,
            color=COLORS.get(name, "#333333"),
            alpha=0.85,
            label=strategy["display_name"],
            edgecolor="white",
            linewidth=1,
        )
        for bar, value in zip(bars, rounds_to_target):
            ax.annotate(
                f"R{value}" if value > 0 else "N/R",
                (bar.get_x() + bar.get_width() / 2, bar.get_height() if value > 0 else 0.5),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold" if value > 0 else "normal",
                color="#333333" if value > 0 else "#999999",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(threshold_labels, fontsize=12, fontweight="bold")
    ax.set_xlabel("Accuracy Threshold", fontsize=12)
    ax.set_ylabel("Cloud Rounds Required", fontsize=12)
    ax.set_title("Convergence Speed: Rounds to Reach Accuracy Thresholds", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_cost_to_accuracy_thresholds(dataset: dict[str, Any], output_path: Path, dpi: int) -> None:
    strategies = _pick_strategies(dataset)
    thresholds = _thresholds(dataset.get("target_accuracy"), strategies)
    threshold_labels = [f"{threshold * 100.0:.0f}%" for threshold in thresholds]

    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.35
    x = list(range(len(thresholds)))

    for strategy_idx, strategy in enumerate(strategies):
        name = strategy["name"]
        accuracies = strategy["metrics"]["accuracy"]
        effective = strategy["metrics"]["effective_cumulative_cost_gb"]
        cost_to_threshold = []
        for threshold in thresholds:
            round_idx = _first_round_at_or_above(accuracies, threshold)
            cost_to_threshold.append(effective[round_idx - 1] if round_idx is not None else 0.0)

        offset = (strategy_idx - (len(strategies) - 1) / 2) * width
        bars = ax.bar(
            [xi + offset for xi in x],
            cost_to_threshold,
            width=width,
            color=COLORS.get(name, "#333333"),
            alpha=0.85,
            label=strategy["display_name"],
            edgecolor="white",
            linewidth=1,
        )
        for bar, value in zip(bars, cost_to_threshold):
            ax.annotate(
                f"{value:.2f}" if value > 0.0 else "N/R",
                (bar.get_x() + bar.get_width() / 2, bar.get_height() if value > 0.0 else 0.1),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold" if value > 0.0 else "normal",
                color="#333333" if value > 0.0 else "#999999",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(threshold_labels, fontsize=12, fontweight="bold")
    ax.set_xlabel("Accuracy Threshold", fontsize=12)
    ax.set_ylabel("Effective Cost to Reach Threshold (GB)", fontsize=12)
    ax.set_title("Cost Efficiency: Effective Cost to Reach Accuracy Thresholds", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_incremental_cost(dataset: dict[str, Any], output_path: Path, dpi: int) -> None:
    strategies = _pick_strategies(dataset)
    fig, ax = plt.subplots(figsize=(12, 6))

    for strategy in strategies:
        name = strategy["name"]
        rounds = strategy["metrics"]["cloud_round"]
        incremental = strategy["metrics"]["effective_per_round_cost_gb"]
        if not incremental:
            cumulative = strategy["metrics"]["effective_cumulative_cost_gb"]
            incremental = [cumulative[0]] + [cumulative[idx] - cumulative[idx - 1] for idx in range(1, len(cumulative))]
        color = COLORS.get(name, "#333333")
        marker = MARKERS.get(name, "o")

        ax.plot(
            rounds,
            incremental,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=4,
            markevery=max(1, len(rounds) // 12),
            label=strategy["display_name"],
        )
        average = sum(incremental) / len(incremental)
        ax.axhline(y=average, color=color, linestyle="--", linewidth=1, alpha=0.5)
        ax.text(rounds[-1] + 0.5, average, f"avg={average:.3f}", fontsize=9, color=color, va="center")

    ax.set_xlabel("Cloud Round", fontsize=12)
    ax.set_ylabel("Incremental Effective Cost (GB per round)", fontsize=12)
    ax.set_title("Per-Round Communication Cost", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_training_stability(dataset: dict[str, Any], output_path: Path, dpi: int) -> None:
    strategies = _pick_strategies(dataset)
    window = 5
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(13, 9), height_ratios=[2, 1], sharex=True)

    for strategy in strategies:
        name = strategy["name"]
        rounds = strategy["metrics"]["cloud_round"]
        acc_pct = [value * 100.0 for value in strategy["metrics"]["accuracy"]]
        stability = _rolling_std(acc_pct, window)
        color = COLORS.get(name, "#333333")
        marker = MARKERS.get(name, "o")
        mark_every = max(1, len(rounds) // 12)

        ax_top.plot(rounds, acc_pct, color=color, marker=marker, linewidth=2, markersize=4, markevery=mark_every, label=strategy["display_name"])
        ax_bottom.plot(rounds, stability, color=color, marker=marker, linewidth=2, markersize=4, markevery=mark_every, label=strategy["display_name"])
        ax_bottom.fill_between(rounds, stability, alpha=0.15, color=color)

    ax_top.set_ylabel("Accuracy (%)", fontsize=12)
    ax_top.set_title("Training Trajectory and Stability Analysis", fontsize=14, fontweight="bold")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(fontsize=11)

    ax_bottom.set_xlabel("Cloud Round", fontsize=12)
    ax_bottom.set_ylabel(f"Rolling Std Dev (w={window})", fontsize=12)
    ax_bottom.set_title("Accuracy Volatility (Lower is More Stable)", fontsize=12)
    ax_bottom.grid(True, alpha=0.25)
    ax_bottom.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _table_value(raw: float | int | None, fmt: str, none_text: str = "N/R") -> str:
    if raw is None:
        return none_text
    return fmt.format(raw)


def plot_comparison_table(dataset: dict[str, Any], output_path: Path, dpi: int) -> None:
    strategies = _pick_strategies(dataset)
    if len(strategies) < 2:
        raise ValueError("comparison_table requires at least two strategies.")

    target_accuracy = dataset.get("target_accuracy")
    target_label = f"{float(target_accuracy) * 100.0:.0f}%" if target_accuracy is not None else "target"

    metric_specs = [
        ("Final Accuracy", "final_accuracy", "higher", lambda v: _table_value(None if v is None else v * 100.0, "{:.2f}%")),
        ("Best Accuracy", "best_accuracy", "higher", lambda v: _table_value(None if v is None else v * 100.0, "{:.2f}%")),
        ("Accuracy @ Common Budget", "accuracy_at_common_budget", "higher", lambda v: _table_value(None if v is None else v * 100.0, "{:.2f}%")),
        ("Final Loss", "final_loss", "lower", lambda v: _table_value(v, "{:.4f}")),
        ("Raw / Paper Cost", "final_paper_cost_gb", "lower", lambda v: _table_value(v, "{:.2f} GB")),
        ("Effective Cost", "final_effective_cost_gb", "lower", lambda v: _table_value(v, "{:.2f} GB")),
        ("Cost Savings", "cost_savings_pct", "higher", lambda v: _table_value(v, "{:.1f}%")),
        (f"Rounds to {target_label}", "rounds_to_target", "lower", lambda v: _table_value(v, "R{:.0f}")),
        (f"Cost to {target_label}", "effective_cost_to_target_gb", "lower", lambda v: _table_value(v, "{:.2f} GB")),
        ("Cost to ShapeFL Final Acc", "cost_to_shapefl_final_accuracy_gb", "lower", lambda v: _table_value(v, "{:.2f} GB")),
        ("Model Payload", "total_model_payload_bytes", "lower", lambda v: _table_value(None if v is None else v / (1024.0 ** 3), "{:.2f} GB")),
        ("Probe Payload", "total_probe_payload_bytes", "lower", lambda v: _table_value(None if v is None else v / (1024.0 ** 2), "{:.2f} MB")),
        ("Runtime", "elapsed_seconds", "lower", lambda v: _table_value(v, "{:.1f}s")),
    ]

    rows_text = []
    for label, key, _, formatter in metric_specs:
        row = [label]
        for strategy in strategies[:2]:
            row.append(formatter(strategy["summary"].get(key)))
        rows_text.append(row)

    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    ax.axis("off")
    table = ax.table(
        cellText=rows_text,
        colLabels=["Metric", strategies[0]["display_name"], strategies[1]["display_name"]],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.55)

    for col_idx in range(3):
        cell = table[0, col_idx]
        cell.set_facecolor("#2e4057")
        cell.set_text_props(color="white", fontweight="bold", fontsize=12)

    for row_idx in range(1, len(rows_text) + 1):
        background = "#f0f4f8" if row_idx % 2 == 0 else "#ffffff"
        for col_idx in range(3):
            table[row_idx, col_idx].set_facecolor(background)
            table[row_idx, col_idx].set_edgecolor("#cccccc")
        table[row_idx, 0].set_text_props(fontweight="bold", ha="left")

    for row_idx, (_, key, direction, _) in enumerate(metric_specs, start=1):
        values = [strategy["summary"].get(key) for strategy in strategies[:2]]
        valid = [(idx, value) for idx, value in enumerate(values) if value is not None]
        if len(valid) < 2:
            continue
        winner_idx = max(valid, key=lambda item: item[1])[0] if direction == "higher" else min(valid, key=lambda item: item[1])[0]
        table[row_idx, winner_idx + 1].set_facecolor("#d4edda")

    ax.set_title("Comprehensive Performance Comparison", fontsize=15, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ShapeFL vs RoSE-Q1S visualizations from saved comparison JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=DEFAULT_RESULT_JSON,
        help="Path to shapefl_vs_rose_q1s_results.json or an already-normalized metrics JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the generated plots and summary files.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="ShapeFL vs RoSE-Q1S Effective-Cost Comparison",
        help="Figure title for the overview plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output DPI for saved PNG files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_dataset(args.result_json)

    output_dir = args.output_dir or Path("plot_outputs") / f"{args.result_json.stem}_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    dpi = int(args.dpi)
    overview_path = output_dir / "comparison_overview.png"
    summary_plot_path = output_dir / "comparison_summary.png"
    normalized_json_path = output_dir / "normalized_metrics.json"
    summary_csv_path = output_dir / "comparison_summary.csv"

    plot_overview(dataset, overview_path, args.title, dpi)
    plot_summary(dataset, summary_plot_path, dpi)
    write_normalized_json(dataset, normalized_json_path)
    write_summary_csv(dataset, summary_csv_path)

    extra_plots = {
        "communication_efficiency.png": lambda path: plot_communication_efficiency(dataset, path, dpi),
        "cost_savings_over_rounds.png": lambda path: plot_cost_savings_over_rounds(dataset, path, dpi),
        "accuracy_vs_cost_zoomed.png": lambda path: plot_accuracy_vs_cost_zoomed(dataset, path, dpi),
        "convergence_speed.png": lambda path: plot_convergence_speed(dataset, path, dpi),
        "cost_to_accuracy_thresholds.png": lambda path: plot_cost_to_accuracy_thresholds(dataset, path, dpi),
        "incremental_cost_per_round.png": lambda path: plot_incremental_cost(dataset, path, dpi),
        "training_stability.png": lambda path: plot_training_stability(dataset, path, dpi),
        "comparison_table.png": lambda path: plot_comparison_table(dataset, path, dpi),
    }
    for filename, plot_fn in extra_plots.items():
        plot_fn(output_dir / filename)

    print(f"Data source: {dataset['source_path']}")
    print(f"Output dir:  {output_dir}")
    print("Generated:")
    print("  - comparison_overview.png")
    print("  - comparison_summary.png")
    for filename in extra_plots:
        print(f"  - {filename}")
    print("  - normalized_metrics.json")
    print("  - comparison_summary.csv")


if __name__ == "__main__":
    main()
