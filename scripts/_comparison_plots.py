"""
Generic comparison plotting for saved comparison JSON bundles.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ._strategy_factory import display_name


COLORS = {
    "shapefl": "#1f77b4",
    "share": "#17a2b8",
    "cost_first": "#2ca02c",
    "data_first": "#9467bd",
    "random": "#8c564b",
    "rose": "#ff7f0e",
    "roseplusplus": "#e6550d",
    "rose_q1": "#d95f02",
    "rose_q1s": "#c44e52",
    "rose_effective": "#bc5090",
    "fedavg": "#7f7f7f",
    "fedprox": "#4c78a8",
    "gtg_shapley": "#54a24b",
    "q_fedavg": "#b279a2",
}
MARKERS = {
    "shapefl": "o",
    "share": "D",
    "cost_first": "^",
    "data_first": "v",
    "random": "P",
    "rose": "s",
    "roseplusplus": "X",
    "rose_q1": "h",
    "rose_q1s": "8",
    "rose_effective": "*",
    "fedavg": "x",
    "fedprox": "d",
    "gtg_shapley": "<",
    "q_fedavg": ">",
}


def _ordered_strategy_names(names: list[str]) -> list[str]:
    priority = [
        "shapefl",
        "share",
        "cost_first",
        "data_first",
        "random",
        "rose",
        "roseplusplus",
        "rose_q1",
        "rose_q1s",
        "rose_effective",
        "fedavg",
        "fedprox",
        "gtg_shapley",
        "q_fedavg",
    ]
    ordered = [name for name in priority if name in names]
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
        (max(strategy["metrics"]["accuracy"]) for strategy in strategies if strategy["metrics"]["accuracy"]),
        default=0.0,
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
    thresholds = []
    for threshold in candidates:
        threshold = round(float(threshold), 2)
        if threshold <= 0.0 or threshold > max_accuracy + 1e-9:
            continue
        if threshold not in thresholds:
            thresholds.append(threshold)
    return thresholds


def _rolling_std(values: list[float], window: int = 5) -> list[float]:
    result = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        if len(chunk) < 2:
            result.append(0.0)
            continue
        mean = sum(chunk) / len(chunk)
        variance = sum((value - mean) ** 2 for value in chunk) / (len(chunk) - 1)
        result.append(variance ** 0.5)
    return result


def _annotation_kwargs(strategy_name: str) -> dict[str, Any]:
    return {
        "color": COLORS.get(strategy_name, "#333333"),
        "fontsize": 9,
        "fontweight": "bold",
    }


def load_comparison_dataset(result_json: Path) -> dict[str, Any]:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    if "per_round_metrics" not in payload:
        raise ValueError(f"{result_json} is not a comparison payload.")

    config = payload.get("config", {})
    target_accuracy = config.get("target_accuracy")
    if target_accuracy is not None:
        target_accuracy = float(target_accuracy)

    common_effective_budget = (
        payload.get("effective_common_budget_gb")
        or payload.get("common_effective_budget_gb")
        or (payload.get("common_budget_gb") if payload.get("comparison_mode") == "effective" else None)
    )

    strategy_names = payload.get("strategy_names") or list(payload["per_round_metrics"].keys())
    strategy_names = _ordered_strategy_names(list(strategy_names))
    strategy_summaries = payload.get("strategy_summaries", {})
    effective_summary = payload.get("effective_summary", {})

    strategies = {}
    for strategy_name in strategy_names:
        metrics = payload["per_round_metrics"][strategy_name]
        accuracies = _as_float_list(metrics.get("accuracy"))
        rounds = _as_int_list(metrics.get("cloud_round"))
        losses = _as_float_list(metrics.get("loss"))
        paper_cumulative = _as_float_list(metrics.get("paper_cumulative_cost_gb")) or _as_float_list(metrics.get("cumulative_cost_gb"))
        effective_cumulative = _as_float_list(metrics.get("effective_cumulative_cost_gb")) or _as_float_list(metrics.get("cumulative_cost_gb"))
        paper_per_round = _as_float_list(metrics.get("paper_per_round_cost_gb")) or _as_float_list(metrics.get("per_round_cost_gb"))
        effective_per_round = _as_float_list(metrics.get("effective_per_round_cost_gb")) or _as_float_list(metrics.get("per_round_cost_gb"))
        model_payload = _as_float_list(metrics.get("model_payload_bytes"))
        probe_payload = _as_float_list(metrics.get("probe_payload_bytes"))

        if not rounds or not accuracies or not effective_cumulative:
            raise ValueError(f"{result_json} strategy '{strategy_name}' is missing required metrics.")

        strategy_summary = strategy_summaries.get(strategy_name, {})
        mode_summary = effective_summary.get(strategy_name, {})

        strategies[strategy_name] = {
            "name": strategy_name,
            "display_name": strategy_summary.get("display_name") or display_name(strategy_name),
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
                "final_accuracy": float(strategy_summary.get("final_accuracy", accuracies[-1])),
                "best_accuracy": float(strategy_summary.get("best_accuracy", max(accuracies))),
                "final_loss": float(strategy_summary.get("final_loss", losses[-1] if losses else 0.0)),
                "final_paper_cost_gb": float(strategy_summary.get("final_paper_cost_gb", paper_cumulative[-1] if paper_cumulative else 0.0)),
                "final_effective_cost_gb": float(strategy_summary.get("final_effective_cost_gb", effective_cumulative[-1])),
                "paper_per_round_cost_gb": float(strategy_summary.get("paper_per_round_cost_gb", paper_per_round[-1] if paper_per_round else 0.0)),
                "effective_per_round_cost_gb": float(strategy_summary.get("effective_per_round_cost_gb", effective_per_round[-1] if effective_per_round else 0.0)),
                "accuracy_at_common_budget": (
                    float(strategy_summary["effective_accuracy_at_common_budget"])
                    if strategy_summary.get("effective_accuracy_at_common_budget") is not None
                    else (
                        float(mode_summary["accuracy_at_common_budget"])
                        if mode_summary.get("accuracy_at_common_budget") is not None
                        else (
                            _accuracy_at_budget(accuracies, effective_cumulative, common_effective_budget)
                            if common_effective_budget is not None
                            else None
                        )
                    )
                ),
                "effective_cost_to_target_gb": (
                    float(strategy_summary["effective_cost_to_target_gb"])
                    if strategy_summary.get("effective_cost_to_target_gb") is not None
                    else _cost_to_target(accuracies, effective_cumulative, target_accuracy)
                ),
                "rounds_to_target": (
                    int(strategy_summary["rounds_to_target"])
                    if strategy_summary.get("rounds_to_target") is not None
                    else _first_round_at_or_above(accuracies, target_accuracy) if target_accuracy is not None else None
                ),
                "cost_savings_gb": strategy_summary.get("cost_savings_gb"),
                "cost_savings_pct": strategy_summary.get("cost_savings_pct"),
                "elapsed_seconds": float(strategy_summary.get("elapsed_seconds", 0.0)),
                "total_model_payload_bytes": int(strategy_summary.get("total_model_payload_bytes", round(sum(model_payload)))),
                "total_probe_payload_bytes": int(strategy_summary.get("total_probe_payload_bytes", round(sum(probe_payload)))),
            },
        }

    return {
        "source_type": "comparison_json",
        "source_path": str(result_json),
        "config": config,
        "comparison_mode": payload.get("comparison_mode"),
        "common_effective_budget_gb": common_effective_budget,
        "target_accuracy": target_accuracy,
        "strategies": strategies,
    }


def _pick_strategies(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    return [dataset["strategies"][name] for name in _ordered_strategy_names(list(dataset["strategies"].keys()))]


def write_summary_csv(dataset: dict[str, Any], output_path: Path) -> None:
    strategies = _pick_strategies(dataset)
    fieldnames = [
        "strategy",
        "display_name",
        "final_accuracy",
        "best_accuracy",
        "final_loss",
        "final_paper_cost_gb",
        "final_effective_cost_gb",
        "paper_per_round_cost_gb",
        "effective_per_round_cost_gb",
        "accuracy_at_common_budget",
        "effective_cost_to_target_gb",
        "rounds_to_target",
        "cost_savings_gb",
        "cost_savings_pct",
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
        effective_costs = metrics["effective_cumulative_cost_gb"]
        loss = metrics["loss"]
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
    target_accuracy = dataset.get("target_accuracy")
    target_label = f"Cost to {float(target_accuracy) * 100.0:.0f}%" if target_accuracy is not None else "Cost to Target"

    fig, (ax_acc, ax_cost) = plt.subplots(1, 2, figsize=(14, 5.5))

    bars_budget_acc = ax_acc.bar(
        [idx - width / 2 for idx in indices], budget_acc, width=width,
        color=[COLORS.get(s["name"], "#333333") for s in strategies], alpha=0.45, label="Accuracy @ Common Budget",
    )
    bars_final_acc = ax_acc.bar(
        [idx + width / 2 for idx in indices], final_acc, width=width,
        color=[COLORS.get(s["name"], "#333333") for s in strategies], alpha=0.9, label="Final Accuracy",
    )
    ax_acc.set_xticks(indices)
    ax_acc.set_xticklabels(labels)
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("Accuracy Summary")
    ax_acc.grid(axis="y", alpha=0.25)
    ax_acc.legend()

    target_cost_values = [value if value is not None else 0.0 for value in cost_to_target]
    bars_target_cost = ax_cost.bar(
        [idx - width / 2 for idx in indices], target_cost_values, width=width,
        color=[COLORS.get(s["name"], "#333333") for s in strategies], alpha=0.45, label=target_label,
    )
    bars_final_cost = ax_cost.bar(
        [idx + width / 2 for idx in indices], final_cost, width=width,
        color=[COLORS.get(s["name"], "#333333") for s in strategies], alpha=0.9, label="Final Effective Cost",
    )
    ax_cost.set_xticks(indices)
    ax_cost.set_xticklabels(labels)
    ax_cost.set_ylabel("Effective Cost (GB)")
    ax_cost.set_title("Cost Summary")
    ax_cost.grid(axis="y", alpha=0.25)
    ax_cost.legend()

    for bars, axis in ((bars_budget_acc, ax_acc), (bars_final_acc, ax_acc), (bars_final_cost, ax_cost)):
        for bar in bars:
            axis.annotate(
                f"{bar.get_height():.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    for bar, raw_value in zip(bars_target_cost, cost_to_target):
        axis_y = bar.get_height() if raw_value is not None else 0.02
        ax_cost.annotate(
            "N/R" if raw_value is None else f"{raw_value:.2f}",
            (bar.get_x() + bar.get_width() / 2, axis_y),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


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
    bars_paper = ax.bar([idx - width / 2 for idx in x], paper_costs, width=width, color="#5b9bd5", alpha=0.85, label="Raw / Paper Cost", edgecolor="white", linewidth=1.2)
    bars_effective = ax.bar([idx + width / 2 for idx in x], effective_costs, width=width, color="#ed7d31", alpha=0.95, label="Effective Cost", edgecolor="white", linewidth=1.2)

    for bar, value in zip(bars_paper, paper_costs):
        ax.annotate(f"{value:.2f} GB", (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=11, fontweight="bold")
    for bar, value, savings in zip(bars_effective, effective_costs, savings_pct):
        text = f"{value:.2f} GB"
        if savings > 0.5:
            text += f"\n({savings:.1f}% saved)"
        ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=11, fontweight="bold")

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
    fig, ax_left = plt.subplots(figsize=(12, 6))
    ax_right = ax_left.twinx()
    for strategy in strategies:
        name = strategy["name"]
        rounds = strategy["metrics"]["cloud_round"]
        paper = strategy["metrics"]["paper_cumulative_cost_gb"]
        effective = strategy["metrics"]["effective_cumulative_cost_gb"]
        savings_gb = [paper_cost - effective_cost for paper_cost, effective_cost in zip(paper, effective)]
        savings_pct = [(1.0 - (effective_cost / paper_cost)) * 100.0 if paper_cost > 0.0 else 0.0 for paper_cost, effective_cost in zip(paper, effective)]
        color = COLORS.get(name, "#333333")
        ax_left.plot(rounds, savings_gb, color=color, linewidth=2.5, label=f"{strategy['display_name']} (GB saved)", marker=MARKERS.get(name, "o"), markersize=4, markevery=max(1, len(rounds) // 10))
        ax_right.plot(rounds, savings_pct, color=color, linewidth=1.5, linestyle="--", alpha=0.6)

    ax_left.set_xlabel("Cloud Round", fontsize=12)
    ax_left.set_ylabel("Cumulative Savings (GB)", fontsize=12, color="#333333")
    ax_right.set_ylabel("Savings (%)", fontsize=12, color="#888888")
    ax_left.set_title("Cost Savings Over Training Rounds", fontsize=14, fontweight="bold")
    ax_left.grid(True, alpha=0.2)
    ax_left.legend(fontsize=11, loc="upper left")
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
        ax.plot(effective, acc_pct, color=color, marker=marker, linewidth=2.5, markersize=5, markevery=max(1, len(effective) // 15), label=strategy["display_name"])
        ax.annotate(f"{acc_pct[-1]:.2f}%\n@ {effective[-1]:.2f} GB", (effective[-1], acc_pct[-1]), xytext=(-80, -25), textcoords="offset points", fontsize=10, fontweight="bold", color=color, arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    if target_accuracy is not None:
        ax.axhline(y=float(target_accuracy) * 100.0, color="#666666", linestyle=":", linewidth=1.0, alpha=0.8)
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
    labels = [f"{threshold * 100.0:.0f}%" for threshold in thresholds]
    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.8 / max(1, len(strategies))
    x = list(range(len(thresholds)))

    for idx, strategy in enumerate(strategies):
        name = strategy["name"]
        accuracies = strategy["metrics"]["accuracy"]
        rounds_to = []
        for threshold in thresholds:
            round_idx = _first_round_at_or_above(accuracies, threshold)
            rounds_to.append(round_idx if round_idx is not None else 0)
        offset = (idx - (len(strategies) - 1) / 2) * width
        bars = ax.bar([value + offset for value in x], rounds_to, width=width, color=COLORS.get(name, "#333333"), alpha=0.85, label=strategy["display_name"], edgecolor="white", linewidth=1)
        for bar, value in zip(bars, rounds_to):
            ax.annotate(f"R{value}" if value > 0 else "N/R", (bar.get_x() + bar.get_width() / 2, bar.get_height() if value > 0 else 0.5), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color="#333333" if value > 0 else "#999999")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_xlabel("Accuracy Threshold", fontsize=12)
    ax.set_ylabel("Cloud Rounds Required", fontsize=12)
    ax.set_title("Convergence Speed: Rounds to Reach Accuracy Thresholds", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_cost_to_accuracy_thresholds(dataset: dict[str, Any], output_path: Path, dpi: int) -> None:
    strategies = _pick_strategies(dataset)
    thresholds = _thresholds(dataset.get("target_accuracy"), strategies)
    labels = [f"{threshold * 100.0:.0f}%" for threshold in thresholds]
    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.8 / max(1, len(strategies))
    x = list(range(len(thresholds)))

    for idx, strategy in enumerate(strategies):
        name = strategy["name"]
        accuracies = strategy["metrics"]["accuracy"]
        effective_costs = strategy["metrics"]["effective_cumulative_cost_gb"]
        values = []
        for threshold in thresholds:
            round_idx = _first_round_at_or_above(accuracies, threshold)
            values.append(effective_costs[round_idx - 1] if round_idx is not None else 0.0)
        offset = (idx - (len(strategies) - 1) / 2) * width
        bars = ax.bar([value + offset for value in x], values, width=width, color=COLORS.get(name, "#333333"), alpha=0.85, label=strategy["display_name"], edgecolor="white", linewidth=1)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.2f}" if value > 0 else "N/R", (bar.get_x() + bar.get_width() / 2, bar.get_height() if value > 0 else 0.1), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color="#333333" if value > 0 else "#999999")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_xlabel("Accuracy Threshold", fontsize=12)
    ax.set_ylabel("Effective Cost to Reach Threshold (GB)", fontsize=12)
    ax.set_title("Cost Efficiency: Effective Cost to Reach Accuracy Thresholds", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=11)
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
        ax.plot(rounds, incremental, color=color, marker=marker, linewidth=2, markersize=4, markevery=max(1, len(rounds) // 12), label=strategy["display_name"])
        avg = sum(incremental) / len(incremental)
        ax.axhline(y=avg, color=color, linestyle="--", linewidth=1, alpha=0.5)
        ax.text(rounds[-1] + 0.5, avg, f"avg={avg:.3f}", fontsize=9, color=color, va="center")

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


def _table_value(value: float | int | None, fmt: str, none_text: str = "N/R") -> str:
    if value is None:
        return none_text
    return fmt.format(value)


def plot_comparison_table(dataset: dict[str, Any], output_path: Path, dpi: int) -> None:
    strategies = _pick_strategies(dataset)
    if not strategies:
        raise ValueError("comparison_table requires at least one strategy.")
    target_accuracy = dataset.get("target_accuracy")
    target_label = f"{float(target_accuracy) * 100.0:.0f}%" if target_accuracy is not None else "target"

    metric_specs = [
        ("Final Accuracy", "final_accuracy", "higher", lambda value: _table_value(None if value is None else value * 100.0, "{:.2f}%")),
        ("Best Accuracy", "best_accuracy", "higher", lambda value: _table_value(None if value is None else value * 100.0, "{:.2f}%")),
        ("Accuracy @ Common Budget", "accuracy_at_common_budget", "higher", lambda value: _table_value(None if value is None else value * 100.0, "{:.2f}%")),
        ("Final Loss", "final_loss", "lower", lambda value: _table_value(value, "{:.4f}")),
        ("Raw / Paper Cost", "final_paper_cost_gb", "lower", lambda value: _table_value(value, "{:.2f} GB")),
        ("Effective Cost", "final_effective_cost_gb", "lower", lambda value: _table_value(value, "{:.2f} GB")),
        ("Cost Savings", "cost_savings_pct", "higher", lambda value: _table_value(value, "{:.1f}%")),
        (f"Rounds to {target_label}", "rounds_to_target", "lower", lambda value: _table_value(value, "R{:.0f}")),
        (f"Cost to {target_label}", "effective_cost_to_target_gb", "lower", lambda value: _table_value(value, "{:.2f} GB")),
        ("Model Payload", "total_model_payload_bytes", "lower", lambda value: _table_value(None if value is None else value / (1024.0 ** 3), "{:.2f} GB")),
        ("Probe Payload", "total_probe_payload_bytes", "lower", lambda value: _table_value(None if value is None else value / (1024.0 ** 2), "{:.2f} MB")),
        ("Runtime", "elapsed_seconds", "lower", lambda value: _table_value(value, "{:.1f}s")),
    ]

    fig_width = max(10.8, 2.4 + len(strategies) * 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 7.0))
    ax.axis("off")
    rows = []
    for label, key, _, formatter in metric_specs:
        rows.append([label, *[formatter(strategy["summary"].get(key)) for strategy in strategies]])

    table = ax.table(
        cellText=rows,
        colLabels=["Metric", *[strategy["display_name"] for strategy in strategies]],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)
    table.scale(1, 1.55)

    for col_idx in range(len(strategies) + 1):
        cell = table[0, col_idx]
        cell.set_facecolor("#2e4057")
        cell.set_text_props(color="white", fontweight="bold", fontsize=11)

    for row_idx in range(1, len(rows) + 1):
        background = "#f0f4f8" if row_idx % 2 == 0 else "#ffffff"
        for col_idx in range(len(strategies) + 1):
            table[row_idx, col_idx].set_facecolor(background)
            table[row_idx, col_idx].set_edgecolor("#cccccc")
        table[row_idx, 0].set_text_props(fontweight="bold", ha="left")

    for row_idx, (_, key, direction, _) in enumerate(metric_specs, start=1):
        values = [strategy["summary"].get(key) for strategy in strategies]
        valid = [(idx, value) for idx, value in enumerate(values) if value is not None]
        if len(valid) < 2:
            continue
        winner_idx = max(valid, key=lambda item: item[1])[0] if direction == "higher" else min(valid, key=lambda item: item[1])[0]
        table[row_idx, winner_idx + 1].set_facecolor("#d4edda")

    ax.set_title("Comprehensive Performance Comparison", fontsize=15, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_comparison_package(result_json: str | Path, *, output_dir: str | Path | None = None, title: str | None = None, dpi: int = 180) -> Path:
    result_json = Path(result_json)
    dataset = load_comparison_dataset(result_json)
    output_dir = Path(output_dir) if output_dir is not None else result_json.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    title = title or "Strategy Comparison"

    plot_overview(dataset, output_dir / "comparison_overview.png", title, dpi)
    plot_summary(dataset, output_dir / "comparison_summary.png", dpi)
    write_normalized_json(dataset, output_dir / "normalized_metrics.json")
    write_summary_csv(dataset, output_dir / "comparison_summary.csv")
    plot_communication_efficiency(dataset, output_dir / "communication_efficiency.png", dpi)
    plot_cost_savings_over_rounds(dataset, output_dir / "cost_savings_over_rounds.png", dpi)
    plot_accuracy_vs_cost_zoomed(dataset, output_dir / "accuracy_vs_cost_zoomed.png", dpi)
    plot_convergence_speed(dataset, output_dir / "convergence_speed.png", dpi)
    plot_cost_to_accuracy_thresholds(dataset, output_dir / "cost_to_accuracy_thresholds.png", dpi)
    plot_incremental_cost(dataset, output_dir / "incremental_cost_per_round.png", dpi)
    plot_training_stability(dataset, output_dir / "training_stability.png", dpi)
    plot_comparison_table(dataset, output_dir / "comparison_table.png", dpi)
    return output_dir
