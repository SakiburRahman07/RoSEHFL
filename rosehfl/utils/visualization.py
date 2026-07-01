"""
Visualization Utilities for Flower-ShapeFL
============================================
Generates publication-quality plots and HTML reports from simulation results.

Charts produced:
  Single simulation  → accuracy curve, loss curve, cost curve, edge topology, summary report
  Comparison run     → acc vs cost (Fig. 11), acc vs round, cost bars, summary table
"""

import os
import numpy as np
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Colour palette (colourblind-friendly, matches typical paper style) ───────
STRATEGY_COLORS = {
    "ShapeFL":     "#1f77b4",
    "Cost First":  "#ff7f0e",
    "Data First":  "#2ca02c",
    "Random":      "#d62728",
    "FedAvg":      "#9467bd",
}
STRATEGY_MARKERS = {
    "ShapeFL":     "o",
    "Cost First":  "s",
    "Data First":  "^",
    "Random":      "D",
    "FedAvg":      "x",
}

_FONT = {"family": "serif", "size": 12}
plt.rc("font", **_FONT)
plt.rc("axes", labelsize=13, titlesize=14)
plt.rc("legend", fontsize=10)
plt.rc("xtick", labelsize=11)
plt.rc("ytick", labelsize=11)


# ═══════════════════════════════════════════════════════════════════════════════
#  Single-simulation plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_curve(
    rounds: List[int],
    accuracies: List[float],
    title: str = "Test Accuracy vs. Cloud Rounds",
    save_path: Optional[str] = None,
):
    """Line plot of accuracy over cloud aggregation rounds."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, [a * 100 for a in accuracies], color="#1f77b4", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Cloud Aggregation Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    if accuracies:
        best = max(accuracies) * 100
        best_r = rounds[accuracies.index(max(accuracies))]
        ax.axhline(y=best, color="red", linestyle="--", alpha=0.5, label=f"Best: {best:.2f}% (round {best_r})")
        ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_loss_curve(
    rounds: List[int],
    losses: List[float],
    title: str = "Training Loss vs. Cloud Rounds",
    save_path: Optional[str] = None,
):
    """Line plot of average training loss over cloud rounds."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, losses, color="#d62728", linewidth=2, marker="s", markersize=3)
    ax.set_xlabel("Cloud Aggregation Round")
    ax.set_ylabel("Training Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_accuracy_and_loss(
    rounds: List[int],
    accuracies: List[float],
    losses: List[float],
    title: str = "Training Progress",
    save_path: Optional[str] = None,
):
    """Dual-axis plot with accuracy (left) and loss (right)."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    color_acc = "#1f77b4"
    color_loss = "#d62728"

    ax1.set_xlabel("Cloud Aggregation Round")
    ax1.set_ylabel("Test Accuracy (%)", color=color_acc)
    ln1 = ax1.plot(rounds, [a * 100 for a in accuracies], color=color_acc, linewidth=2, marker="o", markersize=3, label="Accuracy")
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Training Loss", color=color_loss)
    ln2 = ax2.plot(rounds, losses, color=color_loss, linewidth=2, linestyle="--", marker="s", markersize=2, label="Loss")
    ax2.tick_params(axis="y", labelcolor=color_loss)

    lns = ln1 + ln2
    labs = [ln.get_label() for ln in lns]
    ax1.legend(lns, labs, loc="center right")
    ax1.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_cost_curve(
    rounds: List[int],
    cumulative_costs: List[float],
    title: str = "Cumulative Communication Cost",
    save_path: Optional[str] = None,
):
    """Plot cumulative communication cost over cloud rounds."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, cumulative_costs, color="#2ca02c", linewidth=2, marker="^", markersize=3)
    ax.set_xlabel("Cloud Aggregation Round")
    ax.set_ylabel("Cumulative Communication Cost (GB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if cumulative_costs:
        ax.annotate(f"Total: {cumulative_costs[-1]:.4f} GB",
                    xy=(rounds[-1], cumulative_costs[-1]),
                    xytext=(-80, -30), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="gray"), fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_edge_topology(
    edge_nodes: Dict[int, List[int]],
    title: str = "Edge Aggregator Topology",
    save_path: Optional[str] = None,
):
    """Visualise which nodes are assigned to which edge aggregators."""
    fig, ax = plt.subplots(figsize=(10, 6))
    edges_sorted = sorted(edge_nodes.keys())
    num_edges = len(edges_sorted)

    edge_colors = plt.cm.Set2(np.linspace(0, 1, max(num_edges, 3)))

    # Cloud at the very top
    cloud_y = 3.0
    ax.scatter([0], [cloud_y], s=400, c="gold", edgecolors="black", zorder=5, marker="*")
    ax.annotate("Cloud", (0, cloud_y), textcoords="offset points", xytext=(15, 0),
                ha="left", fontsize=11, fontweight="bold")

    edge_x_positions = np.linspace(-num_edges + 1, num_edges - 1, num_edges) if num_edges > 1 else [0]
    edge_y = 2.0

    for idx, (eid, ex) in enumerate(zip(edges_sorted, edge_x_positions)):
        nodes = sorted(edge_nodes[eid])
        color = edge_colors[idx]

        ax.scatter([ex], [edge_y], s=250, c=[color], edgecolors="black", zorder=5, marker="s")
        ax.annotate(f"Edge {eid}", (ex, edge_y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")
        ax.plot([0, ex], [cloud_y, edge_y], "k-", alpha=0.3, linewidth=1.5)

        if len(nodes) == 0:
            continue
        node_xs = np.linspace(ex - 0.8, ex + 0.8, len(nodes))
        node_y = 1.0
        for nx, nid in zip(node_xs, nodes):
            ax.scatter([nx], [node_y], s=100, c=[color], edgecolors="gray", zorder=4, marker="o")
            ax.annotate(str(nid), (nx, node_y), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=7)
            ax.plot([ex, nx], [edge_y, node_y], color=color, alpha=0.4, linewidth=1)

    ax.set_title(title)
    ax.set_xlim(min(edge_x_positions) - 2, max(edge_x_positions) + 2)
    ax.set_ylim(0.4, 3.6)
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Comparison plots (paper Fig. 11 style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_accuracy_vs_cost(
    all_metrics: Dict[str, Dict],
    title: str = "Test Accuracy vs. Communication Cost",
    target_accuracy: Optional[float] = None,
    save_path: Optional[str] = None,
):
    """
    Paper Fig. 11 — accuracy (y) vs cumulative communication cost (x).
    Each strategy is a separate line.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, m in all_metrics.items():
        costs = m.get("cumulative_cost_gb") or m.get("cumulative_cost") or []
        accs = m.get("accuracy", [])
        if not costs or not accs:
            continue
        color = STRATEGY_COLORS.get(name, "#333333")
        marker = STRATEGY_MARKERS.get(name, "o")
        ax.plot(costs, [a * 100 for a in accs], label=name, color=color,
                linewidth=2, marker=marker, markersize=4, markevery=max(1, len(costs) // 15))

    if target_accuracy:
        ax.axhline(y=target_accuracy * 100, color="gray", linestyle=":", alpha=0.6,
                    label=f"Target: {target_accuracy * 100:.0f}%")

    ax.set_xlabel("Cumulative Communication Cost (GB)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_accuracy_vs_rounds(
    all_metrics: Dict[str, Dict],
    title: str = "Test Accuracy vs. Cloud Rounds",
    save_path: Optional[str] = None,
):
    """All strategies' accuracy over cloud aggregation rounds."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for name, m in all_metrics.items():
        rounds = m.get("cloud_round") or m.get("round") or []
        accs = m.get("accuracy", [])
        if not rounds or not accs:
            continue
        color = STRATEGY_COLORS.get(name, "#333333")
        marker = STRATEGY_MARKERS.get(name, "o")
        ax.plot(rounds, [a * 100 for a in accs], label=name, color=color,
                linewidth=2, marker=marker, markersize=4, markevery=max(1, len(rounds) // 15))

    ax.set_xlabel("Cloud Aggregation Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_per_round_cost_bar(
    summary: Dict[str, Dict],
    title: str = "Per-Round Communication Cost",
    save_path: Optional[str] = None,
):
    """Bar chart of per-round communication cost per strategy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(summary.keys())
    costs = [summary[n].get("per_round_cost_gb", 0) for n in names]
    colors = [STRATEGY_COLORS.get(n, "#333333") for n in names]
    bars = ax.bar(names, costs, color=colors, edgecolor="black", linewidth=0.5)

    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{cost:.6f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Communication Cost (GB)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_cost_to_target_bar(
    summary: Dict[str, Dict],
    target_accuracy: float,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Bar chart of total communication cost to reach target accuracy."""
    if title is None:
        title = f"Total Communication Cost to Reach {target_accuracy * 100:.0f}% Accuracy"

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(summary.keys())
    costs = []
    for n in names:
        c = summary[n].get("cost_to_target_gb")
        costs.append(c if c is not None else 0)
    colors = [STRATEGY_COLORS.get(n, "#333333") for n in names]
    bars = ax.bar(names, costs, color=colors, edgecolor="black", linewidth=0.5)

    for bar, cost, n in zip(bars, costs, names):
        if summary[n].get("cost_to_target_gb") is None:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.001,
                    "N/A", ha="center", va="bottom", fontsize=9, color="red", fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{cost:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Total Communication Cost (GB)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_final_accuracy_bar(
    summary: Dict[str, Dict],
    title: str = "Final Test Accuracy by Strategy",
    save_path: Optional[str] = None,
):
    """Bar chart of final accuracy per strategy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(summary.keys())
    accs = [summary[n].get("final_accuracy", 0) * 100 for n in names]
    colors = [STRATEGY_COLORS.get(n, "#333333") for n in names]
    bars = ax.bar(names, accs, color=colors, edgecolor="black", linewidth=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(accs) * 1.15 if accs else 100)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_summary_table(
    summary: Dict[str, Dict],
    config: Dict,
    target_accuracy: float,
    save_path: Optional[str] = None,
):
    """Render a summary table as a figure image."""
    fig, ax = plt.subplots(figsize=(12, 2 + len(summary) * 0.5))
    ax.axis("off")

    columns = ["Strategy", "Final Acc", "Best Acc", "Per-Round Cost (GB)",
               f"Cost@{target_accuracy*100:.0f}%", f"Rounds@{target_accuracy*100:.0f}%", "Time"]
    rows = []
    for name, s in summary.items():
        ctg = s.get("cost_to_target_gb")
        rtg = s.get("rounds_to_target")
        rows.append([
            name,
            f"{s.get('final_accuracy', 0) * 100:.2f}%",
            f"{s.get('best_accuracy', 0) * 100:.2f}%",
            f"{s.get('per_round_cost_gb', 0):.6f}",
            f"{ctg:.4f}" if ctg is not None else "N/A",
            str(rtg) if rtg is not None else "—",
            f"{s.get('time_seconds', 0):.1f}s",
        ])

    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(columns)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            table[i, j].set_facecolor("#D9E2F3" if i % 2 == 0 else "white")

    subtitle = (f"Model: {config.get('model', '?')}  |  Dataset: {config.get('dataset', '?')}  |  "
                f"Nodes: {config.get('num_nodes', '?')}  |  "
                f"κ_e={config.get('kappa_e', '?')}  κ_c={config.get('kappa_c', '?')}  κ={config.get('kappa', '?')}")
    ax.set_title(f"Strategy Comparison Summary\n{subtitle}", fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML report generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_html_report(
    output_dir: str,
    config: Dict,
    summary: Optional[Dict] = None,
    is_comparison: bool = False,
    target_accuracy: float = 0.70,
):
    """Generate an HTML report embedding all generated PNG plots."""
    title = "Strategy Comparison Report" if is_comparison else "Simulation Report"
    model = config.get("model", "?")
    dataset = config.get("dataset", "?")

    pngs = sorted([f for f in os.listdir(output_dir) if f.endswith(".png")])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Flower-ShapeFL — {title}</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 40px; background: #f8f9fa; color: #333; }}
  h1 {{ color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 10px; }}
  h2 {{ color: #444; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th {{ background: #4472C4; color: white; padding: 10px 14px; text-align: center; }}
  td {{ padding: 8px 14px; text-align: center; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #D9E2F3; }}
  tr:hover {{ background: #B4C7E7; }}
  .config {{ background: white; padding: 15px 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 15px 0; }}
  .config span {{ font-weight: bold; color: #1f77b4; }}
  .chart {{ text-align: center; margin: 20px 0; }}
  .chart img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }}
  .footer {{ margin-top: 40px; color: #888; font-size: 0.85em; border-top: 1px solid #ddd; padding-top: 10px; }}
</style>
</head>
<body>
<h1>Flower-ShapeFL — {title}</h1>
<div class="config">
  <span>Model:</span> {model} &nbsp;|&nbsp;
  <span>Dataset:</span> {dataset} &nbsp;|&nbsp;
  <span>Nodes:</span> {config.get('num_nodes', '?')} &nbsp;|&nbsp;
  <span>κ_p:</span> {config.get('kappa_p', '?')} &nbsp;
  <span>κ_e:</span> {config.get('kappa_e', '?')} &nbsp;
  <span>κ_c:</span> {config.get('kappa_c', '?')} &nbsp;
  <span>κ:</span> {config.get('kappa', '?')} &nbsp;|&nbsp;
  <span>γ:</span> {config.get('gamma', '?')} &nbsp;|&nbsp;
  <span>LR:</span> {config.get('lr', '?')}
</div>
"""

    if summary:
        html += "<h2>Results Summary</h2>\n<table>\n"
        if is_comparison:
            html += ("<tr><th>Strategy</th><th>Final Acc</th><th>Best Acc</th>"
                     f"<th>Per-Round Cost (GB)</th><th>Cost@{target_accuracy*100:.0f}%</th>"
                     f"<th>Rounds@{target_accuracy*100:.0f}%</th><th>Time</th></tr>\n")
            for name, s in summary.items():
                ctg = s.get("cost_to_target_gb")
                rtg = s.get("rounds_to_target")
                html += (f"<tr><td><strong>{name}</strong></td>"
                         f"<td>{s.get('final_accuracy', 0) * 100:.2f}%</td>"
                         f"<td>{s.get('best_accuracy', 0) * 100:.2f}%</td>"
                         f"<td>{s.get('per_round_cost_gb', 0):.6f}</td>"
                         f"<td>{'%.4f' % ctg if ctg is not None else '<span style=\"color:red\">N/A</span>'}</td>"
                         f"<td>{rtg if rtg is not None else '—'}</td>"
                         f"<td>{s.get('time_seconds', 0):.1f}s</td></tr>\n")
        else:
            html += "<tr><th>Metric</th><th>Value</th></tr>\n"
            for k, v in summary.items():
                if isinstance(v, float):
                    html += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>\n"
                else:
                    html += f"<tr><td>{k}</td><td>{v}</td></tr>\n"
        html += "</table>\n"

    for png in pngs:
        label = png.replace(".png", "").replace("_", " ").title()
        html += f'<h2>{label}</h2>\n<div class="chart"><img src="{png}" alt="{label}"></div>\n'

    html += '<div class="footer">Generated by Flower-ShapeFL visualization module</div>\n</body></html>'

    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  High-level entry points (called by scripts)
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_simulation(
    metrics: Dict[str, List],
    config: Dict,
    edge_nodes: Dict[int, List[int]],
    output_dir: str,
):
    """Generate all plots + HTML report for a single simulation run."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n[Visualization] Generating simulation plots...")

    rounds = metrics.get("cloud_round") or metrics.get("round", [])
    accs = metrics.get("accuracy", [])
    losses = metrics.get("loss", [])
    cumulative_costs = metrics.get("cumulative_cost_gb", [])

    plot_accuracy_curve(rounds, accs, save_path=os.path.join(output_dir, "accuracy_curve.png"))
    plot_loss_curve(rounds, losses, save_path=os.path.join(output_dir, "loss_curve.png"))
    plot_accuracy_and_loss(rounds, accs, losses, save_path=os.path.join(output_dir, "accuracy_and_loss.png"))

    if cumulative_costs:
        plot_cost_curve(rounds, cumulative_costs, save_path=os.path.join(output_dir, "cumulative_cost.png"))

    en = {int(k): v for k, v in edge_nodes.items()} if edge_nodes else {}
    if en:
        plot_edge_topology(en, save_path=os.path.join(output_dir, "edge_topology.png"))

    summary = {}
    if accs:
        summary["Final Accuracy"] = f"{accs[-1] * 100:.2f}%"
        summary["Best Accuracy"] = f"{max(accs) * 100:.2f}% (round {rounds[accs.index(max(accs))]})"
        summary["Total Cloud Rounds"] = len(rounds)
    if losses:
        summary["Final Loss"] = f"{losses[-1]:.4f}"
    if cumulative_costs:
        summary["Total Comm. Cost"] = f"{cumulative_costs[-1]:.4f} GB"

    generate_html_report(output_dir, config, summary, is_comparison=False)
    print("[Visualization] Done.\n")


def visualize_comparison(
    all_metrics: Dict[str, Dict],
    summary: Dict[str, Dict],
    config: Dict,
    target_accuracy: float,
    output_dir: str,
):
    """Generate all plots + HTML report for a comparison run."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n[Visualization] Generating comparison plots...")

    plot_accuracy_vs_cost(all_metrics, target_accuracy=target_accuracy,
                          save_path=os.path.join(output_dir, "accuracy_vs_cost.png"))
    plot_accuracy_vs_rounds(all_metrics,
                            save_path=os.path.join(output_dir, "accuracy_vs_rounds.png"))
    plot_per_round_cost_bar(summary,
                            save_path=os.path.join(output_dir, "per_round_cost.png"))
    plot_cost_to_target_bar(summary, target_accuracy,
                            save_path=os.path.join(output_dir, "cost_to_target.png"))
    plot_final_accuracy_bar(summary,
                            save_path=os.path.join(output_dir, "final_accuracy.png"))
    plot_summary_table(summary, config, target_accuracy,
                       save_path=os.path.join(output_dir, "summary_table.png"))

    generate_html_report(output_dir, config, summary, is_comparison=True,
                         target_accuracy=target_accuracy)
    print("[Visualization] Done.\n")


def generate_live_dashboard(metrics: Dict[str, List], output_dir: str) -> None:
    """Generate a live dashboard PNG from current metrics. Called after each round."""
    rounds = metrics.get("cloud_round") or metrics.get("round", [])
    if not rounds:
        return

    accs = metrics.get("accuracy", [])
    losses = metrics.get("loss", [])
    paper_costs = metrics.get("paper_cumulative_cost_gb", [])
    effective_costs = metrics.get("effective_cumulative_cost_gb", [])
    per_round_costs = metrics.get("paper_per_round_cost_gb", [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Live Dashboard — Round {rounds[-1]}", fontsize=16, fontweight="bold")

    # Accuracy
    ax = axes[0, 0]
    if accs:
        ax.plot(rounds, [a * 100 for a in accs], "o-", color="#1f77b4", linewidth=2, markersize=5)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Cloud Round")
        ax.set_title("Test Accuracy")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.annotate(f"{accs[-1]*100:.1f}%", xy=(rounds[-1], accs[-1]*100),
                    xytext=(5, 5), textcoords="offset points", fontsize=11, fontweight="bold")

    # Loss
    ax = axes[0, 1]
    if losses:
        ax.plot(rounds, losses, "s-", color="#d62728", linewidth=2, markersize=5)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Cloud Round")
        ax.set_title("Test Loss")
        ax.grid(True, alpha=0.3)
        ax.annotate(f"{losses[-1]:.3f}", xy=(rounds[-1], losses[-1]),
                    xytext=(5, 5), textcoords="offset points", fontsize=11, fontweight="bold")

    # Cumulative cost
    ax = axes[1, 0]
    if paper_costs:
        ax.plot(rounds, paper_costs, "^-", color="#2ca02c", linewidth=2, markersize=5, label="Paper cost")
        if effective_costs:
            ax.plot(rounds, effective_costs, "D--", color="#ff7f0e", linewidth=2, markersize=5, label="Effective cost")
        ax.set_ylabel("Cumulative Cost (GB)")
        ax.set_xlabel("Cloud Round")
        ax.set_title("Communication Cost")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Per-round cost
    ax = axes[1, 1]
    if per_round_costs:
        ax.bar(range(len(per_round_costs)), per_round_costs, color="#9467bd", alpha=0.8)
        ax.set_ylabel("Cost (GB)")
        ax.set_xlabel("Cloud Round")
        ax.set_title("Per-Round Cost")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    dashboard_path = os.path.join(output_dir, "live_dashboard.png")
    fig.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
