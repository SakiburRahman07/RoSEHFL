# RoSEHFL on Flower

RoSEHFL is the active research implementation in this repository. The main package is `rosehfl/`. The directory `flower_shapefl_code/` is a separate standalone ShapeFL snapshot and is not the canonical runtime for this root project.

## Main Entry Points

- `scripts/run_simulation.py`
  Runs one strategy and writes a resumable result package.
- `scripts/run_comparison.py`
  Runs any strategy combination, writes one subpackage per strategy, and generates the comparison figure bundle automatically.
- `scripts/visualize_results.py`
  Regenerates plots from an existing `simulation_results.json`, `comparison_results.json`, or strategy `metrics.json`.
- `scripts/deploy_server.py`
  Flower deployment server entrypoint.
- `scripts/deploy_client.py`
  Flower deployment client entrypoint.

## Strategy Names

The unified runners support:

- `shapefl`
- `share`
- `cost_first`
- `data_first`
- `random`
- `rose`
- `roseplusplus`
- `rose_q1`
- `rose_q1s`
- `rose_effective`
- `rose_median`
- `rose_trimmed_mean`
- `rose_krum`
- `fedavg`
- `fedprox`
- `gtg_shapley`
- `q_fedavg`

## Setup

```bash
uv sync
```

## Single Strategy Run

```bash
uv run python -m scripts.run_simulation \
  --strategy rose_q1s \
  --model lenet5 \
  --dataset fmnist \
  --topology geant2010 \
  --num-nodes 30 \
  --kappa 50 \
  --kappa-c 10
```

Resume a previous package:

```bash
uv run python -m scripts.run_simulation \
  --resume \
  --strategy rose_q1s \
  --output-dir results/<your_run_dir>
```

## Multi-Strategy Comparison

```bash
uv run python -m scripts.run_comparison \
  --strategies shapefl rose_q1s fedavg \
  --model lenet5 \
  --dataset fmnist \
  --topology geant2010 \
  --num-nodes 30
```

Resume a previous comparison package:

```bash
uv run python -m scripts.run_comparison \
  --resume \
  --strategies shapefl rose_q1s fedavg \
  --output-dir results/<your_comparison_dir>
```

## Visualization

```bash
uv run python -m scripts.visualize_results results/<run_dir_or_json>
```

## Result Package Layout

Single-strategy package:

- `metadata.json`
- `run_status.json`
- `simulation_results.json`
- `summary.json`
- `fairness.json`
- `metrics.json`
- `status.json`
- `checkpoint.pkl`
- simulation plots / `report.html`

Comparison package:

- `metadata.json`
- `run_status.json`
- `comparison_results.json`
- `comparison_summary.csv`
- `normalized_metrics.json`
- comparison plots
- `strategies/<strategy_name>/...` for each strategy subpackage

## Research Utilities

The repository also keeps a few focused research utilities:

- `scripts/run_ablation_grid.py`
- `scripts/run_byzantine_sweep.py`
- `scripts/run_fairness_eval.py`
