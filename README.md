# RoSEHFL on Flower

RoSEHFL is a research implementation of an adaptive hierarchical federated learning
framework built on Flower. This repository root contains the active RoSEHFL codebase.

The directory `flower_shapefl_code/` is a separate standalone ShapeFL project snapshot
and is not the target implementation for this README.

## What RoSEHFL Adds Beyond Baseline ShapeFL

RoSEHFL extends the baseline three-tier HFL workflow with a stack of runtime additions:

1. Hybrid planning signal from probe-based utility and diversity (`shapley`, `cosine`, `hybrid`).
2. Adaptive gamma, warm-start replanning, stage-based replanning, and drift-triggered replanning.
3. Local objective rebalancing (FedProx-style proximal term and logit adjustment).
4. Trust-aware robust edge aggregation (with optional shrinkage stabilization).
5. Communication compression and effective-cost-aware planning/accounting.
6. Local BN handling and optional edge SWA.
7. Optional server optimizer (`fedadam`) at cloud aggregation.

Core implementation lives in:

- `rosehfl/strategy.py`
- `rosehfl/client.py`
- `rosehfl/algorithms/`
- `rosehfl/utils/`
- `scripts/run_rose_simulation.py`
- `scripts/run_rose_comparison.py`

## Repository Layout (Root Project)

```text
RoSEHFL/
├── pyproject.toml
├── requirements.txt
├── rosehfl/                    # active package
│   ├── strategy.py
│   ├── client.py
│   ├── algorithms/
│   ├── data/
│   ├── models/
│   ├── topologies/
│   └── utils/
├── scripts/                    # experiment entrypoints
│   ├── run_rose_simulation.py
│   ├── run_rose_comparison.py
│   ├── run_rose_family_comparison.py
│   ├── run_shapefl_rose_effective_comparison.py
│   ├── run_shapefl_rose_q1s_comparison.py
│   ├── run_ablation_grid.py
│   ├── run_byzantine_sweep.py
│   ├── run_fairness_eval.py
│   └── visualize_results.py
├── tests/
├── results/
└── notebooks/
```

## Requirements

- Python 3.12+
- `uv` package manager
- Windows, Linux, or macOS

Main dependencies (from `pyproject.toml`):

- `flwr[simulation]`
- `torch`, `torchvision`
- `numpy`, `scipy`, `pandas`
- `opacus`, `pulp`
- `matplotlib`, `seaborn`

## Setup

```bash
uv sync
```

This creates `.venv` and installs dependencies from `pyproject.toml`.

## Important Current Wiring Note

The current root scripts under `scripts/` import `shapefl.*` symbols, while the active
root package directory is `rosehfl/`.

In a clean shell with empty `PYTHONPATH`, this currently fails with:

`ModuleNotFoundError: No module named 'shapefl'`

This is a known package-namespace mismatch in the checked-in state.

Before running the RoSE entrypoints below, make sure your local environment resolves
the `shapefl` namespace to the intended RoSEHFL implementation.

## Running Experiments

### 1. Single RoSE Run

Quick smoke run:

```bash
uv run python -m scripts.run_rose_simulation \
  --model lenet5 \
  --dataset fmnist \
  --method rose \
  --num-nodes 30 \
  --kappa 3 \
  --kappa-c 2 \
  --warmup-epochs 1
```

RoSE-Q1S style run (effective planning + delayed compression + FedAdam):

```bash
uv run python -m scripts.run_rose_simulation \
  --model lenet5 \
  --dataset fmnist \
  --topology geant2010 \
  --method rose_q1s \
  --num-nodes 30 \
  --kappa 50 \
  --kappa-c 10 \
  --output-dir results/fig11_rose/fmnist_geant2010/rose_q1s
```

Resume a run from checkpoint:

```bash
uv run python -m scripts.run_rose_simulation \
  --method rose_q1s \
  --resume \
  --output-dir results/fig11_rose/fmnist_geant2010/rose_q1s
```

### 2. Multi-Strategy Comparison

Run default RoSE + baseline comparison bundle:

```bash
uv run python -m scripts.run_rose_comparison
```

Shorter test comparison:

```bash
uv run python -m scripts.run_rose_comparison \
  --kappa 10 \
  --kappa-c 5 \
  --target-accuracy 0.50 \
  --comparison-mode matched
```

### 3. RoSE Family Comparison

```bash
uv run python -m scripts.run_rose_family_comparison
```

This compares ShapeFL with RoSE family variants (`rose`, `roseplusplus`, `rose_q1`,
`rose_q1s`, robust-agg variants), and reports both paper-cost and effective-cost summaries.

### 4. Strict Effective-Cost Comparison

ShapeFL vs selected RoSE effective strategy:

```bash
uv run python -m scripts.run_shapefl_rose_effective_comparison \
  --rose-strategy rose_q1s
```

Convenience wrapper for Q1S:

```bash
uv run python -m scripts.run_shapefl_rose_q1s_comparison
```

### 5. Ablation Study (C1-C5 Stack)

```bash
uv run python -m scripts.run_ablation_grid
```

### 6. Byzantine Robustness Sweep

```bash
uv run python -m scripts.run_byzantine_sweep
```

### 7. Fairness Re-Evaluation for Existing Runs

```bash
uv run python -m scripts.run_fairness_eval results/<run_dir>
```

### 8. Visualization

```bash
uv run python -m scripts.visualize_results results/<run_dir_or_json>
```

## Output Artifacts

For RoSE runs, the strategy persists:

- `metrics.json`
- `plan.json`
- `shapley_history.json`
- `privacy.json`
- `status.json`
- `checkpoint.pkl`

Runner-level summaries typically include:

- `config.json`
- `summary.json`
- comparison JSON outputs (for comparison scripts)

## Reproducibility Notes

- Set `--seed` explicitly for deterministic splits and initialization.
- Topology choices: `geant2010`, `uunet`, `tinet`, `viatel`, `random`.
- Dataset-model defaults follow ShapeFL pairings:
    - `fmnist` + `lenet5`
    - `cifar10` + `mobilenetv2`
    - `cifar100` + `resnet18`
- CIFAR augmentation is off by default unless `--augment` is set.

## Practical Status

The architecture and experiment scripts for RoSEHFL are present and documented,
but the current namespace mismatch (`scripts/*` importing `shapefl.*` while root
package is `rosehfl/`) should be resolved for a clean standalone experience.

Once namespace wiring is aligned, the commands in this README are the intended
workflow for setup, simulation, comparison, ablation, robustness, and analysis.
