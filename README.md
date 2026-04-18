# ShapeFL on Flower

Complete port of the **ShapeFL** three-tier Hierarchical Federated Learning (HFL)
architecture to the [Flower](https://flower.ai/) framework.

> **Paper:** _ShapeFL: Shape-Aware Federated Learning for Resource-Constrained
> Hierarchical Networks_, IEEE/ACM Transactions on Networking, 2024.

## Project Structure

```text
flower_shapefl/
├── requirements.txt          # pip dependencies
├── README.md                 # this file
├── shapefl/                  # core library
│   ├── __init__.py           # exports ShapeFlClient, ShapeFlStrategy, ...
│   ├── client.py             # Flower NumPyClient (computing node)
│   ├── strategy.py           # Flower Strategy (cloud + edge logic)
│   ├── models/               # LeNet-5, MobileNetV2, ResNet-18
│   ├── data/                 # Data loading & non-IID partitioning
│   ├── algorithms/           # GoA (Alg. 1) + LoS (Alg. 2)
│   └── utils/                # Cosine-similarity computation
├── scripts/                  # entry-point scripts
│   ├── run_simulation.py     # single-strategy simulation
│   ├── run_comparison.py     # 5-strategy comparison (Fig. 11)
│   ├── deploy_server.py      # cloud server for real deployment
│   └── deploy_client.py      # client for real deployment
├── dataset/                  # place datasets here (auto-downloaded)
└── results/                  # output directory
```

## Quick Start

### 1. Install

```bash
cd flower_shapefl
uv sync
```

This creates a `.venv` with Python 3.12 and installs all dependencies from `pyproject.toml`.

### 2. Simulation (single machine)

All 30 clients run in a single process — no networking needed.

```bash
# Default ShapeFL (LeNet-5 + FMNIST, κ=50, κ_c=10)
uv run python -m scripts.run_simulation

# Quick smoke test (3 cloud rounds)
uv run python -m scripts.run_simulation --kappa 3 --kappa-c 2 --kappa-p 5

# Different planning mode
uv run python -m scripts.run_simulation --planning-mode cost_first --gamma 0

# Paper topology variants
uv run python -m scripts.run_simulation --topology uunet
uv run python -m scripts.run_simulation --topology tinet
uv run python -m scripts.run_simulation --topology viatel
```

For paper-faithful runs, CIFAR augmentation is disabled by default. Enable it explicitly with `--augment` if you want an augmented variant.

### 3. Strategy Comparison

Compare the default 5 strategies (ShapeFL, Cost First, Data First, Random, FedAvg):

```bash
uv run python -m scripts.run_comparison

# Shorter run for testing
uv run python -m scripts.run_comparison --kappa 10 --kappa-c 5 --kappa-p 10 --target-accuracy 0.50

# Include FedProx as an additional flat baseline
uv run python -m scripts.run_comparison --strategies shapefl cost_first data_first random fedavg fedprox --fedprox-mu 0.01

# Paper HFL comparison including SHARE
uv run python -m scripts.run_comparison --strategies shapefl share cost_first data_first random

# Fixed-budget reproduction run (e.g. Figure 10 style, 500 local epochs)
uv run python -m scripts.run_comparison --total-local-epochs 500 --strategies shapefl fedavg fedprox
```

### 4. Real Deployment (multiple machines)

**On the cloud machine:**

```bash
uv run python -m scripts.deploy_server --num-nodes 30 --address 0.0.0.0:8080
```

**On each computing node:**

```bash
uv run python -m scripts.deploy_client --node-id 0 --server-address cloud_ip:8080
uv run python -m scripts.deploy_client --node-id 1 --server-address cloud_ip:8080
# ... up to node-id 29
```

## Architecture Mapping

| Paper            | Flower                                        |
| ---------------- | --------------------------------------------- |
| Cloud server     | Flower Server + `ShapeFlStrategy`             |
| Edge aggregators | Strategy-internal grouping in `aggregate_fit` |
| Computing nodes  | Flower Clients (`ShapeFlClient`)              |

### Flower Round Mapping

| Flower Round      | What Happens                                 |
| ----------------- | -------------------------------------------- |
| Round 1           | Pre-training (κ_p epochs) + LoS/GoA planning |
| Rounds 2..1+κ·κ_c | Training (edge epochs + cloud aggregation)   |

Each Flower round = one edge epoch. Cloud aggregation happens every κ_c rounds.
Total Flower rounds: **1 + κ × κ_c** (e.g., 501 for default params).

## Paper Hyperparameters

| Parameter  | Default | Description                                          |
| ---------- | ------: | ---------------------------------------------------- |
| κ_p        |      30 | Pre-training epochs                                  |
| κ_e        |       1 | Local epochs per edge round                          |
| κ_c        |      10 | Edge epochs per cloud round                          |
| κ          |      50 | Total cloud rounds                                   |
| γ          |    2800 | Cost-diversity trade-off weight                      |
| B_e        |      10 | Max nodes per edge aggregator                        |
| lr         |   0.001 | SGD learning rate (paper Section V-A)                |
| momentum   |     0.0 | SGD momentum (paper uses pure SGD, Algorithm 3 l.32) |
| batch_size |      32 | Mini-batch size (paper Section V-A)                  |
| N          |      30 | Number of computing nodes                            |

## Dataset–Model Pairings (from paper)

| Dataset       | Model       | Input Size |
| ------------- | :---------- | :--------- |
| Fashion-MNIST | LeNet-5     | 1×28×28    |
| CIFAR-10      | MobileNetV2 | 3×32×32    |
| CIFAR-100     | ResNet-18   | 3×32×32    |

## Planning Modes

| Mode         | γ Value | Description                          |
| ------------ | ------: | ------------------------------------ |
| `shapefl`    |    2800 | Full LoS + GoA (balanced)            |
| `share`      |    2800 | Preliminary KL-to-uniform HFL method |
| `cost_first` |       0 | Minimise communication cost only     |
| `data_first` |     1e8 | Maximise data diversity only         |
| `random`     |     N/A | Random edge selection + round-robin  |
| `fedavg`     |     N/A | Flat FedAvg baseline (no edge layer) |

## Topologies

Paper-backed topology choices are `geant2010`, `uunet`, `tinet`, and `viatel`.
The `random` topology remains available as a synthetic fallback.

## Citation

If you use this implementation in your research, please cite the original ShapeFL paper:

```bibtex
@ARTICLE{deng2024shapefl,
  author={Deng, Yongheng and Lyu, Feng and Xia, Tengyi and Zhou, Yuezhi and Zhang, Yaoxue and Ren, Ju and Yang, Yuanyuan},
  journal={IEEE/ACM Transactions on Networking},
  title={A Communication-Efficient Hierarchical Federated Learning Framework via Shaping Data Distribution at Edge},
  year={2024},
  volume={32},
  number={3},
  pages={2600-2615},
  doi={10.1109/TNET.2024.3363916},
  keywords={Costs;Data models;Servers;Computational modeling;Training data;Federated learning;Distributed databases;Hierarchical federated learning;communication efficiency;edge computing;distributed edge intelligence}
}
```

## 🔗 References

1. Y. Deng et al., "A Communication-Efficient Hierarchical Federated Learning Framework via Shaping Data Distribution at Edge," in IEEE/ACM Transactions on Networking, vol. 32, no. 3, pp. 2600-2615, June 2024, doi: 10.1109/TNET.2024.3363916.

2. H. Brendan McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," in AISTATS, 2017.

3. Y. LeCun et al., "Gradient-based learning applied to document recognition," in Proceedings of the IEEE, 1998.

4. M. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in CVPR, 2018.

5. K. He et al., "Deep Residual Learning for Image Recognition," in CVPR, 2016.

6. Fashion-MNIST Dataset: https://github.com/zalandoresearch/fashion-mnist

7. CIFAR-10/100 Datasets: https://www.cs.toronto.edu/~kriz/cifar.html

## Acknowledgments

- Original ShapeFL paper authors for the groundbreaking research
- PyTorch team for the excellent deep learning framework
- Flower team for the flexible federated learning framework
- Raspberry Pi Foundation for affordable edge computing hardware
- Fashion-MNIST and CIFAR dataset creators for providing benchmark datasets
