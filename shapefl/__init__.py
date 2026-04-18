"""
ShapeFL on Flower — Top-level Package
======================================
Ports the full ShapeFL three-tier HFL architecture (IEEE/ACM ToN 2024)
to the Flower (flwr) framework, supporting both single-machine simulation
and real multi-device deployment.

Architecture mapping
--------------------
    Paper                     Flower
    ─────                     ──────
    Cloud server         →    Flower Server + ShapeFlStrategy
    Edge aggregators     →    Strategy-internal grouping & aggregation
    Computing nodes      →    Flower Clients (ShapeFlClient)

Quick start (simulation)
------------------------
    python -m scripts.run_simulation --model lenet5 --dataset fmnist

Quick start (deployment)
------------------------
    # Cloud:
    python -m scripts.deploy_server --num-nodes 30
    # Each computing node:
    python -m scripts.deploy_client --node-id 0 --server-address cloud_ip:8080
"""

from .client import ShapeFlClient, client_fn_factory
from .strategy import (
    ShapeFlStrategy,
    RoSEHFLStrategy,
    FedAvgFlatStrategy,
    FedProxFlatStrategy,
)
from .utils.shapley import compute_smc_shapley, compute_exact_shapley
from .utils.drift import PageHinkleyBank
from .utils.robust_agg import aggregate_with_rule

__all__ = [
    "ShapeFlClient",
    "client_fn_factory",
    "ShapeFlStrategy",
    "RoSEHFLStrategy",
    "FedAvgFlatStrategy",
    "FedProxFlatStrategy",
    "compute_smc_shapley",
    "compute_exact_shapley",
    "PageHinkleyBank",
    "aggregate_with_rule",
]
