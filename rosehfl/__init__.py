"""
RoSEHFL on Flower — top-level package
=====================================
Primary Python package for the RoSEHFL research implementation.

This package exports:
    - FlClient and client factory helpers.
    - Baseline hierarchical and flat strategies for controlled comparisons.
    - RoSEHFLStrategy, the adaptive hierarchical strategy used by RoSE runs.
    - Core utility entrypoints (Shapley, drift, robust aggregation).

Symbol names keep historical ShapeFL-compatible naming where useful for
experiment scripts and tests.

Typical simulation entrypoint:
    python -m scripts.run_simulation
"""

from .client import FlClient, client_fn_factory
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
    "FlClient",
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
