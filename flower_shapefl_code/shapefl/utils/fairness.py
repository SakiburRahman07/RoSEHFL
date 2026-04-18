"""
Fairness metrics and per-client evaluation helpers for RoSE-HFL.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def load_weights_into_model(
    model: torch.nn.Module,
    parameters_ndarrays: Sequence[np.ndarray],
) -> torch.nn.Module:
    """Load numpy parameters into a freshly constructed model."""
    state_keys = list(model.state_dict().keys())
    state_dict = {
        key: torch.as_tensor(value)
        for key, value in zip(state_keys, parameters_ndarrays)
    }
    model.load_state_dict(state_dict, strict=True)
    return model


def per_client_accuracy(
    model: torch.nn.Module,
    dataset: Dataset,
    partitions: Dict[int, List[int]],
    device: torch.device,
    batch_size: int = 64,
) -> Dict[int, float]:
    """Evaluate a model on each client's held-out partition."""
    model.to(device).eval()
    results: Dict[int, float] = {}
    with torch.no_grad():
        for node_id, indices in partitions.items():
            if not indices:
                results[node_id] = 0.0
                continue
            loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
            correct, total = 0, 0
            for features, labels in loader:
                features, labels = features.to(device), labels.to(device)
                predictions = model(features).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            results[node_id] = correct / max(total, 1)
    return results


def per_client_accuracy_from_weights(
    parameters_ndarrays: Sequence[np.ndarray],
    model_factory: Callable[[], torch.nn.Module],
    dataset: Dataset,
    partitions: Dict[int, List[int]],
    device: torch.device,
    batch_size: int = 64,
) -> Dict[int, float]:
    """Evaluate a weight vector on each client's held-out partition."""
    model = load_weights_into_model(model_factory(), parameters_ndarrays)
    return per_client_accuracy(
        model=model,
        dataset=dataset,
        partitions=partitions,
        device=device,
        batch_size=batch_size,
    )


def jain_index(values: Sequence[float]) -> float:
    values_arr = np.asarray(list(values), dtype=np.float64)
    if values_arr.size == 0:
        return 0.0
    denominator = values_arr.size * float(np.sum(values_arr ** 2))
    if denominator <= 1e-12:
        return 0.0
    return float(values_arr.sum() ** 2 / denominator)


def gini_coefficient(values: Sequence[float]) -> float:
    values_arr = np.sort(np.asarray(list(values), dtype=np.float64))
    if values_arr.size == 0:
        return 0.0
    mean = float(values_arr.mean())
    if mean <= 1e-12:
        return 0.0
    cumulative = np.cumsum(values_arr)
    return float((values_arr.size + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / values_arr.size)


def worst_k_percent(values: Sequence[float], k: float = 10.0) -> float:
    values_arr = np.sort(np.asarray(list(values), dtype=np.float64))
    if values_arr.size == 0:
        return 0.0
    cutoff = max(1, int(np.ceil(values_arr.size * k / 100.0)))
    return float(values_arr[:cutoff].mean())


def bootstrap_ci(
    values: Sequence[float],
    stat_fn: Callable[[Sequence[float]], float] = np.mean,  # type: ignore[arg-type]
    num_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return point estimate and bootstrap confidence interval."""
    values_arr = np.asarray(list(values), dtype=np.float64)
    if values_arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    samples = np.asarray(
        [
            stat_fn(rng.choice(values_arr, size=values_arr.size, replace=True))
            for _ in range(num_resamples)
        ],
        dtype=np.float64,
    )
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1 - alpha))
    return float(stat_fn(values_arr)), lower, upper


def summarise_fairness(
    per_client_acc: Dict[int, float],
    worst_pct: float = 10.0,
    seed: int = 42,
) -> Dict[str, float | Dict[str, float]]:
    """Return the full fairness report for a per-client accuracy map."""
    values = list(per_client_acc.values())
    mean_estimate, mean_ci_lo, mean_ci_hi = bootstrap_ci(values, np.mean, seed=seed)
    worst_estimate, worst_ci_lo, worst_ci_hi = bootstrap_ci(
        values,
        lambda sample: worst_k_percent(sample, k=worst_pct),
        seed=seed,
    )
    jain_estimate, jain_ci_lo, jain_ci_hi = bootstrap_ci(values, jain_index, seed=seed)
    gini_estimate, gini_ci_lo, gini_ci_hi = bootstrap_ci(values, gini_coefficient, seed=seed)

    return {
        "mean_accuracy": float(np.mean(values)) if values else 0.0,
        "median_accuracy": float(np.median(values)) if values else 0.0,
        "std_accuracy": float(np.std(values)) if values else 0.0,
        "min_accuracy": float(np.min(values)) if values else 0.0,
        "max_accuracy": float(np.max(values)) if values else 0.0,
        "jain_index": jain_estimate,
        "gini_coefficient": gini_estimate,
        f"worst_{worst_pct:g}pct_accuracy": worst_estimate,
        "bootstrap_ci": {
            "mean_accuracy": {"estimate": mean_estimate, "lower": mean_ci_lo, "upper": mean_ci_hi},
            "jain_index": {"estimate": jain_estimate, "lower": jain_ci_lo, "upper": jain_ci_hi},
            "gini_coefficient": {"estimate": gini_estimate, "lower": gini_ci_lo, "upper": gini_ci_hi},
            f"worst_{worst_pct:g}pct_accuracy": {
                "estimate": worst_estimate,
                "lower": worst_ci_lo,
                "upper": worst_ci_hi,
            },
        },
        "per_client_accuracy": {str(node_id): float(acc) for node_id, acc in per_client_acc.items()},
    }

