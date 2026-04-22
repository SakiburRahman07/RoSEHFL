"""
Robust aggregation rules for RoSE-HFL.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _flatten(weights: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([layer.astype(np.float64).ravel() for layer in weights])


def _weighted_average(
    weights_list: Sequence[List[np.ndarray]],
    coefficients: Sequence[float],
) -> List[np.ndarray]:
    total = float(sum(coefficients))
    if total <= 0.0:
        raise ValueError("_weighted_average: coefficients sum to zero")
    num_layers = len(weights_list[0])
    aggregate = [
        np.zeros_like(weights_list[0][layer_idx], dtype=np.float64)
        for layer_idx in range(num_layers)
    ]
    for weights, coefficient in zip(weights_list, coefficients):
        fraction = float(coefficient) / total
        for layer_idx in range(num_layers):
            aggregate[layer_idx] += weights[layer_idx].astype(np.float64) * fraction
    return [
        layer.astype(weights_list[0][layer_idx].dtype)
        for layer_idx, layer in enumerate(aggregate)
    ]


def fedavg_aggregate(
    weights_list: List[List[np.ndarray]],
    sizes: Sequence[int],
) -> List[np.ndarray]:
    """Size-weighted FedAvg."""
    return _weighted_average(weights_list, [float(size) for size in sizes])


def median_aggregate(weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
    """Coordinate-wise median."""
    if not weights_list:
        raise ValueError("median_aggregate: empty input")
    outputs: List[np.ndarray] = []
    for layer_idx in range(len(weights_list[0])):
        stacked = np.stack([weights[layer_idx] for weights in weights_list], axis=0)
        outputs.append(np.median(stacked, axis=0).astype(weights_list[0][layer_idx].dtype))
    return outputs


def trimmed_mean_aggregate(
    weights_list: List[List[np.ndarray]],
    trim_ratio: float = 0.2,
) -> List[np.ndarray]:
    """Coordinate-wise trimmed mean."""
    if not weights_list:
        raise ValueError("trimmed_mean_aggregate: empty input")
    num_clients = len(weights_list)
    trim = int(np.floor(trim_ratio * num_clients))
    if 2 * trim >= num_clients:
        return median_aggregate(weights_list)

    outputs: List[np.ndarray] = []
    for layer_idx in range(len(weights_list[0])):
        stacked = np.stack(
            [weights[layer_idx].astype(np.float64) for weights in weights_list],
            axis=0,
        )
        sorted_values = np.sort(stacked, axis=0)
        kept = sorted_values[trim : num_clients - trim]
        outputs.append(kept.mean(axis=0).astype(weights_list[0][layer_idx].dtype))
    return outputs


def krum_aggregate(
    weights_list: List[List[np.ndarray]],
    f: int = 1,
    multi_krum_m: int = 1,
) -> List[np.ndarray]:
    """Krum / Multi-Krum robust aggregation."""
    if not weights_list:
        raise ValueError("krum_aggregate: empty input")
    num_clients = len(weights_list)
    if num_clients <= f + 2:
        return median_aggregate(weights_list)

    flattened = np.stack([_flatten(weights) for weights in weights_list], axis=0)
    distances = np.zeros((num_clients, num_clients), dtype=np.float64)
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = float(np.sum((flattened[i] - flattened[j]) ** 2))
            distances[i, j] = dist
            distances[j, i] = dist

    scores = np.zeros(num_clients, dtype=np.float64)
    neighbours = max(1, num_clients - f - 2)
    for i in range(num_clients):
        scores[i] = np.sort(distances[i])[1 : neighbours + 1].sum()

    chosen = np.argsort(scores)[: max(1, multi_krum_m)]
    selected = [weights_list[int(index)] for index in chosen]
    if len(selected) == 1:
        return [layer.copy() for layer in selected[0]]
    return _weighted_average(selected, [1.0] * len(selected))


def trust_edge_aggregate(
    node_ids: Sequence[int],
    weights_list: List[List[np.ndarray]],
    sizes: Sequence[int],
    phi: Optional[Dict[int, float]] = None,
    beta: float = 2.0,
    eta: float = 0.5,
    xi: float = 1.0,
    zeta: float = 2.0,
    alpha_cap_multiplier: float = 2.0,
    use_shrinkage: bool = True,
    prior_a: float = 2.0,
    prior_b: Optional[float] = None,
    nu: float = 1.0,
    dev_clip_q: float = 0.9,
) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
    """Contribution-trust edge aggregation used by RoSE-HFL.

    When ``use_shrinkage`` is True (C4: Bayesian-shrinkage trust), a
    Gamma(``prior_a``, ``prior_b``) prior is used to stabilise per-client
    variance estimates on small edges:

        sigma2_i = ((n-1) * dev_i^2 + 2 * b0) / (n + 2 * prior_a - 2)
        alpha_i ∝ (sigma2_i + eps)^{-nu} * size_i^eta * phi_i^xi

    When ``prior_b`` is None it is auto-fit as ``median(dev_capped^2) *
    (prior_a - 1)`` so the prior mean matches the observed deviation scale
    without letting a single extreme tail observation dominate the prior fit.

    When ``use_shrinkage`` is False the legacy ``exp(-beta * dev / median_dev)``
    trust score is used (kept for ablation).
    """
    if not weights_list:
        raise ValueError("trust_edge_aggregate: empty input")
    num_clients = len(weights_list)

    median_weights = median_aggregate(weights_list)
    median_flat = _flatten(median_weights)
    deviations = np.asarray(
        [float(np.linalg.norm(_flatten(weights) - median_flat)) for weights in weights_list],
        dtype=np.float64,
    )
    eps = 1e-6

    if use_shrinkage:
        q = float(np.clip(dev_clip_q, 0.0, 1.0))
        if q < 1.0 and num_clients >= 2:
            dev_cap = float(np.quantile(deviations, q))
            dev_capped = np.minimum(deviations, dev_cap)
        else:
            dev_capped = deviations.copy()

        a0 = float(max(prior_a, 1.0 + 1e-3))
        if prior_b is None:
            b0 = float(max(np.median(dev_capped ** 2), 1e-12)) * (a0 - 1.0)
        else:
            b0 = float(max(prior_b, 0.0))
        denom = float(num_clients + 2.0 * a0 - 2.0)
        if denom <= 0.0:
            denom = 1.0
        sigma2 = ((num_clients - 1) * deviations ** 2 + 2.0 * b0) / denom
        trust_scores = 1.0 / np.power(sigma2 + eps, float(nu))
    else:
        median_deviation = float(np.median(deviations))
        if median_deviation <= 1e-12:
            median_deviation = 1e-12
        trust_scores = np.exp(-beta * deviations / median_deviation)

    logits = eta * np.log(np.asarray(sizes, dtype=np.float64) + eps)
    if phi is not None:
        phi_values = np.asarray(
            [max(float(phi.get(int(node_id), eps)), eps) for node_id in node_ids],
            dtype=np.float64,
        )
        logits += xi * np.log(phi_values)
    else:
        phi_values = np.ones(num_clients, dtype=np.float64)
    trust_weight = 1.0 if use_shrinkage else zeta
    logits += trust_weight * np.log(np.clip(trust_scores, eps, None))

    logits -= logits.max()
    alpha = np.exp(logits)
    alpha /= alpha.sum() + 1e-12

    alpha_cap = alpha_cap_multiplier / max(num_clients, 1)
    if (alpha > alpha_cap).any():
        alpha = np.minimum(alpha, alpha_cap)
        alpha /= alpha.sum() + 1e-12

    aggregate = _weighted_average(weights_list, alpha.tolist())
    info = {
        "alpha": alpha.astype(np.float64),
        "trust_scores": trust_scores.astype(np.float64),
        "deviations": deviations.astype(np.float64),
        "phi": phi_values.astype(np.float64),
        "use_shrinkage": bool(use_shrinkage),
    }
    return aggregate, info


def aggregate_with_rule(
    rule: str,
    node_ids: Sequence[int],
    weights_list: List[List[np.ndarray]],
    sizes: Sequence[int],
    phi: Optional[Dict[int, float]] = None,
    trim_ratio: float = 0.2,
    krum_f: int = 1,
    beta: float = 2.0,
    eta: float = 0.5,
    xi: float = 1.0,
    zeta: float = 2.0,
    alpha_cap_multiplier: float = 2.0,
    use_shrinkage: bool = True,
    prior_a: float = 2.0,
    prior_b: Optional[float] = None,
    nu: float = 1.0,
    dev_clip_q: float = 0.9,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    """Apply one of the supported edge aggregation rules."""
    name = (rule or "trust").lower()
    if name in {"trust", "rose", "trust_legacy"}:
        aggregate, info = trust_edge_aggregate(
            node_ids=node_ids,
            weights_list=weights_list,
            sizes=sizes,
            phi=phi,
            beta=beta,
            eta=eta,
            xi=xi,
            zeta=zeta,
            alpha_cap_multiplier=alpha_cap_multiplier,
            use_shrinkage=(use_shrinkage and name != "trust_legacy"),
            prior_a=prior_a,
            prior_b=prior_b,
            nu=nu,
            dev_clip_q=dev_clip_q,
        )
        return aggregate, {"rule": "trust", **info}
    if name in {"uniform", "fedavg"}:
        return fedavg_aggregate(weights_list, sizes), {"rule": "fedavg"}
    if name == "median":
        return median_aggregate(weights_list), {"rule": "median"}
    if name in {"trimmed_mean", "trimmedmean"}:
        return trimmed_mean_aggregate(weights_list, trim_ratio=trim_ratio), {
            "rule": "trimmed_mean",
            "trim_ratio": float(trim_ratio),
        }
    if name == "krum":
        return krum_aggregate(weights_list, f=krum_f, multi_krum_m=1), {
            "rule": "krum",
            "f": int(krum_f),
        }
    raise ValueError(
        f"aggregate_with_rule: unknown rule '{rule}'. "
        "Choose from trust, uniform, median, trimmed_mean, krum."
    )

