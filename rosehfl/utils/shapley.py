"""
Stratified Monte-Carlo Shapley helpers for RoSE-HFL.

This module provides:
- probe-set construction
- probe-logit serialisation for Flower metrics payloads
- probe-set evaluation / inference helpers
- sMC-Shapley and exact-Shapley estimators
- Gaussian-noise helpers for privacy experiments
"""

from __future__ import annotations

import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .model_state import head_state_keys, state_key_indices


def build_probe_set(
    test_dataset: Dataset,
    probe_size: int,
    num_classes: int,
    seed: int = 42,
) -> Subset:
    """Return a stratified probe subset of ``test_dataset``."""
    rng = np.random.RandomState(seed)
    targets = extract_targets(test_dataset)
    per_class = max(1, probe_size // max(num_classes, 1))

    indices: List[int] = []
    for class_id in range(num_classes):
        class_indices = np.where(targets == class_id)[0]
        if len(class_indices) == 0:
            continue
        chosen = rng.choice(
            class_indices,
            size=min(per_class, len(class_indices)),
            replace=False,
        )
        indices.extend(chosen.tolist())

    if len(indices) < probe_size:
        pool = np.setdiff1d(np.arange(len(targets)), np.asarray(indices))
        if len(pool) > 0:
            extra = rng.choice(
                pool,
                size=min(probe_size - len(indices), len(pool)),
                replace=False,
            )
            indices.extend(extra.tolist())

    return Subset(test_dataset, sorted(indices))


def extract_targets(dataset: Dataset) -> np.ndarray:
    """Best-effort extraction of integer labels from a dataset."""
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.cpu().numpy()
        return np.asarray(targets)
    return np.asarray([int(dataset[idx][1]) for idx in range(len(dataset))])


def weighted_aggregate(
    weights_list: Sequence[List[np.ndarray]],
    sizes: Sequence[int],
) -> List[np.ndarray]:
    """Standard FedAvg across a list of model weights."""
    total = float(sum(sizes))
    if total <= 0.0:
        raise ValueError("weighted_aggregate: total size is zero")
    num_layers = len(weights_list[0])
    aggregate = [
        np.zeros_like(weights_list[0][layer_idx], dtype=np.float64)
        for layer_idx in range(num_layers)
    ]
    for weights, size in zip(weights_list, sizes):
        fraction = float(size) / total
        for layer_idx in range(num_layers):
            aggregate[layer_idx] += weights[layer_idx].astype(np.float64) * fraction
    return [
        layer.astype(weights_list[0][layer_idx].dtype)
        for layer_idx, layer in enumerate(aggregate)
    ]


def leave_one_out(
    coalition_weights: List[np.ndarray],
    client_weights: List[np.ndarray],
    alpha_client: float,
) -> List[np.ndarray]:
    """Undo a single client's contribution from an aggregated coalition."""
    if alpha_client >= 1.0 - 1e-12:
        raise ValueError("leave_one_out: coalition has a single member")
    scale = 1.0 / (1.0 - alpha_client)
    return [
        (coalition.astype(np.float64) - alpha_client * client.astype(np.float64)) * scale
        for coalition, client in zip(coalition_weights, client_weights)
    ]


def _load_weights_into_model(
    model: torch.nn.Module,
    weights: List[np.ndarray],
) -> torch.nn.Module:
    keys = list(model.state_dict().keys())
    if len(keys) != len(weights):
        raise ValueError(
            f"_load_weights_into_model: expected {len(keys)} arrays, got {len(weights)}"
        )
    state_dict = {key: torch.as_tensor(value.copy()) for key, value in zip(keys, weights)}
    model.load_state_dict(state_dict, strict=True)
    return model


def evaluate_on_probe(
    weights: List[np.ndarray],
    model_factory,
    probe_loader: DataLoader,
    device: torch.device,
) -> float:
    """Return probe-set accuracy for a weight vector."""
    model = _load_weights_into_model(model_factory(), weights)
    model.to(device).eval()

    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in probe_loader:
            features, labels = features.to(device), labels.to(device)
            predictions = model(features).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def predict_probe_logits(
    weights: List[np.ndarray],
    model_factory,
    probe_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Return concatenated probe logits for a weight vector."""
    model = _load_weights_into_model(model_factory(), weights)
    model.to(device).eval()
    logits: List[np.ndarray] = []
    with torch.no_grad():
        for features, _ in probe_loader:
            features = features.to(device)
            batch_logits = model(features).detach().cpu().numpy().astype(np.float32)
            logits.append(batch_logits)
    if not logits:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(logits, axis=0)


def mean_softmax_distribution(logits: np.ndarray) -> np.ndarray:
    """Return the mean softmax distribution over the probe set."""
    if logits.size == 0:
        return np.zeros((0,), dtype=np.float32)
    logits_tensor = torch.as_tensor(np.ascontiguousarray(logits), dtype=torch.float32)
    probs = F.softmax(logits_tensor, dim=1).mean(dim=0)
    probs = probs / probs.sum().clamp_min(1e-12)
    return probs.cpu().numpy().astype(np.float64)


def accuracy_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Return classification accuracy from logits and integer targets."""
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    if logits.size == 0 or targets.size == 0:
        return 0.0
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("accuracy_from_logits: logits and targets length mismatch")
    predictions = logits.argmax(axis=1)
    return float(np.mean(predictions == targets))


def gaussian_noise_sigma(
    epsilon: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
) -> float:
    """Return the Gaussian-mechanism standard deviation."""
    if epsilon <= 0:
        raise ValueError("gaussian_noise_sigma: epsilon must be positive")
    if delta <= 0:
        raise ValueError("gaussian_noise_sigma: delta must be positive")
    return float(sensitivity * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon)


def add_gaussian_noise(
    array: np.ndarray,
    epsilon: float,
    delta: float = 1e-5,
    sensitivity: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply Gaussian-mechanism noise to an ndarray."""
    sigma = gaussian_noise_sigma(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
    orig_dtype = array.dtype
    rng = np.random.RandomState(seed)
    noisy = array.astype(np.float64) + rng.normal(0.0, sigma, size=array.shape)
    return noisy.astype(orig_dtype)


def serialize_probe_logits(
    logits: np.ndarray,
    key_prefix: str = "probe_logits",
    compression_level: int = 3,
) -> Dict[str, bytes | int | str]:
    """Serialise a logits array into Flower-scalar-compatible payload fields."""
    logits = np.asarray(logits, dtype=np.float32, order="C")
    payload = zlib.compress(logits.tobytes(order="C"), level=compression_level)
    return {
        f"{key_prefix}_version": 1,
        f"{key_prefix}_shape": ",".join(str(dim) for dim in logits.shape),
        f"{key_prefix}_dtype": str(logits.dtype),
        f"{key_prefix}_payload": payload,
        f"{key_prefix}_num_bytes": int(len(payload)),
    }


def deserialize_probe_logits(
    metrics: Dict[str, object],
    key_prefix: str = "probe_logits",
) -> Optional[np.ndarray]:
    """Deserialize probe logits from a metrics dictionary."""
    payload = metrics.get(f"{key_prefix}_payload")
    if payload is None:
        return None
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("deserialize_probe_logits: payload must be bytes")
    shape_str = metrics.get(f"{key_prefix}_shape")
    dtype_str = metrics.get(f"{key_prefix}_dtype", "float32")
    if not isinstance(shape_str, str):
        raise TypeError("deserialize_probe_logits: shape metadata missing")
    shape = tuple(int(part) for part in shape_str.split(",") if part)
    array = np.frombuffer(zlib.decompress(bytes(payload)), dtype=np.dtype(dtype_str))
    return array.reshape(shape).astype(np.float32, copy=False)


def probe_payload_num_bytes(
    metrics: Dict[str, object],
    key_prefix: str = "probe_logits",
) -> int:
    """Return the compressed payload size from a metrics dictionary."""
    value = metrics.get(f"{key_prefix}_num_bytes", 0)
    return int(value) if value is not None else 0


def _replace_selected_layers(
    base_weights: List[np.ndarray],
    selected_weights: List[np.ndarray],
    selected_indices: Sequence[int],
) -> List[np.ndarray]:
    combined = [layer.copy() for layer in base_weights]
    for local_index, global_index in enumerate(selected_indices):
        combined[global_index] = selected_weights[local_index].copy()
    return combined


def _selected_layers(
    weights: List[np.ndarray],
    selected_indices: Sequence[int],
) -> List[np.ndarray]:
    return [weights[index] for index in selected_indices]


def _selected_weighted_aggregate(
    weights_list: Sequence[List[np.ndarray]],
    sizes: Sequence[int],
) -> List[np.ndarray]:
    total = float(sum(sizes))
    if total <= 0.0:
        raise ValueError("_selected_weighted_aggregate: total size is zero")
    num_layers = len(weights_list[0])
    aggregate = [
        np.zeros_like(weights_list[0][layer_idx], dtype=np.float64)
        for layer_idx in range(num_layers)
    ]
    for weights, size in zip(weights_list, sizes):
        fraction = float(size) / total
        for layer_idx in range(num_layers):
            aggregate[layer_idx] += weights[layer_idx].astype(np.float64) * fraction
    return [
        layer.astype(weights_list[0][layer_idx].dtype)
        for layer_idx, layer in enumerate(aggregate)
    ]


def _stable_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    if first.size < 2 or second.size < 2:
        return 0.0
    if np.allclose(first, first[0]) or np.allclose(second, second[0]):
        return 0.0
    corr = float(np.corrcoef(first, second)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return corr


def _cosine_distance(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    eps: float = 1e-12,
) -> float:
    denom = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denom <= eps:
        return 0.0
    cosine = float(np.dot(vector_a, vector_b) / denom)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return 1.0 - cosine


def compute_smc_shapley(
    client_weights: Dict[int, List[np.ndarray]],
    client_sizes: Dict[int, int],
    probe_loader: DataLoader,
    model_factory,
    device: torch.device,
    T: int = 4,
    K: int = 6,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[int, float]:
    """Stratified Monte-Carlo Shapley over ``T * K`` coalition samples."""
    node_ids = sorted(client_weights.keys())
    num_nodes = len(node_ids)
    if num_nodes == 0:
        return {}
    if num_nodes == 1:
        score = evaluate_on_probe(
            client_weights[node_ids[0]],
            model_factory,
            probe_loader,
            device,
        )
        return {node_ids[0]: float(score)}

    rng = np.random.RandomState(seed)
    if K >= num_nodes - 1:
        strata = list(range(2, num_nodes + 1))
    else:
        strata = sorted({max(2, int(round(num_nodes * k / K))) for k in range(1, K + 1)})
        if strata[-1] != num_nodes:
            strata.append(num_nodes)

    phi_sum = {node_id: 0.0 for node_id in node_ids}
    phi_count = {node_id: 0 for node_id in node_ids}

    for stratum_idx, coalition_size in enumerate(strata):
        for permutation_idx in range(T):
            coalition = list(rng.permutation(node_ids))[:coalition_size]
            coalition_weights = [client_weights[node_id] for node_id in coalition]
            coalition_sizes = [client_sizes[node_id] for node_id in coalition]
            aggregated = weighted_aggregate(coalition_weights, coalition_sizes)
            utility_coalition = evaluate_on_probe(
                aggregated,
                model_factory,
                probe_loader,
                device,
            )
            total_examples = float(sum(coalition_sizes))

            for node_id in coalition:
                alpha_node = float(client_sizes[node_id]) / total_examples
                if alpha_node >= 1.0 - 1e-12:
                    continue
                without_node = leave_one_out(
                    coalition_weights=aggregated,
                    client_weights=client_weights[node_id],
                    alpha_client=alpha_node,
                )
                utility_without = evaluate_on_probe(
                    without_node,
                    model_factory,
                    probe_loader,
                    device,
                )
                phi_sum[node_id] += utility_coalition - utility_without
                phi_count[node_id] += 1

            if verbose:
                print(
                    f"sMC-Shapley stratum {stratum_idx + 1}/{len(strata)} "
                    f"perm {permutation_idx + 1}/{T} | |C|={coalition_size}"
                )

    return {
        node_id: phi_sum[node_id] / max(phi_count[node_id], 1)
        for node_id in node_ids
    }


def compute_hybrid_phi(
    client_weights: Dict[int, List[np.ndarray]],
    client_sizes: Dict[int, int],
    probe_loader: DataLoader,
    model_factory,
    device: torch.device,
    reference_weights: List[np.ndarray],
    probe_logits: Optional[Dict[int, np.ndarray]] = None,
    probe_targets: Optional[np.ndarray] = None,
    T: int = 1,
    K: int = 64,
    seed: int = 42,
    eps: float = 1e-6,
    lambda_floor: float = 0.1,
    lambda_ceiling: float = 0.9,
) -> Tuple[Dict[int, float], Dict[str, object]]:
    """Compute C1's hybrid phi from head-only Shapley and head cosine diversity."""
    node_ids = sorted(client_weights.keys())
    if not node_ids:
        return {}, {
            "lambda": 0.5,
            "head_shapley_raw": {},
            "head_shapley": {},
            "cosine_diversity": {},
            "probe_accuracy": {},
        }

    metadata_model = model_factory()
    state_keys = list(metadata_model.state_dict().keys())
    head_indices = state_key_indices(state_keys, head_state_keys(metadata_model))
    if not head_indices:
        uniform_phi = {node_id: 1.0 for node_id in node_ids}
        return uniform_phi, {
            "lambda": 0.5,
            "head_shapley_raw": uniform_phi.copy(),
            "head_shapley": uniform_phi.copy(),
            "cosine_diversity": uniform_phi.copy(),
            "probe_accuracy": {node_id: 0.0 for node_id in node_ids},
        }

    if len(node_ids) == 1:
        head_only = _replace_selected_layers(
            reference_weights,
            _selected_layers(client_weights[node_ids[0]], head_indices),
            head_indices,
        )
        score = evaluate_on_probe(head_only, model_factory, probe_loader, device)
        single_phi = {node_ids[0]: 1.0}
        return single_phi, {
            "lambda": 0.5,
            "head_shapley_raw": {node_ids[0]: float(score)},
            "head_shapley": single_phi.copy(),
            "cosine_diversity": single_phi.copy(),
            "probe_accuracy": {node_ids[0]: float(score)},
        }

    rng = np.random.RandomState(seed)
    if K >= len(node_ids) - 1:
        strata = list(range(2, len(node_ids) + 1))
    else:
        strata = sorted({max(2, int(round(len(node_ids) * k / K))) for k in range(1, K + 1)})
        if strata[-1] != len(node_ids):
            strata.append(len(node_ids))

    phi_sum = {node_id: 0.0 for node_id in node_ids}
    phi_count = {node_id: 0 for node_id in node_ids}
    head_client_weights = {
        node_id: _selected_layers(weights, head_indices)
        for node_id, weights in client_weights.items()
    }

    for coalition_size in strata:
        for _ in range(T):
            coalition = list(rng.permutation(node_ids))[:coalition_size]
            coalition_weights = [head_client_weights[node_id] for node_id in coalition]
            coalition_sizes = [client_sizes[node_id] for node_id in coalition]
            aggregated_head = _selected_weighted_aggregate(coalition_weights, coalition_sizes)
            utility_coalition = evaluate_on_probe(
                _replace_selected_layers(reference_weights, aggregated_head, head_indices),
                model_factory,
                probe_loader,
                device,
            )
            total_examples = float(sum(coalition_sizes))
            for node_id in coalition:
                alpha_node = float(client_sizes[node_id]) / max(total_examples, 1.0)
                if alpha_node >= 1.0 - 1e-12:
                    continue
                without_node_head = leave_one_out(
                    coalition_weights=aggregated_head,
                    client_weights=head_client_weights[node_id],
                    alpha_client=alpha_node,
                )
                utility_without = evaluate_on_probe(
                    _replace_selected_layers(reference_weights, without_node_head, head_indices),
                    model_factory,
                    probe_loader,
                    device,
                )
                phi_sum[node_id] += utility_coalition - utility_without
                phi_count[node_id] += 1

    head_shapley_raw = {
        node_id: phi_sum[node_id] / max(phi_count[node_id], 1)
        for node_id in node_ids
    }
    head_shapley = normalise_shapley(head_shapley_raw, eps=eps)

    update_vectors = {}
    weighted_mean_update = None
    total_weight = float(sum(client_sizes[node_id] for node_id in node_ids))
    for node_id in node_ids:
        deltas = [
            client_weights[node_id][index].astype(np.float64) - reference_weights[index].astype(np.float64)
            for index in head_indices
        ]
        vector = np.concatenate([delta.ravel() for delta in deltas])
        update_vectors[node_id] = vector
        contribution = vector * (float(client_sizes[node_id]) / max(total_weight, 1.0))
        weighted_mean_update = contribution if weighted_mean_update is None else weighted_mean_update + contribution
    cosine_raw = {
        node_id: _cosine_distance(update_vectors[node_id], weighted_mean_update)
        for node_id in node_ids
    }
    cosine_diversity = normalise_shapley(cosine_raw, eps=eps)

    probe_accuracy: Dict[int, float] = {}
    if probe_logits is not None and probe_targets is not None:
        for node_id in node_ids:
            logits = probe_logits.get(node_id)
            if logits is not None:
                probe_accuracy[node_id] = accuracy_from_logits(logits, probe_targets)
    missing_nodes = [node_id for node_id in node_ids if node_id not in probe_accuracy]
    for node_id in missing_nodes:
        probe_accuracy[node_id] = evaluate_on_probe(
            client_weights[node_id],
            model_factory,
            probe_loader,
            device,
        )

    accuracy_values = np.asarray([probe_accuracy[node_id] for node_id in node_ids], dtype=np.float64)
    head_values = np.asarray([head_shapley[node_id] for node_id in node_ids], dtype=np.float64)
    cosine_values = np.asarray([cosine_diversity[node_id] for node_id in node_ids], dtype=np.float64)
    head_corr = max(0.0, _stable_correlation(head_values, accuracy_values))
    cosine_corr = max(0.0, _stable_correlation(cosine_values, accuracy_values))
    if head_corr + cosine_corr <= 1e-12:
        hybrid_lambda = 0.5
    else:
        hybrid_lambda = head_corr / (head_corr + cosine_corr)
    hybrid_lambda = float(np.clip(hybrid_lambda, lambda_floor, lambda_ceiling))

    hybrid_raw = {
        node_id: hybrid_lambda * head_shapley[node_id]
        + (1.0 - hybrid_lambda) * cosine_diversity[node_id]
        for node_id in node_ids
    }
    hybrid_phi = normalise_shapley(hybrid_raw, eps=eps)
    return hybrid_phi, {
        "lambda": hybrid_lambda,
        "head_shapley_raw": head_shapley_raw,
        "head_shapley": head_shapley,
        "cosine_diversity": cosine_diversity,
        "probe_accuracy": probe_accuracy,
        "head_accuracy_corr": head_corr,
        "cosine_accuracy_corr": cosine_corr,
    }


def compute_exact_shapley(
    client_weights: Dict[int, List[np.ndarray]],
    client_sizes: Dict[int, int],
    probe_loader: DataLoader,
    model_factory,
    device: torch.device,
) -> Dict[int, float]:
    """Exact Shapley by enumerating all coalitions. Use only for small N."""
    from itertools import combinations
    from math import comb

    node_ids = sorted(client_weights.keys())
    num_nodes = len(node_ids)
    if num_nodes == 0:
        return {}
    if num_nodes > 12:
        raise ValueError("compute_exact_shapley: N > 12 is infeasible")

    utility_cache: Dict[frozenset[int], float] = {frozenset(): 0.0}
    for coalition_size in range(1, num_nodes + 1):
        for subset in combinations(node_ids, coalition_size):
            coalition = list(subset)
            weights = weighted_aggregate(
                [client_weights[node_id] for node_id in coalition],
                [client_sizes[node_id] for node_id in coalition],
            )
            utility_cache[frozenset(coalition)] = evaluate_on_probe(
                weights,
                model_factory,
                probe_loader,
                device,
            )

    shapley = {node_id: 0.0 for node_id in node_ids}
    for node_id in node_ids:
        others = [other for other in node_ids if other != node_id]
        for coalition_size in range(num_nodes):
            weight = 1.0 / (num_nodes * comb(num_nodes - 1, coalition_size))
            for subset in combinations(others, coalition_size):
                coalition = frozenset(subset)
                shapley[node_id] += weight * (
                    utility_cache[coalition | {node_id}] - utility_cache[coalition]
                )
    return shapley


def normalise_shapley(
    phi: Dict[int, float],
    eps: float = 1e-6,
) -> Dict[int, float]:
    """Shift and min-max scale Shapley values into ``[eps, 1]``."""
    if not phi:
        return {}
    values = np.asarray(list(phi.values()), dtype=np.float64)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < 1e-12:
        return {node_id: 1.0 for node_id in phi}
    return {
        node_id: eps + (1.0 - eps) * (value - minimum) / (maximum - minimum)
        for node_id, value in phi.items()
    }
