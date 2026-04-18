"""
Error-feedback top-k model-update compression utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import numpy as np


@dataclass
class CompressedLayer:
    """Compressed representation of a single model-update layer."""

    kind: str
    values: np.ndarray
    indices: Optional[np.ndarray] = None
    shape: Optional[tuple[int, ...]] = None

    @property
    def payload_bytes(self) -> int:
        total = int(self.values.nbytes)
        if self.indices is not None:
            total += int(self.indices.nbytes)
        if self.shape is not None and self.kind == "sparse":
            total += int(np.asarray(self.shape, dtype=np.int32).nbytes)
        return total


@dataclass
class CompressionResult:
    """Compression result for a full model update."""

    encoded_layers: List[CompressedLayer]
    reconstructed_delta: List[np.ndarray]
    reconstructed_weights: List[np.ndarray]
    residuals: List[np.ndarray]
    payload_bytes: int
    dense_payload_bytes: int

    @property
    def realised_ratio(self) -> float:
        if self.dense_payload_bytes <= 0:
            return 1.0
        return float(self.payload_bytes) / float(self.dense_payload_bytes)


def dense_payload_num_bytes(weights: Sequence[np.ndarray]) -> int:
    """Return the raw dense payload size for a weight list."""
    return int(sum(int(layer.nbytes) for layer in weights))


def zero_residuals_like(weights: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Return zero residual buffers matching a weight list."""
    residuals: List[np.ndarray] = []
    for layer in weights:
        if np.issubdtype(layer.dtype, np.floating):
            residuals.append(np.zeros_like(layer, dtype=np.float32))
        else:
            residuals.append(np.zeros(layer.shape, dtype=np.float32))
    return residuals


def _normalise_residuals(
    weights: Sequence[np.ndarray],
    residuals: Optional[Sequence[np.ndarray]],
) -> List[np.ndarray]:
    if residuals is None:
        return zero_residuals_like(weights)
    if len(residuals) != len(weights):
        raise ValueError("compression residual length mismatch")
    normalised: List[np.ndarray] = []
    for layer, residual in zip(weights, residuals):
        cast = np.asarray(residual, dtype=np.float32)
        if cast.shape != layer.shape:
            raise ValueError("compression residual shape mismatch")
        normalised.append(cast.copy())
    return normalised


def _dense_layer(delta: np.ndarray) -> CompressedLayer:
    return CompressedLayer(kind="dense", values=np.asarray(delta).copy())


def decompress_layers(
    encoded_layers: Sequence[CompressedLayer],
    reference_weights: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """Reconstruct model weights from compressed deltas and references."""
    if len(encoded_layers) != len(reference_weights):
        raise ValueError("compressed/reference length mismatch")
    reconstructed: List[np.ndarray] = []
    for layer, reference in zip(encoded_layers, reference_weights):
        if layer.kind == "dense":
            delta = layer.values.astype(reference.dtype, copy=False)
        elif layer.kind == "sparse":
            flat = np.zeros(reference.size, dtype=np.float32)
            if layer.indices is not None and layer.values.size:
                flat[layer.indices.astype(np.int64, copy=False)] = layer.values.astype(np.float32)
            delta = flat.reshape(reference.shape).astype(reference.dtype, copy=False)
        else:
            raise ValueError(f"unsupported compressed layer kind: {layer.kind}")
        reconstructed.append(reference + delta)
    return reconstructed


def compress_weight_update(
    *,
    reference_weights: Sequence[np.ndarray],
    target_weights: Sequence[np.ndarray],
    keep_ratio: float,
    residuals: Optional[Sequence[np.ndarray]] = None,
    dense_layer_indices: Optional[Set[int]] = None,
) -> CompressionResult:
    """Compress ``target_weights - reference_weights`` with error feedback."""
    if len(reference_weights) != len(target_weights):
        raise ValueError("compress_weight_update: reference/target length mismatch")

    dense_indices = set(int(index) for index in (dense_layer_indices or set()))
    ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    dense_payload = dense_payload_num_bytes(target_weights)
    residual_buffers = _normalise_residuals(target_weights, residuals)

    encoded_layers: List[CompressedLayer] = []
    reconstructed_delta: List[np.ndarray] = []
    reconstructed_weights: List[np.ndarray] = []
    next_residuals: List[np.ndarray] = []
    payload_bytes = 0

    for layer_idx, (reference, target, residual) in enumerate(
        zip(reference_weights, target_weights, residual_buffers)
    ):
        delta = target.astype(np.float32) - reference.astype(np.float32)
        combined = delta + residual
        if (
            layer_idx in dense_indices
            or not np.issubdtype(target.dtype, np.floating)
            or reference.size <= 1
            or ratio >= 1.0
        ):
            encoded = _dense_layer(combined.astype(target.dtype, copy=False))
            reconstructed = combined.astype(target.dtype, copy=False)
            next_residual = np.zeros_like(combined, dtype=np.float32)
        else:
            flat = combined.reshape(-1).astype(np.float32, copy=False)
            nonzero = np.flatnonzero(flat)
            if nonzero.size == 0 or ratio <= 0.0:
                encoded = CompressedLayer(
                    kind="sparse",
                    values=np.zeros((0,), dtype=np.float16),
                    indices=np.zeros((0,), dtype=np.int32),
                    shape=target.shape,
                )
                reconstructed_flat = np.zeros_like(flat, dtype=np.float32)
            else:
                k = int(np.ceil(ratio * flat.size))
                k = max(1, min(k, nonzero.size))
                if k >= flat.size:
                    encoded = _dense_layer(flat.reshape(target.shape).astype(target.dtype, copy=False))
                    reconstructed = flat.reshape(target.shape).astype(target.dtype, copy=False)
                    next_residual = np.zeros_like(combined, dtype=np.float32)
                    encoded_layers.append(encoded)
                    reconstructed_delta.append(reconstructed.copy())
                    reconstructed_weights.append(reference + reconstructed)
                    next_residuals.append(next_residual)
                    payload_bytes += encoded.payload_bytes
                    continue
                magnitudes = np.abs(flat)
                chosen = np.argpartition(magnitudes, -k)[-k:]
                chosen = np.sort(chosen.astype(np.int32, copy=False))
                values = flat[chosen].astype(np.float16)
                encoded = CompressedLayer(
                    kind="sparse",
                    values=values,
                    indices=chosen,
                    shape=target.shape,
                )
                reconstructed_flat = np.zeros_like(flat, dtype=np.float32)
                reconstructed_flat[chosen.astype(np.int64, copy=False)] = values.astype(np.float32)
            reconstructed = reconstructed_flat.reshape(target.shape).astype(target.dtype, copy=False)
            next_residual = (combined - reconstructed.astype(np.float32)).astype(np.float32, copy=False)

        encoded_layers.append(encoded)
        reconstructed_delta.append(reconstructed.copy())
        reconstructed_weights.append(reference + reconstructed)
        next_residuals.append(next_residual.copy())
        payload_bytes += encoded.payload_bytes

    return CompressionResult(
        encoded_layers=encoded_layers,
        reconstructed_delta=reconstructed_delta,
        reconstructed_weights=reconstructed_weights,
        residuals=next_residuals,
        payload_bytes=int(payload_bytes),
        dense_payload_bytes=int(dense_payload),
    )


def scaled_cost_from_payload(
    base_cost_gb: float,
    payload_bytes: int,
    dense_payload_bytes: int,
) -> float:
    """Scale a dense-link cost by the realised payload ratio."""
    if dense_payload_bytes <= 0:
        return float(base_cost_gb)
    return float(base_cost_gb) * (float(payload_bytes) / float(dense_payload_bytes))
