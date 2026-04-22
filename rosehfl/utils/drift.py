"""
Page-Hinkley drift detection utilities for RoSE-HFL.

The detector is maintained per edge aggregator and is driven by the
L2 distance between the current edge model and the anchor model from the
last planning event.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def weights_l2_distance(
    weights_a: Sequence[np.ndarray],
    weights_b: Sequence[np.ndarray],
) -> float:
    """Return the L2 distance between two model-weight vectors."""
    if len(weights_a) != len(weights_b):
        raise ValueError("weights_l2_distance: weight layouts do not match")
    total = 0.0
    for arr_a, arr_b in zip(weights_a, weights_b):
        diff = arr_a.astype(np.float64) - arr_b.astype(np.float64)
        total += float(np.sum(diff * diff))
    return float(np.sqrt(total))


@dataclass
class PageHinkleyState:
    """Running state for a single Page-Hinkley detector."""

    mean: float = 0.0
    cumulative: float = 0.0
    minimum: float = 0.0
    num_updates: int = 0
    last_value: float = 0.0
    last_statistic: float = 0.0
    triggered: bool = False

    def to_dict(self) -> Dict[str, float | int | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, float | int | bool]) -> "PageHinkleyState":
        return cls(
            mean=float(payload.get("mean", 0.0)),
            cumulative=float(payload.get("cumulative", 0.0)),
            minimum=float(payload.get("minimum", 0.0)),
            num_updates=int(payload.get("num_updates", 0)),
            last_value=float(payload.get("last_value", 0.0)),
            last_statistic=float(payload.get("last_statistic", 0.0)),
            triggered=bool(payload.get("triggered", False)),
        )


class PageHinkleyBank:
    """Bank of Page-Hinkley detectors keyed by edge id."""

    def __init__(
        self,
        delta: float = 1e-3,
        threshold: float = 0.5,
        initial_edges: Iterable[int] | None = None,
    ) -> None:
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.states: Dict[int, PageHinkleyState] = {}
        if initial_edges is not None:
            self.reset(initial_edges)

    def reset(self, edge_ids: Iterable[int]) -> None:
        self.states = {int(edge_id): PageHinkleyState() for edge_id in edge_ids}

    def ensure_edges(self, edge_ids: Iterable[int]) -> None:
        for edge_id in edge_ids:
            self.states.setdefault(int(edge_id), PageHinkleyState())

    def update(self, edge_id: int, value: float) -> Tuple[float, bool]:
        state = self.states.setdefault(int(edge_id), PageHinkleyState())
        state.num_updates += 1
        state.last_value = float(value)
        state.mean += (value - state.mean) / state.num_updates
        state.cumulative += value - state.mean - self.delta
        state.minimum = min(state.minimum, state.cumulative)
        state.last_statistic = state.cumulative - state.minimum
        state.triggered = state.last_statistic > self.threshold
        return state.last_statistic, state.triggered

    def update_many(
        self,
        values: Dict[int, float],
    ) -> Tuple[Dict[int, float], List[int]]:
        statistics: Dict[int, float] = {}
        triggered_edges: List[int] = []
        for edge_id, value in values.items():
            statistic, triggered = self.update(int(edge_id), float(value))
            statistics[int(edge_id)] = statistic
            if triggered:
                triggered_edges.append(int(edge_id))
        return statistics, sorted(triggered_edges)

    def snapshot(self) -> Dict[int, Dict[str, float | int | bool]]:
        return {edge_id: state.to_dict() for edge_id, state in self.states.items()}

    def load_snapshot(self, payload: Dict[int | str, Dict[str, float | int | bool]]) -> None:
        self.states = {
            int(edge_id): PageHinkleyState.from_dict(state)
            for edge_id, state in payload.items()
        }

