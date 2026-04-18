"""
q-FedAvg-inspired flat baseline.
"""

from __future__ import annotations

from typing import Dict, Tuple

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from shapefl.strategy import FedAvgFlatStrategy, _weighted_average


class QFedAvgFlatStrategy(FedAvgFlatStrategy):
    """Simple q-FedAvg-inspired weighting based on local loss."""

    def __init__(self, *args, q: float = 2.0, eps: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = float(q)
        self.eps = float(eps)

    def aggregate_fit(self, server_round, results, failures):
        weights_list, coefficients = [], []
        for _, fit_res in results:
            weights_list.append(parameters_to_ndarrays(fit_res.parameters))
            local_loss = float(fit_res.metrics.get("loss", 1.0))
            coefficients.append(float(fit_res.num_examples) * (local_loss + self.eps) ** self.q)
        aggregate = _weighted_average(weights_list, coefficients)
        self.global_parameters = ndarrays_to_parameters(aggregate)
        self.cumulative_cost_gb += self.per_round_cost_gb
        self.completed_local_epochs += self._current_round_local_epochs
        return self.global_parameters, {"round": server_round, "q": self.q}

