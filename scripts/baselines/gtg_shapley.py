"""
GTG-Shapley-inspired flat baseline for comparison with RoSE-HFL.
"""

from __future__ import annotations

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from rosehfl.strategy import FedAvgFlatStrategy, _weighted_average
from rosehfl.utils.shapley import compute_smc_shapley, normalise_shapley


class GTGShapleyFlatStrategy(FedAvgFlatStrategy):
    """Flat strategy that weights client aggregation by probe-set Shapley."""

    def __init__(
        self,
        *args,
        probe_loader,
        model_factory,
        server_device,
        shapley_T: int = 4,
        shapley_K: int = 6,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.probe_loader = probe_loader
        self.model_factory = model_factory
        self.server_device = server_device
        self.shapley_T = int(shapley_T)
        self.shapley_K = int(shapley_K)
        self.seed = int(seed)
        self.shapley_history = []

    def aggregate_fit(self, server_round, results, failures):
        client_weights = {}
        client_sizes = {}
        for index, (_, fit_res) in enumerate(results):
            client_weights[index] = parameters_to_ndarrays(fit_res.parameters)
            client_sizes[index] = int(fit_res.num_examples)

        phi_raw = compute_smc_shapley(
            client_weights=client_weights,
            client_sizes=client_sizes,
            probe_loader=self.probe_loader,
            model_factory=self.model_factory,
            device=self.server_device,
            T=self.shapley_T,
            K=self.shapley_K,
            seed=self.seed + server_round,
        )
        phi = normalise_shapley(phi_raw)
        coefficients = [
            max(float(phi[index]), 1e-6) * float(client_sizes[index])
            for index in range(len(results))
        ]
        aggregate = _weighted_average(
            [client_weights[index] for index in range(len(results))],
            coefficients,
        )
        self.global_parameters = ndarrays_to_parameters(aggregate)
        self.cumulative_cost_gb += self.per_round_cost_gb
        self.completed_local_epochs += self._current_round_local_epochs
        self.shapley_history.append({"round": server_round, "phi": phi})
        return self.global_parameters, {"round": server_round}

