import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from shapefl.utils.shapley import compute_exact_shapley, compute_smc_shapley


class TinyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class ShapleyTests(unittest.TestCase):
    def test_smc_shapley_tracks_exact_ranking(self):
        probe_x = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
            dtype=torch.float32,
        )
        probe_y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        probe_loader = DataLoader(TensorDataset(probe_x, probe_y), batch_size=2, shuffle=False)

        def model_factory():
            return TinyLinear()

        client_weights = {
            0: [
                np.array([[4.0, -2.0], [-4.0, 2.0]], dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32),
            ],
            1: [
                np.array([[2.0, -1.0], [-2.0, 1.0]], dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32),
            ],
            2: [
                np.array([[0.5, 0.2], [0.5, -0.2]], dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32),
            ],
            3: [
                np.array([[-2.0, 2.0], [2.0, -2.0]], dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32),
            ],
            4: [
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32),
            ],
        }
        client_sizes = {node_id: 10 for node_id in client_weights}

        exact = compute_exact_shapley(
            client_weights=client_weights,
            client_sizes=client_sizes,
            probe_loader=probe_loader,
            model_factory=model_factory,
            device=torch.device("cpu"),
        )
        estimate = compute_smc_shapley(
            client_weights=client_weights,
            client_sizes=client_sizes,
            probe_loader=probe_loader,
            model_factory=model_factory,
            device=torch.device("cpu"),
            T=8,
            K=4,
            seed=7,
        )

        exact_ranking = sorted(exact, key=exact.get, reverse=True)
        estimate_ranking = sorted(estimate, key=estimate.get, reverse=True)
        overlap = sum(a == b for a, b in zip(exact_ranking, estimate_ranking))
        self.assertGreaterEqual(overlap, 4)


if __name__ == "__main__":
    unittest.main()

