import unittest

import numpy as np

from rosehfl.algorithms.los_rose import run_los_rose


class RoSEPlannerTests(unittest.TestCase):
    def test_fill_regularisation_discourages_singleton_edges(self):
        candidate_edges = [0, 1, 2, 3]
        all_nodes = [0, 1, 2, 3]
        phi = {node_id: 1.0 for node_id in all_nodes}
        client_class_distributions = {
            0: np.array([1.0, 0.0]),
            1: np.array([1.0, 0.0]),
            2: np.array([0.0, 1.0]),
            3: np.array([0.0, 1.0]),
        }
        data_sizes = {node_id: 1 for node_id in all_nodes}
        communication_costs_ne = {
            (0, 0): 0.0, (0, 1): 1.0, (0, 2): 10.0, (0, 3): 10.0,
            (1, 0): 1.0, (1, 1): 0.0, (1, 2): 10.0, (1, 3): 10.0,
            (2, 0): 10.0, (2, 1): 10.0, (2, 2): 0.0, (2, 3): 1.0,
            (3, 0): 10.0, (3, 1): 10.0, (3, 2): 1.0, (3, 3): 0.0,
        }
        communication_costs_ec = {edge_id: 0.5 for edge_id in candidate_edges}

        unregularised = run_los_rose(
            candidate_edges=candidate_edges,
            all_nodes=all_nodes,
            communication_costs_ne=communication_costs_ne,
            communication_costs_ec=communication_costs_ec,
            phi=phi,
            client_class_distributions=client_class_distributions,
            data_sizes=data_sizes,
            kappa_c=1,
            gamma=0.0,
            B_e=2,
            T_max=10,
            initial_edges=set(candidate_edges),
            edge_min_members=0,
            edge_underfill_penalty=0.0,
            verbose=False,
        )
        regularised = run_los_rose(
            candidate_edges=candidate_edges,
            all_nodes=all_nodes,
            communication_costs_ne=communication_costs_ne,
            communication_costs_ec=communication_costs_ec,
            phi=phi,
            client_class_distributions=client_class_distributions,
            data_sizes=data_sizes,
            kappa_c=1,
            gamma=0.0,
            B_e=2,
            T_max=10,
            initial_edges=set(candidate_edges),
            edge_min_members=2,
            edge_underfill_penalty=2.0,
            verbose=False,
        )

        self.assertGreater(len(unregularised.selected_edges), len(regularised.selected_edges))
        self.assertEqual(len(regularised.selected_edges), 2)
        occupied_edges = [
            nodes for nodes in regularised.node_associations.edge_nodes.values() if nodes
        ]
        self.assertTrue(all(len(nodes) >= 2 for nodes in occupied_edges))


if __name__ == "__main__":
    unittest.main()
