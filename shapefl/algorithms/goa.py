"""
Greedy Node Association (GoA) Algorithm
=======================================
Implementation of Algorithm 1 from the ShapeFL paper.

Objective (Eq. 14):
    min_Y  κ_c · Σ y_ne·c_ne  −  γ/|E| · Σ_{e∈E} [1/C(D_e,2)] · Σ_{i,j∈M_e} S_ij·D_i·D_j

Reference: Paper Section IV-C, Algorithm 1
"""

import numpy as np
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass


@dataclass
class NodeAssociationResult:
    """Result of the GoA algorithm."""
    associations: Dict[int, int]
    edge_nodes: Dict[int, Set[int]]
    edge_data_sizes: Dict[int, int]
    objective_value: float
    edge_diversity_sums: Dict[int, float]


class GreedyNodeAssociation:
    """
    GoA (Algorithm 1) — associate each distributed computing node with an
    optimal edge aggregator.
    """

    def __init__(
        self,
        edge_aggregators: List[int],
        communication_costs: Dict[Tuple[int, int], float],
        similarity_matrix: np.ndarray,
        data_sizes: Dict[int, int],
        kappa_c: int = 10,
        gamma: float = 2800.0,
        B_e: int = 10,
    ):
        self.edge_aggregators = set(edge_aggregators)
        self.c_ne = communication_costs
        self.S = similarity_matrix
        self.D = data_sizes
        self.kappa_c = kappa_c
        self.gamma = gamma
        self.B_e = B_e
        self.all_nodes = set(data_sizes.keys())

    @staticmethod
    def _comb2(d: float) -> float:
        return d * (d - 1) / 2.0 if d > 1 else 0.0

    def run(self) -> NodeAssociationResult:
        M_e: Dict[int, Set[int]] = {e: set() for e in self.edge_aggregators}
        D_e: Dict[int, int] = {e: 0 for e in self.edge_aggregators}
        pair_sums: Dict[int, float] = {e: 0.0 for e in self.edge_aggregators}
        associations: Dict[int, int] = {}
        N_a = set(self.all_nodes)
        num_edges = len(self.edge_aggregators)

        while len(N_a) > 0:
            best_node = None
            best_edge = None
            best_delta_J = float("inf")
            best_new_pairs_sum = 0.0

            for n in N_a:
                D_n = self.D[n]
                for e in self.edge_aggregators:
                    if len(M_e[e]) >= self.B_e:
                        continue

                    comm_cost = self.kappa_c * self.c_ne.get((n, e), float("inf"))

                    new_pairs_sum = 0.0
                    for m in M_e[e]:
                        new_pairs_sum += self.S[n, m] * D_n * self.D[m]

                    old_comb = self._comb2(D_e[e])
                    old_term = pair_sums[e] / old_comb if old_comb > 0 else 0.0

                    new_sum = pair_sums[e] + new_pairs_sum
                    new_de = D_e[e] + D_n
                    new_comb = self._comb2(new_de)
                    new_term = new_sum / new_comb if new_comb > 0 else 0.0

                    delta_S = new_term - old_term
                    delta_J = comm_cost - self.gamma * (1.0 / num_edges) * delta_S

                    if delta_J < best_delta_J:
                        best_delta_J = delta_J
                        best_node = n
                        best_edge = e
                        best_new_pairs_sum = new_pairs_sum

            if best_node is None:
                print(f"Warning: Could not assign {len(N_a)} nodes — all edges at capacity")
                break

            pair_sums[best_edge] += best_new_pairs_sum
            M_e[best_edge].add(best_node)
            D_e[best_edge] += self.D[best_node]
            N_a.remove(best_node)
            associations[best_node] = best_edge

        # Final objective J_m (Eq. 14) — sum over ALL nodes n ∈ N
        comm_total = 0.0
        for node, edge in associations.items():
            comm_total += self.kappa_c * self.c_ne.get((node, edge), 0)

        diversity_total = 0.0
        for e in self.edge_aggregators:
            comb = self._comb2(D_e[e])
            if comb > 0:
                diversity_total += pair_sums[e] / comb
        if num_edges > 0:
            diversity_total /= num_edges

        J_m = comm_total - self.gamma * diversity_total

        return NodeAssociationResult(
            associations=associations,
            edge_nodes=M_e,
            edge_data_sizes=D_e,
            objective_value=J_m,
            edge_diversity_sums=pair_sums,
        )


def run_goa(
    edge_aggregators: List[int],
    nodes: List[int],
    communication_costs: Dict[Tuple[int, int], float],
    similarity_matrix: np.ndarray,
    data_sizes: Dict[int, int],
    kappa_c: int = 10,
    gamma: float = 2800.0,
    B_e: int = 10,
) -> NodeAssociationResult:
    """Convenience wrapper for GoA."""
    goa = GreedyNodeAssociation(
        edge_aggregators=edge_aggregators,
        communication_costs=communication_costs,
        similarity_matrix=similarity_matrix,
        data_sizes=data_sizes,
        kappa_c=kappa_c,
        gamma=gamma,
        B_e=B_e,
    )
    return goa.run()
