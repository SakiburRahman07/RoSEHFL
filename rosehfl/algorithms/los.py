"""
Local Search Edge Selection (LoS) Algorithm
============================================
Implementation of Algorithm 2 from the ShapeFL paper.

Objective (Eq. 19):
    J(E_s) = J_m(E_s) + Σ_{e∈E_s} c_ec

Local search operations: open, close, swap.
Reference: Paper Section IV-C, Algorithm 2
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

from .goa import GreedyNodeAssociation, NodeAssociationResult


@dataclass
class EdgeSelectionResult:
    """Result of the LoS algorithm."""
    selected_edges: Set[int]
    node_associations: NodeAssociationResult
    objective_value: float


class LocalSearchEdgeSelection:
    """LoS (Algorithm 2) — select the optimal set of edge aggregators."""

    def __init__(
        self,
        candidate_edges: List[int],
        all_nodes: List[int],
        communication_costs_ne: Dict[Tuple[int, int], float],
        communication_costs_ec: Dict[int, float],
        similarity_matrix: np.ndarray,
        data_sizes: Dict[int, int],
        kappa_c: int = 10,
        gamma: float = 2800.0,
        B_e: int = 10,
        T_max: int = 30,
    ):
        self.N_c = set(candidate_edges)
        self.N = set(all_nodes)
        self.c_ne = communication_costs_ne
        self.c_ec = communication_costs_ec
        self.S = similarity_matrix
        self.D = data_sizes
        self.kappa_c = kappa_c
        self.gamma = gamma
        self.B_e = B_e
        self.T_max = T_max

    def compute_objective_J(
        self, edge_set: Set[int]
    ) -> Tuple[float, Optional[NodeAssociationResult]]:
        if len(edge_set) == 0:
            return float("inf"), None

        total_capacity = len(edge_set) * self.B_e
        if total_capacity < len(self.N):
            return float("inf"), None

        goa = GreedyNodeAssociation(
            edge_aggregators=list(edge_set),
            communication_costs=self.c_ne,
            similarity_matrix=self.S,
            data_sizes=self.D,
            kappa_c=self.kappa_c,
            gamma=self.gamma,
            B_e=self.B_e,
        )
        association_result = goa.run()

        if len(association_result.associations) < len(self.N):
            return float("inf"), association_result

        J_m = association_result.objective_value
        edge_cloud_cost = sum(self.c_ec.get(e, 0) for e in edge_set)
        J = J_m + edge_cloud_cost
        return J, association_result

    def initialize_random(self, num_edges: int = 3) -> Set[int]:
        num_edges = min(num_edges, len(self.N_c))
        return set(np.random.choice(list(self.N_c), num_edges, replace=False))

    def run(self, initial_edges: Optional[Set[int]] = None) -> EdgeSelectionResult:
        if initial_edges is None:
            E_s = self.initialize_random()
        else:
            E_s = set(initial_edges)

        J_current, assoc_current = self.compute_objective_J(E_s)
        print(f"Initial: {len(E_s)} edges, J = {J_current:.2f}")

        # Paper Algorithm 2: each iteration tries ALL three operations
        # (open, close, swap) in sequence.  The `break` inside each
        # operation exits only its own foreach loop; execution then
        # falls through to the next operation.  The outer `repeat`
        # terminates after T iterations or when NO operation improved J.
        for t in range(self.T_max):
            any_improved = False

            # ── 'open' operation ──
            for e in list(self.N_c - E_s):
                E_new = E_s | {e}
                J_new, assoc_new = self.compute_objective_J(E_new)
                if J_new < J_current:
                    E_s, J_current, assoc_current = E_new, J_new, assoc_new
                    any_improved = True
                    print(f"  [open] Added edge {e}, J = {J_current:.2f}")
                    break

            # ── 'close' operation ──
            if len(E_s) > 1:
                for e in list(E_s):
                    E_new = E_s - {e}
                    J_new, assoc_new = self.compute_objective_J(E_new)
                    if J_new < J_current:
                        E_s, J_current, assoc_current = E_new, J_new, assoc_new
                        any_improved = True
                        print(f"  [close] Removed edge {e}, J = {J_current:.2f}")
                        break

            # ── 'swap' operation ──
            swap_done = False
            for e_new in list(self.N_c - E_s):
                for e_old in list(E_s):
                    E_new = (E_s - {e_old}) | {e_new}
                    J_new, assoc_new = self.compute_objective_J(E_new)
                    if J_new < J_current:
                        E_s, J_current, assoc_current = E_new, J_new, assoc_new
                        any_improved = True
                        swap_done = True
                        print(f"  [swap] {e_old} -> {e_new}, J = {J_current:.2f}")
                        break
                if swap_done:
                    break

            if not any_improved:
                print(f"Converged at iteration {t + 1}")
                break

        return EdgeSelectionResult(
            selected_edges=E_s,
            node_associations=assoc_current,
            objective_value=J_current,
        )


def run_los(
    candidate_edges: List[int],
    all_nodes: List[int],
    communication_costs_ne: Dict[Tuple[int, int], float],
    communication_costs_ec: Dict[int, float],
    similarity_matrix: np.ndarray,
    data_sizes: Dict[int, int],
    kappa_c: int = 10,
    gamma: float = 2800.0,
    B_e: int = 10,
    T_max: int = 30,
    initial_edges: Optional[Set[int]] = None,
) -> EdgeSelectionResult:
    """Convenience wrapper for LoS."""
    los = LocalSearchEdgeSelection(
        candidate_edges=candidate_edges,
        all_nodes=all_nodes,
        communication_costs_ne=communication_costs_ne,
        communication_costs_ec=communication_costs_ec,
        similarity_matrix=similarity_matrix,
        data_sizes=data_sizes,
        kappa_c=kappa_c,
        gamma=gamma,
        B_e=B_e,
        T_max=T_max,
    )
    return los.run(initial_edges=initial_edges)
