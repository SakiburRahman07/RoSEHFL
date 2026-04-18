"""
RoSE-HFL Greedy Node Association (GoA-RoSE).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .goa import NodeAssociationResult


class GreedyNodeAssociationRoSE:
    """Shapley- and class-balance-aware greedy node association."""

    def __init__(
        self,
        edge_aggregators: List[int],
        communication_costs: Dict[Tuple[int, int], float],
        phi: Dict[int, float],
        client_class_distributions: Dict[int, np.ndarray],
        data_sizes: Dict[int, int],
        kappa_c: int = 10,
        gamma: float = 2800.0,
        B_e: int = 10,
        entropy_eps: float = 1e-9,
        initial_associations: Optional[Dict[int, int]] = None,
        warm_start_threshold: float = 0.0,
    ) -> None:
        self.edge_aggregators = set(edge_aggregators)
        self.c_ne = communication_costs
        self.phi = phi
        self.client_class_distributions = client_class_distributions
        self.data_sizes = data_sizes
        self.kappa_c = kappa_c
        self.gamma = gamma
        self.B_e = B_e
        self.entropy_eps = entropy_eps
        self.initial_associations = initial_associations or {}
        self.warm_start_threshold = float(max(warm_start_threshold, 0.0))
        self.all_nodes = set(data_sizes.keys())

        if not client_class_distributions:
            raise ValueError("GoA-RoSE requires probe-derived client class distributions")
        first_distribution = next(iter(client_class_distributions.values()))
        self.num_classes = int(len(first_distribution))

    def _empty_state(self) -> Dict[str, object]:
        return {
            "edge_members": {edge_id: set() for edge_id in self.edge_aggregators},
            "edge_sizes": {edge_id: 0 for edge_id in self.edge_aggregators},
            "edge_phi_sum": {edge_id: 0.0 for edge_id in self.edge_aggregators},
            "edge_member_counts": {edge_id: 0 for edge_id in self.edge_aggregators},
            "edge_class_mass": {
                edge_id: np.zeros(self.num_classes, dtype=np.float64)
                for edge_id in self.edge_aggregators
            },
            "edge_terms": {edge_id: 0.0 for edge_id in self.edge_aggregators},
            "associations": {},
        }

    def _clone_state(self, state: Dict[str, object]) -> Dict[str, object]:
        return {
            "edge_members": {
                edge_id: set(nodes)
                for edge_id, nodes in state["edge_members"].items()
            },
            "edge_sizes": dict(state["edge_sizes"]),
            "edge_phi_sum": dict(state["edge_phi_sum"]),
            "edge_member_counts": dict(state["edge_member_counts"]),
            "edge_class_mass": {
                edge_id: values.copy()
                for edge_id, values in state["edge_class_mass"].items()
            },
            "edge_terms": dict(state["edge_terms"]),
            "associations": dict(state["associations"]),
        }

    def _entropy(self, class_mass: np.ndarray) -> float:
        total = float(class_mass.sum())
        if total <= self.entropy_eps:
            return 0.0
        probabilities = np.clip(class_mass / total, self.entropy_eps, None)
        probabilities = probabilities / probabilities.sum()
        return float(-(probabilities * np.log(probabilities)).sum())

    def _edge_term(
        self,
        phi_sum: float,
        member_count: int,
        class_mass: np.ndarray,
    ) -> float:
        if member_count <= 0:
            return 0.0
        phi_mean = phi_sum / max(member_count, 1)
        return float(phi_mean * self._entropy(class_mass))

    def _add_assignment(
        self,
        state: Dict[str, object],
        node_id: int,
        edge_id: int,
    ) -> None:
        phi_node = float(self.phi.get(node_id, 1.0))
        data_node = int(self.data_sizes[node_id])
        class_distribution = self.client_class_distributions[node_id].astype(np.float64)

        state["edge_members"][edge_id].add(node_id)
        state["associations"][node_id] = edge_id
        state["edge_sizes"][edge_id] += data_node
        state["edge_phi_sum"][edge_id] += phi_node
        state["edge_member_counts"][edge_id] += 1
        state["edge_class_mass"][edge_id] += data_node * class_distribution
        state["edge_terms"][edge_id] = self._edge_term(
            state["edge_phi_sum"][edge_id],
            state["edge_member_counts"][edge_id],
            state["edge_class_mass"][edge_id],
        )

    def _remove_assignment(
        self,
        state: Dict[str, object],
        node_id: int,
        edge_id: int,
    ) -> None:
        phi_node = float(self.phi.get(node_id, 1.0))
        data_node = int(self.data_sizes[node_id])
        class_distribution = self.client_class_distributions[node_id].astype(np.float64)

        state["edge_members"][edge_id].remove(node_id)
        state["associations"].pop(node_id, None)
        state["edge_sizes"][edge_id] -= data_node
        state["edge_phi_sum"][edge_id] -= phi_node
        state["edge_member_counts"][edge_id] -= 1
        state["edge_class_mass"][edge_id] -= data_node * class_distribution
        state["edge_terms"][edge_id] = self._edge_term(
            state["edge_phi_sum"][edge_id],
            state["edge_member_counts"][edge_id],
            state["edge_class_mass"][edge_id],
        )

    def _objective(self, state: Dict[str, object]) -> float:
        communication_total = 0.0
        for node_id, edge_id in state["associations"].items():
            communication_total += self.kappa_c * self.c_ne.get((node_id, edge_id), 0.0)
        inverse_num_edges = 1.0 / max(len(self.edge_aggregators), 1)
        diversity_total = sum(state["edge_terms"].values()) * inverse_num_edges
        return communication_total - self.gamma * diversity_total

    def _seed_initial_assignments(self, state: Dict[str, object]) -> Sequence[int]:
        unassigned = set(self.all_nodes)
        for node_id, edge_id in self.initial_associations.items():
            if node_id not in self.all_nodes or edge_id not in self.edge_aggregators:
                continue
            if len(state["edge_members"][edge_id]) >= self.B_e:
                continue
            if node_id not in self.client_class_distributions:
                continue
            self._add_assignment(state, int(node_id), int(edge_id))
            unassigned.discard(int(node_id))
        return sorted(unassigned)

    def _assign_remaining_greedily(
        self,
        state: Dict[str, object],
        unassigned: Sequence[int],
    ) -> None:
        remaining = set(unassigned)
        inverse_num_edges = 1.0 / max(len(self.edge_aggregators), 1)

        while remaining:
            best_node = None
            best_edge = None
            best_delta = float("inf")

            for node_id in remaining:
                phi_node = float(self.phi.get(node_id, 1.0))
                data_node = int(self.data_sizes[node_id])
                class_distribution = self.client_class_distributions[node_id].astype(np.float64)

                for edge_id in self.edge_aggregators:
                    if len(state["edge_members"][edge_id]) >= self.B_e:
                        continue

                    communication_cost = self.kappa_c * self.c_ne.get((node_id, edge_id), float("inf"))
                    new_phi_sum = state["edge_phi_sum"][edge_id] + phi_node
                    new_member_count = state["edge_member_counts"][edge_id] + 1
                    new_class_mass = state["edge_class_mass"][edge_id] + data_node * class_distribution
                    new_term = self._edge_term(new_phi_sum, new_member_count, new_class_mass)
                    delta_term = new_term - state["edge_terms"][edge_id]
                    delta_objective = communication_cost - self.gamma * inverse_num_edges * delta_term

                    if delta_objective < best_delta:
                        best_delta = delta_objective
                        best_node = node_id
                        best_edge = edge_id

            if best_node is None or best_edge is None:
                break

            self._add_assignment(state, int(best_node), int(best_edge))
            remaining.remove(int(best_node))

    def _warm_start_refine(self, state: Dict[str, object]) -> Dict[str, object]:
        if not self.initial_associations:
            return state

        current_state = state
        current_objective = self._objective(current_state)
        max_passes = max(len(self.all_nodes), 1)

        for _ in range(max_passes):
            improved = False
            for node_id in sorted(self.all_nodes):
                current_edge = current_state["associations"].get(node_id)
                if current_edge is None:
                    continue

                best_state = None
                best_objective = current_objective
                for edge_id in self.edge_aggregators:
                    if edge_id == current_edge:
                        continue
                    if len(current_state["edge_members"][edge_id]) >= self.B_e:
                        continue

                    candidate_state = self._clone_state(current_state)
                    self._remove_assignment(candidate_state, node_id, current_edge)
                    self._add_assignment(candidate_state, node_id, edge_id)
                    candidate_objective = self._objective(candidate_state)
                    improvement = current_objective - candidate_objective

                    if (
                        improvement > self.warm_start_threshold
                        and candidate_objective < best_objective
                    ):
                        best_state = candidate_state
                        best_objective = candidate_objective

                if best_state is not None:
                    current_state = best_state
                    current_objective = best_objective
                    improved = True

            if not improved:
                break

        return current_state

    def run(self) -> NodeAssociationResult:
        state = self._empty_state()
        unassigned = self._seed_initial_assignments(state)
        self._assign_remaining_greedily(state, unassigned)
        state = self._warm_start_refine(state)

        return NodeAssociationResult(
            associations={
                int(node_id): int(edge_id)
                for node_id, edge_id in state["associations"].items()
            },
            edge_nodes={
                int(edge_id): set(int(node_id) for node_id in nodes)
                for edge_id, nodes in state["edge_members"].items()
            },
            edge_data_sizes={
                int(edge_id): int(size)
                for edge_id, size in state["edge_sizes"].items()
            },
            objective_value=float(self._objective(state)),
            edge_diversity_sums={
                int(edge_id): float(term)
                for edge_id, term in state["edge_terms"].items()
            },
        )


def run_goa_rose(
    edge_aggregators: List[int],
    nodes: List[int],
    communication_costs: Dict[Tuple[int, int], float],
    phi: Dict[int, float],
    client_class_distributions: Dict[int, np.ndarray],
    data_sizes: Dict[int, int],
    kappa_c: int = 10,
    gamma: float = 2800.0,
    B_e: int = 10,
    initial_associations: Optional[Dict[int, int]] = None,
    warm_start_threshold: float = 0.0,
) -> NodeAssociationResult:
    """Convenience wrapper for GoA-RoSE."""
    return GreedyNodeAssociationRoSE(
        edge_aggregators=edge_aggregators,
        communication_costs=communication_costs,
        phi=phi,
        client_class_distributions=client_class_distributions,
        data_sizes=data_sizes,
        kappa_c=kappa_c,
        gamma=gamma,
        B_e=B_e,
        initial_associations=initial_associations,
        warm_start_threshold=warm_start_threshold,
    ).run()
