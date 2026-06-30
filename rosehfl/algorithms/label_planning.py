"""
Label-aware planning utilities for paper baselines.

These planners use ground-truth label counts available in simulation to
reproduce the paper's non-ShapeFL HFL baselines:
    - Data First: optimize edge distributions towards uniform
    - SHARE: trade off communication cost and KL divergence to uniform
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class LabelAssociationResult:
    associations: Dict[int, int]
    edge_nodes: Dict[int, Set[int]]
    edge_data_sizes: Dict[int, int]
    edge_label_counts: Dict[int, np.ndarray]
    mean_kl_divergence: float
    objective_value: float


@dataclass
class LabelPlanningResult:
    selected_edges: Set[int]
    node_associations: LabelAssociationResult
    objective_value: float


def kl_divergence_to_uniform(label_counts: np.ndarray) -> float:
    """KL(P || U) where U is the uniform class distribution."""
    total = int(label_counts.sum())
    if total <= 0:
        return float("inf")
    probabilities = label_counts.astype(np.float64) / total
    num_classes = len(label_counts)
    uniform_prob = 1.0 / num_classes
    nonzero = probabilities > 0
    return float(
        np.sum(probabilities[nonzero] * np.log(probabilities[nonzero] / uniform_prob))
    )


def _mean_edge_kl(edge_label_counts: Dict[int, np.ndarray]) -> float:
    if not edge_label_counts:
        return float("inf")
    return float(
        np.mean([kl_divergence_to_uniform(counts) for counts in edge_label_counts.values()])
    )


class GreedyLabelAssociation:
    """
    Greedy association for Data First and SHARE.

    A selected edge is seeded with its own node so every edge has a valid
    empirical distribution and the planner mirrors the edge-is-a-node setup
    used in the paper.
    """

    def __init__(
        self,
        edge_aggregators: List[int],
        node_label_counts: Dict[int, np.ndarray],
        data_sizes: Dict[int, int],
        communication_costs: Optional[Dict[Tuple[int, int], float]] = None,
        kappa_c: int = 10,
        gamma: float = 2800.0,
        B_e: int = 10,
        objective_mode: str = "share",
    ):
        self.edge_aggregators = sorted(edge_aggregators)
        self.node_label_counts = node_label_counts
        self.data_sizes = data_sizes
        self.communication_costs = communication_costs or {}
        self.kappa_c = kappa_c
        self.gamma = gamma
        self.B_e = B_e
        self.objective_mode = objective_mode
        self.all_nodes = sorted(node_label_counts.keys())

    def _delta_objective(
        self,
        node_id: int,
        edge_id: int,
        edge_nodes: Dict[int, Set[int]],
        edge_sizes: Dict[int, int],
        edge_counts: Dict[int, np.ndarray],
    ) -> float:
        old_mean_kl = _mean_edge_kl(edge_counts)

        new_edge_counts = {e: counts.copy() for e, counts in edge_counts.items()}
        new_edge_sizes = dict(edge_sizes)
        new_edge_counts[edge_id] = new_edge_counts[edge_id] + self.node_label_counts[node_id]
        new_edge_sizes[edge_id] += self.data_sizes[node_id]
        new_mean_kl = _mean_edge_kl(new_edge_counts)

        delta = new_mean_kl - old_mean_kl
        if self.objective_mode == "data_first":
            return delta

        comm_cost = self.kappa_c * self.communication_costs[(node_id, edge_id)]
        return comm_cost + self.gamma * delta

    def run(self) -> LabelAssociationResult:
        edge_nodes: Dict[int, Set[int]] = {e: set() for e in self.edge_aggregators}
        edge_sizes: Dict[int, int] = {e: 0 for e in self.edge_aggregators}
        sample_shape = next(iter(self.node_label_counts.values())).shape
        edge_counts: Dict[int, np.ndarray] = {
            e: np.zeros(sample_shape, dtype=np.int64) for e in self.edge_aggregators
        }
        associations: Dict[int, int] = {}
        unassigned = set(self.all_nodes)

        # Seed every selected edge with itself to avoid empty edge distributions.
        for edge_id in self.edge_aggregators:
            if edge_id not in unassigned:
                continue
            edge_nodes[edge_id].add(edge_id)
            edge_sizes[edge_id] += self.data_sizes[edge_id]
            edge_counts[edge_id] = edge_counts[edge_id] + self.node_label_counts[edge_id]
            associations[edge_id] = edge_id
            unassigned.remove(edge_id)

        while unassigned:
            best_node = None
            best_edge = None
            best_delta = float("inf")
            for node_id in sorted(unassigned):
                for edge_id in self.edge_aggregators:
                    if len(edge_nodes[edge_id]) >= self.B_e:
                        continue
                    delta = self._delta_objective(
                        node_id, edge_id, edge_nodes, edge_sizes, edge_counts
                    )
                    if delta < best_delta:
                        best_delta = delta
                        best_node = node_id
                        best_edge = edge_id

            if best_node is None or best_edge is None:
                raise RuntimeError("Unable to find a feasible label-aware association.")

            edge_nodes[best_edge].add(best_node)
            edge_sizes[best_edge] += self.data_sizes[best_node]
            edge_counts[best_edge] = edge_counts[best_edge] + self.node_label_counts[best_node]
            associations[best_node] = best_edge
            unassigned.remove(best_node)

        mean_kl = _mean_edge_kl(edge_counts)
        if self.objective_mode == "data_first":
            objective = mean_kl
        else:
            comm_total = sum(
                self.kappa_c * self.communication_costs[(node_id, edge_id)]
                for node_id, edge_id in associations.items()
            )
            objective = comm_total + self.gamma * mean_kl

        return LabelAssociationResult(
            associations=associations,
            edge_nodes=edge_nodes,
            edge_data_sizes=edge_sizes,
            edge_label_counts=edge_counts,
            mean_kl_divergence=mean_kl,
            objective_value=float(objective),
        )


class LocalSearchLabelPlanning:
    """Local search over edge selection for Data First and SHARE."""

    def __init__(
        self,
        candidate_edges: List[int],
        all_nodes: List[int],
        node_label_counts: Dict[int, np.ndarray],
        data_sizes: Dict[int, int],
        communication_costs_ne: Dict[Tuple[int, int], float],
        communication_costs_ec: Dict[int, float],
        kappa_c: int = 10,
        gamma: float = 2800.0,
        B_e: int = 10,
        T_max: int = 30,
        objective_mode: str = "share",
    ):
        self.candidate_edges = sorted(candidate_edges)
        self.all_nodes = sorted(all_nodes)
        self.node_label_counts = node_label_counts
        self.data_sizes = data_sizes
        self.communication_costs_ne = communication_costs_ne
        self.communication_costs_ec = communication_costs_ec
        self.kappa_c = kappa_c
        self.gamma = gamma
        self.B_e = B_e
        self.T_max = T_max
        self.objective_mode = objective_mode

    def _initial_edges(self) -> Set[int]:
        min_edges = max(1, math.ceil(len(self.all_nodes) / self.B_e))
        ordered = sorted(self.candidate_edges, key=lambda edge_id: self.communication_costs_ec[edge_id])
        return set(ordered[:min_edges])

    def compute_objective(
        self, edge_set: Set[int]
    ) -> Tuple[float, Optional[LabelAssociationResult]]:
        if not edge_set:
            return float("inf"), None
        if len(edge_set) * self.B_e < len(self.all_nodes):
            return float("inf"), None

        assoc = GreedyLabelAssociation(
            edge_aggregators=list(edge_set),
            node_label_counts=self.node_label_counts,
            data_sizes=self.data_sizes,
            communication_costs=self.communication_costs_ne,
            kappa_c=self.kappa_c,
            gamma=self.gamma,
            B_e=self.B_e,
            objective_mode=self.objective_mode,
        ).run()

        if self.objective_mode == "data_first":
            objective = assoc.mean_kl_divergence
        else:
            objective = assoc.objective_value + sum(self.communication_costs_ec[e] for e in edge_set)
        return float(objective), assoc

    def run(self, initial_edges: Optional[Set[int]] = None) -> LabelPlanningResult:
        edge_set = set(initial_edges) if initial_edges is not None else self._initial_edges()
        current_objective, current_assoc = self.compute_objective(edge_set)

        for _ in range(self.T_max):
            improved = False

            for edge_id in [e for e in self.candidate_edges if e not in edge_set]:
                new_set = set(edge_set)
                new_set.add(edge_id)
                new_objective, new_assoc = self.compute_objective(new_set)
                if new_objective < current_objective:
                    edge_set, current_objective, current_assoc = new_set, new_objective, new_assoc
                    improved = True
                    break

            if len(edge_set) > 1:
                for edge_id in sorted(edge_set):
                    new_set = set(edge_set)
                    new_set.remove(edge_id)
                    new_objective, new_assoc = self.compute_objective(new_set)
                    if new_objective < current_objective:
                        edge_set, current_objective, current_assoc = new_set, new_objective, new_assoc
                        improved = True
                        break

            swapped = False
            for open_edge in [e for e in self.candidate_edges if e not in edge_set]:
                for close_edge in sorted(edge_set):
                    new_set = set(edge_set)
                    new_set.remove(close_edge)
                    new_set.add(open_edge)
                    new_objective, new_assoc = self.compute_objective(new_set)
                    if new_objective < current_objective:
                        edge_set, current_objective, current_assoc = new_set, new_objective, new_assoc
                        improved = True
                        swapped = True
                        break
                if swapped:
                    break

            if not improved:
                break

        return LabelPlanningResult(
            selected_edges=edge_set,
            node_associations=current_assoc,
            objective_value=float(current_objective),
        )


def run_label_planning(
    candidate_edges: List[int],
    all_nodes: List[int],
    node_label_counts: Dict[int, np.ndarray],
    data_sizes: Dict[int, int],
    communication_costs_ne: Dict[Tuple[int, int], float],
    communication_costs_ec: Dict[int, float],
    kappa_c: int = 10,
    gamma: float = 2800.0,
    B_e: int = 10,
    T_max: int = 30,
    objective_mode: str = "share",
) -> LabelPlanningResult:
    planner = LocalSearchLabelPlanning(
        candidate_edges=candidate_edges,
        all_nodes=all_nodes,
        node_label_counts=node_label_counts,
        data_sizes=data_sizes,
        communication_costs_ne=communication_costs_ne,
        communication_costs_ec=communication_costs_ec,
        kappa_c=kappa_c,
        gamma=gamma,
        B_e=B_e,
        T_max=T_max,
        objective_mode=objective_mode,
    )
    return planner.run()
