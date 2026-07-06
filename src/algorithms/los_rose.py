"""
RoSE-HFL Local Search Edge Selection (LoS-RoSE).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .goa import NodeAssociationResult
from .goa_rose import GreedyNodeAssociationRoSE
from .los import EdgeSelectionResult


@dataclass
class EdgeSelectionCandidate:
    """Feasible RoSE edge-selection candidate captured during local search."""

    selected_edges: Set[int]
    node_associations: NodeAssociationResult
    objective_value: float


class LocalSearchEdgeSelectionRoSE:
    """Local-search edge selection driven by the RoSE objective."""

    def __init__(
        self,
        candidate_edges: List[int],
        all_nodes: List[int],
        communication_costs_ne: Dict[Tuple[int, int], float],
        communication_costs_ec: Dict[int, float],
        phi: Dict[int, float],
        client_class_distributions: Dict[int, np.ndarray],
        data_sizes: Dict[int, int],
        kappa_c: int = 10,
        gamma: float = 2800.0,
        B_e: int = 10,
        T_max: int = 30,
        verbose: bool = False,
        initial_associations: Optional[Dict[int, int]] = None,
        warm_start_threshold: float = 0.0,
        edge_min_members: int = 0,
        edge_underfill_penalty: float = 0.0,
    ) -> None:
        self.candidate_edges = set(candidate_edges)
        self.all_nodes = set(all_nodes)
        self.c_ne = communication_costs_ne
        self.c_ec = communication_costs_ec
        self.phi = phi
        self.client_class_distributions = client_class_distributions
        self.data_sizes = data_sizes
        self.kappa_c = kappa_c
        self.gamma = gamma
        self.B_e = B_e
        self.T_max = T_max
        self.verbose = verbose
        self.initial_associations = initial_associations or {}
        self.warm_start_threshold = float(max(warm_start_threshold, 0.0))
        self.edge_min_members = int(max(edge_min_members, 0))
        self.edge_underfill_penalty = float(max(edge_underfill_penalty, 0.0))

    def compute_objective_J(
        self,
        edge_set: Set[int],
    ) -> Tuple[float, Optional[NodeAssociationResult]]:
        if not edge_set:
            return float("inf"), None
        if len(edge_set) * self.B_e < len(self.all_nodes):
            return float("inf"), None

        association = GreedyNodeAssociationRoSE(
            edge_aggregators=list(edge_set),
            communication_costs=self.c_ne,
            phi=self.phi,
            client_class_distributions=self.client_class_distributions,
            data_sizes=self.data_sizes,
            kappa_c=self.kappa_c,
            gamma=self.gamma,
            B_e=self.B_e,
            initial_associations=self.initial_associations,
            warm_start_threshold=self.warm_start_threshold,
            edge_min_members=self.edge_min_members,
            edge_underfill_penalty=self.edge_underfill_penalty,
        ).run()

        if len(association.associations) < len(self.all_nodes):
            return float("inf"), association

        objective = association.objective_value + sum(self.c_ec.get(edge_id, 0.0) for edge_id in edge_set)
        return objective, association

    @staticmethod
    def _record_candidate(
        store: Dict[frozenset[int], EdgeSelectionCandidate],
        *,
        edge_set: Set[int],
        objective_value: float,
        association: Optional[NodeAssociationResult],
    ) -> None:
        if association is None or not np.isfinite(objective_value):
            return
        key = frozenset(int(edge_id) for edge_id in edge_set)
        candidate = EdgeSelectionCandidate(
            selected_edges=set(int(edge_id) for edge_id in edge_set),
            node_associations=association,
            objective_value=float(objective_value),
        )
        existing = store.get(key)
        if existing is None or candidate.objective_value < existing.objective_value:
            store[key] = candidate

    @staticmethod
    def _top_candidates(
        store: Dict[frozenset[int], EdgeSelectionCandidate],
        max_candidates: int,
    ) -> List[EdgeSelectionCandidate]:
        ordered = sorted(
            store.values(),
            key=lambda candidate: (
                float(candidate.objective_value),
                len(candidate.selected_edges),
                tuple(sorted(int(edge_id) for edge_id in candidate.selected_edges)),
            ),
        )
        if max_candidates <= 0:
            return ordered
        return ordered[:max_candidates]

    def initialise_random(self, num_edges: int = 3) -> Set[int]:
        num_edges = min(num_edges, len(self.candidate_edges))
        return set(np.random.choice(list(self.candidate_edges), num_edges, replace=False))

    def run_with_candidates(
        self,
        initial_edges: Optional[Set[int]] = None,
        *,
        max_candidates: int = 1,
    ) -> Tuple[EdgeSelectionResult, List[EdgeSelectionCandidate]]:
        current_edges = set(initial_edges) if initial_edges is not None else self.initialise_random()
        current_objective, current_association = self.compute_objective_J(current_edges)
        candidate_store: Dict[frozenset[int], EdgeSelectionCandidate] = {}
        self._record_candidate(
            candidate_store,
            edge_set=current_edges,
            objective_value=current_objective,
            association=current_association,
        )
        if self.verbose:
            print(f"[LoS-RoSE] Initial objective = {current_objective:.4f}")

        for iteration in range(self.T_max):
            improved = False

            for edge_id in list(self.candidate_edges - current_edges):
                candidate = current_edges | {edge_id}
                candidate_objective, candidate_association = self.compute_objective_J(candidate)
                self._record_candidate(
                    candidate_store,
                    edge_set=candidate,
                    objective_value=candidate_objective,
                    association=candidate_association,
                )
                if candidate_objective < current_objective:
                    current_edges = candidate
                    current_objective = candidate_objective
                    current_association = candidate_association
                    improved = True
                    break
            if improved:
                continue

            if len(current_edges) > 1:
                for edge_id in list(current_edges):
                    candidate = current_edges - {edge_id}
                    candidate_objective, candidate_association = self.compute_objective_J(candidate)
                    self._record_candidate(
                        candidate_store,
                        edge_set=candidate,
                        objective_value=candidate_objective,
                        association=candidate_association,
                    )
                    if candidate_objective < current_objective:
                        current_edges = candidate
                        current_objective = candidate_objective
                        current_association = candidate_association
                        improved = True
                        break
            if improved:
                continue

            swap_found = False
            for new_edge in list(self.candidate_edges - current_edges):
                for old_edge in list(current_edges):
                    candidate = (current_edges - {old_edge}) | {new_edge}
                    candidate_objective, candidate_association = self.compute_objective_J(candidate)
                    self._record_candidate(
                        candidate_store,
                        edge_set=candidate,
                        objective_value=candidate_objective,
                        association=candidate_association,
                    )
                    if candidate_objective < current_objective:
                        current_edges = candidate
                        current_objective = candidate_objective
                        current_association = candidate_association
                        improved = True
                        swap_found = True
                        break
                if swap_found:
                    break

            if not improved:
                if self.verbose:
                    print(f"[LoS-RoSE] Converged at iteration {iteration + 1}")
                break

        result = EdgeSelectionResult(
            selected_edges=current_edges,
            node_associations=current_association,
            objective_value=current_objective,
        )
        return result, self._top_candidates(candidate_store, max_candidates=max_candidates)

    def run(self, initial_edges: Optional[Set[int]] = None) -> EdgeSelectionResult:
        result, _ = self.run_with_candidates(initial_edges=initial_edges, max_candidates=1)
        return result


def run_los_rose(
    candidate_edges: List[int],
    all_nodes: List[int],
    communication_costs_ne: Dict[Tuple[int, int], float],
    communication_costs_ec: Dict[int, float],
    phi: Dict[int, float],
    client_class_distributions: Dict[int, np.ndarray],
    data_sizes: Dict[int, int],
    kappa_c: int = 10,
    gamma: float = 2800.0,
    B_e: int = 10,
    T_max: int = 30,
    initial_edges: Optional[Set[int]] = None,
    initial_associations: Optional[Dict[int, int]] = None,
    warm_start_threshold: float = 0.0,
    edge_min_members: int = 0,
    edge_underfill_penalty: float = 0.0,
    verbose: bool = False,
) -> EdgeSelectionResult:
    """Convenience wrapper for LoS-RoSE."""
    return LocalSearchEdgeSelectionRoSE(
        candidate_edges=candidate_edges,
        all_nodes=all_nodes,
        communication_costs_ne=communication_costs_ne,
        communication_costs_ec=communication_costs_ec,
        phi=phi,
        client_class_distributions=client_class_distributions,
        data_sizes=data_sizes,
        kappa_c=kappa_c,
        gamma=gamma,
        B_e=B_e,
        T_max=T_max,
        verbose=verbose,
        initial_associations=initial_associations,
        warm_start_threshold=warm_start_threshold,
        edge_min_members=edge_min_members,
        edge_underfill_penalty=edge_underfill_penalty,
    ).run(initial_edges=initial_edges)


def run_los_rose_candidates(
    candidate_edges: List[int],
    all_nodes: List[int],
    communication_costs_ne: Dict[Tuple[int, int], float],
    communication_costs_ec: Dict[int, float],
    phi: Dict[int, float],
    client_class_distributions: Dict[int, np.ndarray],
    data_sizes: Dict[int, int],
    kappa_c: int = 10,
    gamma: float = 2800.0,
    B_e: int = 10,
    T_max: int = 30,
    initial_edges: Optional[Set[int]] = None,
    initial_associations: Optional[Dict[int, int]] = None,
    warm_start_threshold: float = 0.0,
    edge_min_members: int = 0,
    edge_underfill_penalty: float = 0.0,
    max_candidates: int = 1,
    verbose: bool = False,
) -> Tuple[EdgeSelectionResult, List[EdgeSelectionCandidate]]:
    """Return the best RoSE local-search result plus a bounded feasible candidate pool."""
    return LocalSearchEdgeSelectionRoSE(
        candidate_edges=candidate_edges,
        all_nodes=all_nodes,
        communication_costs_ne=communication_costs_ne,
        communication_costs_ec=communication_costs_ec,
        phi=phi,
        client_class_distributions=client_class_distributions,
        data_sizes=data_sizes,
        kappa_c=kappa_c,
        gamma=gamma,
        B_e=B_e,
        T_max=T_max,
        verbose=verbose,
        initial_associations=initial_associations,
        warm_start_threshold=warm_start_threshold,
        edge_min_members=edge_min_members,
        edge_underfill_penalty=edge_underfill_penalty,
    ).run_with_candidates(initial_edges=initial_edges, max_candidates=max_candidates)
