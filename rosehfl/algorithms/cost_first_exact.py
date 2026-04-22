"""
Exact Cost First planner using mixed-integer linear programming.

This implements the paper's cost-only baseline objective:
    min  kappa_c * sum_{n,e} y_ne * c_ne + sum_e x_e * c_ec
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


try:
    import pulp
except ImportError:  # pragma: no cover - explicit runtime error handled below
    pulp = None


@dataclass
class CostFirstResult:
    selected_edges: Set[int]
    associations: Dict[int, int]
    edge_nodes: Dict[int, Set[int]]
    edge_data_sizes: Dict[int, int]
    objective_value: float


def run_cost_first_exact(
    candidate_edges: List[int],
    all_nodes: List[int],
    communication_costs_ne: Dict[Tuple[int, int], float],
    communication_costs_ec: Dict[int, float],
    data_sizes: Dict[int, int],
    kappa_c: int = 10,
    B_e: int = 10,
) -> CostFirstResult:
    if pulp is None:
        raise RuntimeError(
            "PuLP is required for the exact Cost First baseline. "
            "Install project dependencies with `uv sync`."
        )

    problem = pulp.LpProblem("cost_first_exact", pulp.LpMinimize)
    x = {
        edge_id: pulp.LpVariable(f"x_{edge_id}", cat="Binary")
        for edge_id in candidate_edges
    }
    y = {
        (node_id, edge_id): pulp.LpVariable(f"y_{node_id}_{edge_id}", cat="Binary")
        for node_id in all_nodes
        for edge_id in candidate_edges
    }

    problem += (
        pulp.lpSum(
            kappa_c * communication_costs_ne[(node_id, edge_id)] * y[(node_id, edge_id)]
            for node_id in all_nodes
            for edge_id in candidate_edges
        )
        + pulp.lpSum(communication_costs_ec[edge_id] * x[edge_id] for edge_id in candidate_edges)
    )

    for node_id in all_nodes:
        problem += pulp.lpSum(y[(node_id, edge_id)] for edge_id in candidate_edges) == 1

    for node_id in all_nodes:
        for edge_id in candidate_edges:
            problem += y[(node_id, edge_id)] <= x[edge_id]

    for edge_id in candidate_edges:
        problem += pulp.lpSum(y[(node_id, edge_id)] for node_id in all_nodes) <= B_e

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = problem.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Cost First MILP did not solve optimally: {pulp.LpStatus[status]}")

    selected_edges = {edge_id for edge_id in candidate_edges if pulp.value(x[edge_id]) > 0.5}
    associations: Dict[int, int] = {}
    edge_nodes: Dict[int, Set[int]] = {edge_id: set() for edge_id in selected_edges}
    edge_data_sizes: Dict[int, int] = {edge_id: 0 for edge_id in selected_edges}

    for node_id in all_nodes:
        assigned_edge = None
        for edge_id in candidate_edges:
            if pulp.value(y[(node_id, edge_id)]) > 0.5:
                assigned_edge = edge_id
                break
        if assigned_edge is None:
            raise RuntimeError(f"Node {node_id} was not assigned by the MILP solution.")
        associations[node_id] = assigned_edge
        edge_nodes[assigned_edge].add(node_id)
        edge_data_sizes[assigned_edge] += data_sizes[node_id]

    return CostFirstResult(
        selected_edges=selected_edges,
        associations=associations,
        edge_nodes=edge_nodes,
        edge_data_sizes=edge_data_sizes,
        objective_value=float(pulp.value(problem.objective)),
    )
