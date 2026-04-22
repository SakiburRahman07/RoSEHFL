"""
Network topology utilities for ShapeFL.

The shapefl paper evaluates ShapeFL on Topology Zoo graphs:
    - Geant2010 (37 nodes)
    - Uunet (42 nodes)
    - Tinet (46 nodes)
    - Viatel (91 nodes)

Communication cost formulas (shapefl paper Section V-A):
    c_ne = 0.002 * d_ne * S_m
    c_ec = 0.02  * d_ec * S_m

where d is the shortest-path distance in km and S_m is model size in GB.
"""

from __future__ import annotations

import heapq
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


_GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
_INVALID_LABELS = {"", "none", "null"}
_CLOUD_LAT, _CLOUD_LON = 47.6062, -122.3321

_TOPOLOGY_FILES = {
    "geant2010": "Geant2010.graphml",
    "uunet": "Uunet.graphml",
    "tinet": "Tinet.graphml",
    "viatel": "VtlWavenet2011.graphml",
}


@dataclass
class TopologyNode:
    label: str
    latitude: float
    longitude: float


@dataclass
class ParsedTopology:
    name: str
    nodes: List[TopologyNode]
    edges: List[Tuple[int, int, float]]


@dataclass
class TopologyInfo:
    """Complete topology together with ShapeFL communication-cost tables."""

    name: str
    num_nodes: int
    num_candidate_edges: int
    node_labels: List[str]
    node_edge_costs: Dict[Tuple[int, int], float]
    edge_cloud_costs: Dict[int, float]
    model_size_gb: float


def _topology_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "topologies",
    )


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Earth in km."""
    earth_radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def _dijkstra_all_pairs(
    num_nodes: int,
    edges: List[Tuple[int, int, float]],
) -> Dict[Tuple[int, int], float]:
    """All-pairs shortest paths on an undirected weighted graph."""
    adjacency: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(num_nodes)}
    for u, v, weight in edges:
        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))

    shortest_paths: Dict[Tuple[int, int], float] = {}
    for source in range(num_nodes):
        distances = [float("inf")] * num_nodes
        distances[source] = 0.0
        priority_queue = [(0.0, source)]

        while priority_queue:
            dist_u, u = heapq.heappop(priority_queue)
            if dist_u > distances[u]:
                continue
            for v, weight in adjacency[u]:
                candidate = dist_u + weight
                if candidate < distances[v]:
                    distances[v] = candidate
                    heapq.heappush(priority_queue, (candidate, v))

        for target, distance in enumerate(distances):
            if distance < float("inf"):
                shortest_paths[(source, target)] = distance

    return shortest_paths


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _node_is_usable(node_data: Dict[str, str]) -> bool:
    label = (node_data.get("label") or "").strip()
    latitude = _safe_float(node_data.get("Latitude"))
    longitude = _safe_float(node_data.get("Longitude"))
    return (
        label.lower() not in _INVALID_LABELS
        and latitude is not None
        and longitude is not None
    )


def _parse_graphml_topology(topology: str) -> ParsedTopology:
    filename = _TOPOLOGY_FILES[topology]
    path = os.path.join(_topology_dir(), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Topology file not found for '{topology}': {path}"
        )

    root = ET.parse(path).getroot()
    key_map = {
        key.attrib["id"]: (key.attrib.get("for"), key.attrib.get("attr.name"))
        for key in root.findall("g:key", _GRAPHML_NS)
    }
    graph = root.find("g:graph", _GRAPHML_NS)
    if graph is None:
        raise ValueError(f"GraphML file does not contain a graph: {path}")

    kept_nodes: List[TopologyNode] = []
    original_to_kept: Dict[str, int] = {}

    for node in graph.findall("g:node", _GRAPHML_NS):
        node_values = {
            key_map[data.attrib["key"]][1]: (data.text or "").strip()
            for data in node.findall("g:data", _GRAPHML_NS)
            if data.attrib["key"] in key_map
            and key_map[data.attrib["key"]][0] == "node"
        }
        if not _node_is_usable(node_values):
            continue

        original_to_kept[node.attrib["id"]] = len(kept_nodes)
        kept_nodes.append(
            TopologyNode(
                label=node_values["label"].strip(),
                latitude=float(node_values["Latitude"]),
                longitude=float(node_values["Longitude"]),
            )
        )

    weighted_edges: List[Tuple[int, int, float]] = []
    seen_edges = set()
    for edge in graph.findall("g:edge", _GRAPHML_NS):
        source = edge.attrib.get("source")
        target = edge.attrib.get("target")
        if source not in original_to_kept or target not in original_to_kept:
            continue

        u = original_to_kept[source]
        v = original_to_kept[target]
        if u == v:
            continue

        edge_key = (min(u, v), max(u, v))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        node_u = kept_nodes[u]
        node_v = kept_nodes[v]
        weight = _haversine_km(
            node_u.latitude,
            node_u.longitude,
            node_v.latitude,
            node_v.longitude,
        )
        weighted_edges.append((u, v, weight))

    if not weighted_edges:
        return ParsedTopology(name=topology, nodes=kept_nodes, edges=weighted_edges)

    node_degrees = [0] * len(kept_nodes)
    for u, v, _ in weighted_edges:
        node_degrees[u] += 1
        node_degrees[v] += 1

    retained_indices = [idx for idx, degree in enumerate(node_degrees) if degree > 0]
    if len(retained_indices) == len(kept_nodes):
        return ParsedTopology(name=topology, nodes=kept_nodes, edges=weighted_edges)

    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(retained_indices)}
    compact_nodes = [kept_nodes[idx] for idx in retained_indices]
    compact_edges = [
        (old_to_new[u], old_to_new[v], weight)
        for u, v, weight in weighted_edges
        if u in old_to_new and v in old_to_new
    ]
    return ParsedTopology(name=topology, nodes=compact_nodes, edges=compact_edges)


def generate_graphml_topology(
    topology: str,
    num_clients: int,
    num_edges: int,
    model_size_bytes: int = 246824,
    seed: int = 42,
) -> TopologyInfo:
    """Generate communication costs from a Topology Zoo GraphML file."""
    parsed = _parse_graphml_topology(topology)
    available_nodes = len(parsed.nodes)
    if num_clients > available_nodes:
        raise ValueError(
            f"Topology '{topology}' has {available_nodes} usable nodes; "
            f"cannot sample {num_clients} clients."
        )

    shortest_paths = _dijkstra_all_pairs(available_nodes, parsed.edges)
    model_size_gb = model_size_bytes / (1024 ** 3)
    rng = np.random.default_rng(seed)

    selected_clients = sorted(
        rng.choice(available_nodes, size=num_clients, replace=False).tolist()
    )
    chosen_nodes = [parsed.nodes[idx] for idx in selected_clients]

    num_edges = min(num_edges, num_clients)
    selected_edge_locals = sorted(
        rng.choice(num_clients, size=num_edges, replace=False).tolist()
    )
    local_edge_to_slot = {
        local_client_idx: edge_slot
        for edge_slot, local_client_idx in enumerate(selected_edge_locals)
    }

    node_edge_costs: Dict[Tuple[int, int], float] = {}
    for client_local, topo_client_idx in enumerate(selected_clients):
        for edge_local, edge_slot in local_edge_to_slot.items():
            topo_edge_idx = selected_clients[edge_local]
            distance = shortest_paths.get((topo_client_idx, topo_edge_idx))
            if distance is None:
                node = parsed.nodes[topo_client_idx]
                edge = parsed.nodes[topo_edge_idx]
                distance = _haversine_km(
                    node.latitude,
                    node.longitude,
                    edge.latitude,
                    edge.longitude,
                )
            node_edge_costs[(client_local, edge_slot)] = 0.002 * distance * model_size_gb

    edge_cloud_costs: Dict[int, float] = {}
    for edge_local, edge_slot in local_edge_to_slot.items():
        edge = chosen_nodes[edge_local]
        distance = _haversine_km(
            edge.latitude,
            edge.longitude,
            _CLOUD_LAT,
            _CLOUD_LON,
        )
        edge_cloud_costs[edge_slot] = 0.02 * distance * model_size_gb

    return TopologyInfo(
        name=topology,
        num_nodes=num_clients,
        num_candidate_edges=num_edges,
        node_labels=[node.label for node in chosen_nodes],
        node_edge_costs=node_edge_costs,
        edge_cloud_costs=edge_cloud_costs,
        model_size_gb=model_size_gb,
    )


def generate_random_topology(
    num_clients: int,
    num_edges: int,
    model_size_bytes: int = 246824,
    seed: int = 42,
) -> TopologyInfo:
    """Fallback synthetic topology for non-paper experiments."""
    rng = np.random.default_rng(seed)
    model_size_gb = model_size_bytes / (1024 ** 3)

    positions = rng.random((num_clients, 2)) * 1000.0
    cloud_position = np.array([500.0, 3500.0])

    num_edges = min(num_edges, num_clients)
    selected_edge_locals = sorted(
        rng.choice(num_clients, size=num_edges, replace=False).tolist()
    )

    node_edge_costs: Dict[Tuple[int, int], float] = {}
    for node_idx in range(num_clients):
        for edge_slot, edge_local in enumerate(selected_edge_locals):
            if node_idx == edge_local:
                node_edge_costs[(node_idx, edge_slot)] = 0.0
                continue
            distance = float(np.linalg.norm(positions[node_idx] - positions[edge_local]))
            node_edge_costs[(node_idx, edge_slot)] = 0.002 * distance * model_size_gb

    edge_cloud_costs: Dict[int, float] = {}
    for edge_slot, edge_local in enumerate(selected_edge_locals):
        distance = float(np.linalg.norm(positions[edge_local] - cloud_position))
        edge_cloud_costs[edge_slot] = 0.02 * distance * model_size_gb

    return TopologyInfo(
        name="random",
        num_nodes=num_clients,
        num_candidate_edges=num_edges,
        node_labels=[f"node_{i}" for i in range(num_clients)],
        node_edge_costs=node_edge_costs,
        edge_cloud_costs=edge_cloud_costs,
        model_size_gb=model_size_gb,
    )


def generate_topology(
    topology: str = "geant2010",
    num_clients: int = 30,
    num_edges: int = 10,
    model_size_bytes: int = 246824,
    seed: int = 42,
) -> TopologyInfo:
    """
    Create a topology by name.

    Supported paper topologies:
        - geant2010
        - uunet
        - tinet
        - viatel

    Legacy synthetic fallback:
        - random
    """
    normalized = topology.lower()
    if normalized == "random":
        return generate_random_topology(num_clients, num_edges, model_size_bytes, seed)
    if normalized not in _TOPOLOGY_FILES:
        raise ValueError(
            f"Unknown topology '{topology}'. Choose from: "
            f"{', '.join(sorted(_TOPOLOGY_FILES))}, random"
        )
    return generate_graphml_topology(
        normalized,
        num_clients,
        num_edges,
        model_size_bytes,
        seed,
    )
