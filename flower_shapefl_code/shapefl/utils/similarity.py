"""
Similarity Computation Utilities for ShapeFL
=============================================
S_ij = 1 − cos(Δw_i^(L), Δw_j^(L))
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """Cosine similarity between two vectors, clipped to [-1, 1].

    Returns 0.0 for zero-length vectors (maximum dissimilarity in S_ij
    after the ``1 - cos`` transform).
    """
    vec1 = vec1.float().flatten()
    vec2 = vec2.float().flatten()
    dot_product = torch.dot(vec1, vec2)
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    cos_sim = (dot_product / (norm1 * norm2)).item()
    # Clip to handle floating-point rounding past ±1
    return float(np.clip(cos_sim, -1.0, 1.0))


def compute_data_distribution_diversity(
    vec1: torch.Tensor, vec2: torch.Tensor
) -> float:
    """S_ij = 1 − cos(Δw_i, Δw_j).  Higher → more diverse."""
    return 1.0 - compute_cosine_similarity(vec1, vec2)


def compute_similarity_matrix(linear_updates: Dict[int, torch.Tensor]) -> np.ndarray:
    """Full N×N diversity matrix S."""
    node_ids = sorted(linear_updates.keys())
    n = len(node_ids)
    S = np.zeros((n, n))
    for i, ni in enumerate(node_ids):
        for j, nj in enumerate(node_ids):
            if i < j:
                d = compute_data_distribution_diversity(
                    linear_updates[ni], linear_updates[nj]
                )
                S[i, j] = d
                S[j, i] = d
    return S


def compute_similarity_from_updates(
    updates: List[Tuple[int, torch.Tensor]], num_nodes: int
) -> np.ndarray:
    """Build S from a list of ``(node_id, update)`` tuples."""
    update_dict = {nid: u for nid, u in updates}
    S = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i in update_dict and j in update_dict:
                d = compute_data_distribution_diversity(update_dict[i], update_dict[j])
                S[i, j] = d
                S[j, i] = d
    return S
