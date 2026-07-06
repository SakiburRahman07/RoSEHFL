"""CID-to-partition mapping for Flower deployment and simulation."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CidMapper:
    """Maps Flower client CIDs to partition indices 0..N-1.

    In simulation, Flower assigns integer CIDs (0, 1, 2, ...).
    In real deployment (gRPC), Flower assigns random UUID hex CIDs.

    The mapper uses two mechanisms:
    1. Metrics-based: clients report ``node_id`` in fit/evaluate metrics.
       The server registers the mapping in ``aggregate_fit``.
    2. Sort-order fallback: before metrics arrive (round 1), CIDs are
       sorted lexicographically and the index is used as the partition.
    """

    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes
        self.cid_to_node_id: Dict[str, int] = {}
        self._sorted_cids: Optional[List[str]] = None

    def register_from_metrics(self, cid: str, metrics: dict) -> None:
        """Extract node_id from client fit/evaluate metrics and store mapping."""
        node_id = int(metrics.get("node_id", -1))
        if 0 <= node_id < self.num_nodes:
            self.cid_to_node_id[cid] = node_id

    def resolve(self, cid: str) -> int:
        """Map a CID to its partition index.

        Uses metrics-based mapping if available, else falls back to
        sort-order index.
        """
        if cid in self.cid_to_node_id:
            return self.cid_to_node_id[cid]
        node_id = self._sort_order_index(cid)
        logger.warning(
            "CID %s resolved via sort-order to node_id=%d (no metrics yet). ",
            cid[:12], node_id,
        )
        return node_id

    def _sort_order_index(self, cid: str) -> int:
        if self._sorted_cids is None or cid not in self._sorted_cids:
            raise ValueError(
                f"CID {cid!r} not in sort-order map. "
                "Call build_sort_order() first from configure_fit."
            )
        return self._sorted_cids.index(cid)

    def build_sort_order(self, clients) -> None:
        """Pre-populate sort-order mapping from the client manager.

        Called from ``_build_cid_map`` at the start of ``configure_fit``.
        All clients must be registered before the first round.
        """
        self._sorted_cids = sorted(c.cid for c in clients)

    def to_checkpoint(self) -> Dict[str, int]:
        """Serialise the metrics-based mapping for checkpointing."""
        return dict(self.cid_to_node_id)

    def from_checkpoint(self, state: Dict[str, int]) -> None:
        """Restore the metrics-based mapping from a checkpoint."""
        self.cid_to_node_id = dict(state)
