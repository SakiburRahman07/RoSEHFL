"""
JSON Utilities
==============
Handles numpy and torch types that standard ``json.dump`` cannot serialise.
Adopted from ShapeFL-Flower's NumpyEncoder.
"""

import json
import numpy as np

from .atomic_io import atomic_write_bytes

__all__ = ["NumpyEncoder", "save_json"]


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that transparently converts numpy / torch types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "item"):  # scalar tensors
            return obj.item()
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


def save_json(data, path: str, indent: int = 2) -> str:
    """Write *data* to *path* as pretty-printed JSON using :class:`NumpyEncoder`.

    Writes atomically (temp file + rename) so a process kill mid-write
    cannot leave a truncated, unreadable file behind.
    """
    payload = json.dumps(data, indent=indent, cls=NumpyEncoder).encode("utf-8")
    atomic_write_bytes(path, payload)
    return path
