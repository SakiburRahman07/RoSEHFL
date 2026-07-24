"""Atomic file writes.

A process kill mid-write (Kaggle's 12h hard timeout, a dropped Colab
connection while writing to a Drive-mounted path) can leave a truncated,
corrupt file behind. Writing to a temp file first and swapping it into
place with ``os.replace`` (atomic on POSIX and Windows) means every reader
always sees either the fully-old or the fully-new content, never a
partial write.
"""

from __future__ import annotations

import os


def atomic_write_bytes(path: str, data: bytes) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp-{os.getpid()}"
    with open(tmp_path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
