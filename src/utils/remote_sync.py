"""Opt-in live sync of a run directory to a remote (e.g. Google Drive via rclone).

Disabled by default and everywhere: local runs, lab deployment runs, and the
Colab notebook (which already writes straight to a mounted Drive path) never
set the environment variables below, so :func:`request_sync` is a single
cheap no-op check and nothing else in the codebase behaves differently.

Enable it (e.g. from a Kaggle notebook, before launching the run) with:

    os.environ["ROSEHFL_SYNC_LOCAL"] = str(LOCAL_RUN_DIR)
    os.environ["ROSEHFL_SYNC_REMOTE"] = REMOTE_RUN_DIR   # e.g. "gdrive:RoSEHFL/kaggle_runs/<run>"

Once enabled, every call to :func:`request_sync` pushes the whole local run
directory to the remote in a background thread, throttled so a burst of
fast rounds cannot pile up overlapping ``rclone`` invocations. ``rclone
copy`` only transfers files that changed, so re-syncing the whole (growing)
run directory every round is cheap in practice.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time

_LOCAL = os.environ.get("ROSEHFL_SYNC_LOCAL")
_REMOTE = os.environ.get("ROSEHFL_SYNC_REMOTE")
_MIN_INTERVAL_SECONDS = float(os.environ.get("ROSEHFL_SYNC_MIN_INTERVAL", "20"))
_ENABLED = bool(_LOCAL and _REMOTE)

_lock = threading.Lock()
_last_sync_at = 0.0
_sync_in_flight = False


def request_sync() -> None:
    """Ask for a sync soon. No-op unless ROSEHFL_SYNC_LOCAL/REMOTE are set.

    Safe to call every round: skips silently if a sync is already running
    or the minimum interval hasn't elapsed, so callers never need to
    throttle it themselves.
    """
    if not _ENABLED:
        return
    global _sync_in_flight
    with _lock:
        if _sync_in_flight or (time.time() - _last_sync_at) < _MIN_INTERVAL_SECONDS:
            return
        _sync_in_flight = True
    threading.Thread(target=_run_sync, daemon=True).start()


def _run_sync() -> None:
    global _last_sync_at, _sync_in_flight
    try:
        subprocess.run(
            ["rclone", "copy", _LOCAL, _REMOTE, "--transfers", "4", "--checkers", "4"],
            check=False,
            capture_output=True,
            timeout=300,
        )
    except Exception:
        pass
    finally:
        with _lock:
            _last_sync_at = time.time()
            _sync_in_flight = False
