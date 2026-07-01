"""Shared helpers for deployment scripts."""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
from datetime import datetime, timezone
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def setup_logging(output_dir: str, component: str, node_id: int | None = None) -> logging.Logger:
    """Set up file + console logging for a deployment process."""
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    name = f"{component}" if node_id is None else f"{component}_{node_id}"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_path = os.path.join(logs_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(name)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(console_handler)

    return logger


def write_deploy_config(
    output_dir: str,
    *,
    strategy: str,
    model: str,
    dataset: str,
    num_nodes: int,
    topology: str,
    cost_mode: str,
    lan_bandwidth_mbps: float | None,
    delay_scale: float | None,
    host_assignments: dict | None = None,
) -> None:
    """Write deployment configuration to output_dir/deploy_config.json."""
    config = {
        "strategy": strategy,
        "model": model,
        "dataset": dataset,
        "num_nodes": num_nodes,
        "topology": topology,
        "cost_mode": cost_mode,
        "lan_bandwidth_mbps": lan_bandwidth_mbps,
        "delay_scale": delay_scale,
        "host_assignments": host_assignments,
        "machine": socket.gethostname(),
        "created_at": utc_timestamp(),
    }
    path = os.path.join(output_dir, "deploy_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def get_local_ip() -> str:
    """Get the local IP address for LAN communication."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def ensure_partitions(partitions_dir: str, dataset: str, num_nodes: int,
                      shard_size: int, shards_per_node: int,
                      classes_per_node: int, probe_size: int, seed: int) -> None:
    """Generate partition files if they don't exist."""
    metadata_path = os.path.join(partitions_dir, "metadata.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        if (meta.get("dataset") == dataset and meta.get("num_nodes") == num_nodes
                and meta.get("seed") == seed and meta.get("shard_size") == shard_size
                and meta.get("shards_per_node") == shards_per_node
                and meta.get("classes_per_node") == classes_per_node):
            return

    from .generate_partitions import generate_partition_files
    generate_partition_files(
        dataset_name=dataset,
        num_nodes=num_nodes,
        shard_size=shard_size,
        shards_per_node=shards_per_node,
        classes_per_node=classes_per_node,
        probe_size=probe_size,
        seed=seed,
        output_dir=partitions_dir,
    )
