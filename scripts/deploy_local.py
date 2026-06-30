#!/usr/bin/env python3
"""
RoSEHFL Single-Machine Deployment Orchestrator (Pipeline A)
=============================================================
Launches a server and N client processes on one machine.
Provides process supervision, GPU sharing, and log collection.

Usage:
    python -m scripts.deploy_local \
      --strategy rose_q1s --model mobilenetv2 --dataset cifar10 \
      --num-nodes 30 --topology geant2010 \
      --cost-mode analytical --client-gpu-sharing round-robin
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List

from scripts._cli_args import add_common_experiment_args, add_single_strategy_arg
from scripts._deploy_common import ensure_partitions, setup_logging, write_deploy_config
from scripts._experiment_bundle import ensure_dataset_defaults
from scripts._rose_common import timestamped_dir


def build_client_cmd(node_id: int, server_addr: str, args, project_root: str) -> List[str]:
    cmd = [
        sys.executable, "-m", "scripts.deploy_client",
        "--node-id", str(node_id),
        "--server-address", server_addr,
        "--model", args.model,
        "--dataset", args.dataset,
        "--num-nodes", str(args.num_nodes),
        "--batch-size", str(args.batch_size),
        "--shard-size", str(args.shard_size),
        "--shards-per-node", str(args.shards_per_node),
        "--classes-per-node", str(args.classes_per_node),
        "--probe-size", str(args.probe_size),
        "--seed", str(args.seed),
    ]
    if args.augment:
        cmd.append("--augment")
    return cmd


def build_server_cmd(args, output_dir: str) -> List[str]:
    cmd = [
        sys.executable, "-m", "scripts.deploy_server",
        "--strategy", args.strategy,
        "--model", args.model,
        "--dataset", args.dataset,
        "--num-nodes", str(args.num_nodes),
        "--topology", args.topology,
        "--address", args.address,
        "--cost-mode", args.cost_mode,
        "--output-dir", output_dir,
        "--seed", str(args.seed),
        "--shard-size", str(args.shard_size),
        "--shards-per-node", str(args.shards_per_node),
        "--classes-per-node", str(args.classes_per_node),
        "--probe-size", str(args.probe_size),
        "--lr", str(args.lr),
        "--momentum", str(args.momentum),
        "--kappa-e", str(args.kappa_e),
        "--kappa-c", str(args.kappa_c),
        "--kappa", str(args.kappa),
        "--gamma-max", str(args.gamma_max),
        "--B-e", str(args.B_e),
        "--T-max", str(args.T_max),
    ]
    if args.resume:
        cmd.append("--resume")
    if args.cost_mode == "delayed":
        cmd.extend(["--lan-bandwidth-mbps", str(args.lan_bandwidth_mbps),
                    "--delay-scale", str(args.delay_scale)])
    return cmd


def torch_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoSEHFL single-machine deployment orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_experiment_args(parser)
    add_single_strategy_arg(parser)
    parser.add_argument("--address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--cost-mode", type=str, default="analytical", choices=["analytical", "delayed"])
    parser.add_argument("--lan-bandwidth-mbps", type=float, default=100.0)
    parser.add_argument("--delay-scale", type=float, default=1.0)
    parser.add_argument("--client-gpu-sharing", type=str, default="round-robin",
                        choices=["round-robin", "sequential", "cpu"])
    parser.add_argument("--max-client-retries", type=int, default=3)

    args = parser.parse_args()
    if args.no_augment:
        args.augment = False

    ensure_dataset_defaults(args)

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "results",
            timestamped_dir(f"deploy_local_{args.strategy}_{args.model}_{args.dataset}"),
        )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    logger = setup_logging(args.output_dir, "orchestrator")
    logger.info(f"Starting single-machine deployment: {args.strategy}")
    logger.info(f"Nodes: {args.num_nodes}, Model: {args.model}, Dataset: {args.dataset}")
    logger.info(f"GPU sharing: {args.client_gpu_sharing}")
    logger.info(f"Output: {args.output_dir}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    partitions_dir = os.path.join(project_root, "partitions")
    ensure_partitions(
        partitions_dir, args.dataset, args.num_nodes,
        args.shard_size, args.shards_per_node, args.classes_per_node,
        args.probe_size, args.seed,
    )
    logger.info("Partitions ready")

    host_assignments = {
        "localhost": list(range(args.num_nodes)),
    }
    write_deploy_config(
        args.output_dir,
        strategy=args.strategy,
        model=args.model,
        dataset=args.dataset,
        num_nodes=args.num_nodes,
        topology=args.topology,
        cost_mode=args.cost_mode,
        lan_bandwidth_mbps=args.lan_bandwidth_mbps if args.cost_mode == "delayed" else None,
        delay_scale=args.delay_scale if args.cost_mode == "delayed" else None,
        host_assignments=host_assignments,
    )

    _addr_parts = args.address.rsplit(":", 1)
    _port = _addr_parts[1] if len(_addr_parts) > 1 else "8080"
    server_addr = f"localhost:{_port}" if ":" in args.address else "localhost:8080"

    # Start server
    server_cmd = build_server_cmd(args, args.output_dir)
    logger.info(f"Starting server: {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(
        server_cmd, cwd=project_root,
        stdout=open(os.path.join(args.output_dir, "logs", "server_stdout.log"), "w"),
        stderr=subprocess.STDOUT,
    )

    # Wait for server to start
    time.sleep(3)

    # Start clients
    client_procs: Dict[int, subprocess.Popen] = {}
    client_retries: Dict[int, int] = {}

    for node_id in range(args.num_nodes):
        env = os.environ.copy()
        if args.client_gpu_sharing == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif args.client_gpu_sharing == "round-robin" and torch_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                env["CUDA_VISIBLE_DEVICES"] = str(node_id % num_gpus)

        cmd = build_client_cmd(node_id, server_addr, args, project_root)
        logger.info(f"Starting client {node_id}")
        proc = subprocess.Popen(
            cmd, cwd=project_root, env=env,
            stdout=open(os.path.join(args.output_dir, "logs", f"client_{node_id}_stdout.log"), "w"),
            stderr=subprocess.STDOUT,
        )
        client_procs[node_id] = proc
        client_retries[node_id] = 0

    # Supervise
    try:
        while server_proc.poll() is None:
            for node_id, proc in list(client_procs.items()):
                if proc.poll() is not None:
                    if proc.returncode != 0 and client_retries[node_id] < args.max_client_retries:
                        client_retries[node_id] += 1
                        logger.warning(
                            f"Client {node_id} exited with code {proc.returncode}, "
                            f"restarting (attempt {client_retries[node_id]}/{args.max_client_retries})"
                        )
                        time.sleep(5)
                        env = os.environ.copy()
                        if args.client_gpu_sharing == "cpu":
                            env["CUDA_VISIBLE_DEVICES"] = ""
                        elif args.client_gpu_sharing == "round-robin" and torch_available():
                            num_gpus = torch.cuda.device_count()
                            if num_gpus > 0:
                                env["CUDA_VISIBLE_DEVICES"] = str(node_id % num_gpus)
                        cmd = build_client_cmd(node_id, server_addr, args, project_root)
                        client_procs[node_id] = subprocess.Popen(
                            cmd, cwd=project_root, env=env,
                            stdout=open(os.path.join(args.output_dir, "logs", f"client_{node_id}_stdout.log"), "a"),
                            stderr=subprocess.STDOUT,
                        )
                    elif proc.returncode == 0:
                        logger.info(f"Client {node_id} finished successfully")
                        del client_procs[node_id]
                    else:
                        logger.error(f"Client {node_id} exhausted retries, removing")
                        del client_procs[node_id]
            time.sleep(5)

        # Server done
        if server_proc.returncode == 0:
            logger.info("Server finished successfully")
        else:
            logger.error(f"Server exited with code {server_proc.returncode}")

    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down...")
        server_proc.terminate()
        for proc in client_procs.values():
            proc.terminate()
        server_proc.wait()
        for proc in client_procs.values():
            proc.wait()

    # Kill any remaining clients
    for proc in client_procs.values():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    logger.info("Deployment complete")


if __name__ == "__main__":
    main()
