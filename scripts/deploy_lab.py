#!/usr/bin/env python3
"""
RoSEHFL Multi-Machine Lab Deployment Orchestrator (Pipeline B)
================================================================
Launches a server locally and N clients on remote machines (e.g., Raspberry Pi 5)
via SSH. Provides remote process supervision, health monitoring, and log collection.

Usage:
    python -m scripts.deploy_lab \
      --strategy rose_q1s --model mobilenetv2 --dataset cifar10 \
      --num-nodes 30 --topology geant2010 \
      --host-config lab_hosts.yaml --cost-mode delayed --resume
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from typing import Dict, List

try:
    import yaml
except ImportError:
    yaml = None

from scripts._cli_args import add_common_experiment_args, add_single_strategy_arg
from scripts._deploy_common import ensure_partitions, get_local_ip, setup_logging, write_deploy_config
from scripts._experiment_bundle import ensure_dataset_defaults
from scripts._rose_common import timestamped_dir


def load_host_config(path: str) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required for lab deployment. Install with: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def build_ssh_cmd(host: str, node_id: int, server_ip: str, port: int, args, remote_dir: str, ssh_key: str = "") -> List[str]:
    remote_cmd = (
        f"cd {remote_dir} && "
        f".venv/bin/python -m scripts.deploy_client "
        f"--node-id {node_id} "
        f"--server-address {shlex.quote(f'{server_ip}:{port}')} "
        f"--model {shlex.quote(args.model)} --dataset {shlex.quote(args.dataset)} "
        f"--num-nodes {args.num_nodes} "
        f"--batch-size {args.batch_size} "
        f"--shard-size {args.shard_size} "
        f"--shards-per-node {args.shards_per_node} "
        f"--classes-per-node {args.classes_per_node} "
        f"--probe-size {args.probe_size} "
        f"--seed {args.seed}"
    )
    if args.augment:
        remote_cmd += " --augment"
    ssh_args = ["ssh", "-o", "StrictHostKeyChecking=no"]
    if ssh_key:
        ssh_args.extend(["-i", ssh_key])
    ssh_args.extend([host, remote_cmd])
    return ssh_args


def build_server_cmd(args, output_dir: str) -> List[str]:
    cmd = [
        sys.executable, "-m", "scripts.deploy_server",
        "--strategy", args.strategy,
        "--model", args.model,
        "--dataset", args.dataset,
        "--num-nodes", str(args.num_nodes),
        "--topology", args.topology,
        "--address", f"0.0.0.0:{args.port}",
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
    cmd.extend(["--round-timeout", str(args.round_timeout)])
    return cmd


def sync_partitions_to_host(host: str, partitions_dir: str, remote_dir: str, logger, ssh_key: str = "") -> None:
    logger.info(f"Syncing partitions to {host}")
    remote_partitions = f"{host}:{remote_dir}/partitions/"
    rsync_cmd = ["rsync", "-avz", "--ignore-errors"]
    if ssh_key:
        rsync_cmd.extend(["-e", f"ssh -i {ssh_key} -o StrictHostKeyChecking=no"])
    rsync_cmd.extend([f"{partitions_dir}/", remote_partitions])
    result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        logger.warning(f"rsync to {host} returned {result.returncode}: {result.stderr}")
    else:
        logger.info(f"Partitions synced to {host}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoSEHFL multi-machine lab deployment orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_experiment_args(parser)
    add_single_strategy_arg(parser)
    parser.add_argument("--host-config", type=str, required=True,
                        help="YAML file with host addresses and node assignments.")
    parser.add_argument("--remote-dir", type=str, default="~/RoSEHFL",
                        help="Project directory on remote hosts.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--cost-mode", type=str, default="analytical", choices=["analytical", "delayed"])
    parser.add_argument("--lan-bandwidth-mbps", type=float, default=100.0)
    parser.add_argument("--delay-scale", type=float, default=1.0)
    parser.add_argument("--max-client-retries", type=int, default=3)
    parser.add_argument("--round-timeout", type=float, default=300.0,
                        help="Seconds before a training round times out if clients don't respond.")
    parser.add_argument("--skip-sync", action="store_true",
                        help="Skip partition sync (use if partitions already exist on remote hosts).")
    parser.add_argument("--ssh-key", type=str, default="",
                        help="Path to SSH private key for connecting to remote hosts (e.g., ~/.ssh/ssh_rosehfl).")

    args = parser.parse_args()
    if args.no_augment:
        args.augment = False

    ensure_dataset_defaults(args)

    if args.output_dir is None:
        args.output_dir = os.path.join(
            "results",
            timestamped_dir(f"deploy_lab_{args.strategy}_{args.model}_{args.dataset}"),
        )
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    logger = setup_logging(args.output_dir, "orchestrator")
    logger.info(f"Starting lab deployment: {args.strategy}")
    logger.info(f"Host config: {args.host_config}")

    host_config = load_host_config(args.host_config)
    hosts = host_config.get("hosts", [])

    all_node_ids: List[int] = []
    for host_entry in hosts:
        all_node_ids.extend(host_entry["node_ids"])
    if len(all_node_ids) != args.num_nodes:
        logger.error(f"Host config has {len(all_node_ids)} node_ids but --num-nodes={args.num_nodes}")
        sys.exit(1)

    server_ip = host_config.get("server", {}).get("local_ip", get_local_ip())
    logger.info(f"Server IP for clients: {server_ip}")

    host_assignments = {h["address"]: h["node_ids"] for h in hosts}
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

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    partitions_dir = os.path.join(project_root, "partitions")
    ensure_partitions(
        partitions_dir, args.dataset, args.num_nodes,
        args.shard_size, args.shards_per_node, args.classes_per_node,
        args.probe_size, args.seed,
    )
    logger.info("Partitions ready")

    if not args.skip_sync:
        for host_entry in hosts:
            host = host_entry["address"]
            sync_partitions_to_host(host, partitions_dir, args.remote_dir, logger, args.ssh_key)

    # Start server
    server_cmd = build_server_cmd(args, args.output_dir)
    logger.info(f"Starting server: {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(
        server_cmd, cwd=project_root,
        stdout=open(os.path.join(args.output_dir, "logs", "server_stdout.log"), "w"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(5)

    # Clean up any leftover client processes from previous runs
    logger.info("Cleaning up old processes on all Pis...")
    for host_entry in hosts:
        host = host_entry["address"]
        kill_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5"]
        if args.ssh_key:
            kill_cmd.extend(["-i", args.ssh_key])
        kill_cmd.extend([host, "pkill -f deploy_client 2>/dev/null; echo cleaned"])
        try:
            subprocess.run(kill_cmd, capture_output=True, timeout=10)
        except Exception:
            pass
    logger.info("Cleanup complete")

    # Launch remote clients
    client_procs: Dict[int, subprocess.Popen] = {}
    client_retries: Dict[int, int] = {}
    client_hosts: Dict[int, str] = {}

    for host_entry in hosts:
        host = host_entry["address"]
        for node_id in host_entry["node_ids"]:
            ssh_cmd = build_ssh_cmd(host, node_id, server_ip, args.port, args, args.remote_dir, args.ssh_key)
            logger.info(f"Launching client {node_id} on {host}")
            proc = subprocess.Popen(
                ssh_cmd,
                stdout=open(os.path.join(args.output_dir, "logs", f"client_{node_id}_stdout.log"), "w"),
                stderr=subprocess.STDOUT,
            )
            client_procs[node_id] = proc
            client_retries[node_id] = 0
            client_hosts[node_id] = host
            time.sleep(1)

    # Supervise
    try:
        while server_proc.poll() is None:
            for node_id, proc in list(client_procs.items()):
                if proc.poll() is not None:
                    if proc.returncode != 0 and client_retries[node_id] < args.max_client_retries:
                        client_retries[node_id] += 1
                        host = client_hosts[node_id]
                        logger.warning(
                            f"Client {node_id} on {host} exited (code {proc.returncode}), "
                            f"restarting (attempt {client_retries[node_id]}/{args.max_client_retries})"
                        )
                        time.sleep(10 * client_retries[node_id])
                        ssh_cmd = build_ssh_cmd(host, node_id, server_ip, args.port, args, args.remote_dir, args.ssh_key)
                        client_procs[node_id] = subprocess.Popen(
                            ssh_cmd,
                            stdout=open(os.path.join(args.output_dir, "logs", f"client_{node_id}_stdout.log"), "a"),
                            stderr=subprocess.STDOUT,
                        )
                    elif proc.returncode == 0:
                        logger.info(f"Client {node_id} finished")
                        del client_procs[node_id]
                    else:
                        logger.error(f"Client {node_id} exhausted retries")
                        del client_procs[node_id]
            time.sleep(5)

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
        # Kill remote clients
        for host_entry in hosts:
            host = host_entry["address"]
            kill_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5"]
            if args.ssh_key:
                kill_cmd.extend(["-i", args.ssh_key])
            kill_cmd.extend([host, "pkill -f deploy_client 2>/dev/null"])
            try:
                subprocess.run(kill_cmd, capture_output=True, timeout=10)
            except Exception:
                pass
        logger.info("Shutdown complete")

    # Kill remaining clients
    for node_id, proc in client_procs.items():
        if proc.poll() is None:
            host = client_hosts.get(node_id, "")
            logger.info(f"Killing client {node_id} on {host}")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    # Collect remote logs
    key_flag = ["-i", args.ssh_key] if args.ssh_key else []
    for host_entry in hosts:
        host = host_entry["address"]
        for node_id in host_entry["node_ids"]:
            try:
                subprocess.run(
                    ["scp"] + key_flag + ["-o", "StrictHostKeyChecking=no",
                     f"{host}:~/rosehfl_logs/client_{node_id}.log",
                     os.path.join(args.output_dir, "logs", f"client_{node_id}.log")],
                    timeout=30, capture_output=True,
                )
            except Exception as e:
                logger.warning(f"Could not fetch log for client {node_id}: {e}")

    logger.info("Lab deployment complete")


if __name__ == "__main__":
    main()
