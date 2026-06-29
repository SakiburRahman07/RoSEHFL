"""Tests for partition generation script."""
import json
import os
import tempfile
import pytest


def test_generate_partitions_creates_per_node_files():
    from scripts.generate_partitions import generate_partition_files

    with tempfile.TemporaryDirectory() as tmpdir:
        generate_partition_files(
            dataset_name="fmnist",
            num_nodes=4,
            shard_size=15,
            shards_per_node=12,
            classes_per_node=4,
            probe_size=100,
            seed=42,
            output_dir=tmpdir,
        )

        # Check metadata.json
        with open(os.path.join(tmpdir, "metadata.json")) as f:
            meta = json.load(f)
        assert meta["dataset"] == "fmnist"
        assert meta["num_nodes"] == 4
        assert meta["seed"] == 42

        # Check per-node files
        for node_id in range(4):
            path = os.path.join(tmpdir, f"node_{node_id}_indices.json")
            assert os.path.isfile(path), f"Missing {path}"
            with open(path) as f:
                indices = json.load(f)
            assert isinstance(indices, list)
            assert len(indices) > 0, f"Node {node_id} has empty partition"

        # Check probe_indices.json
        probe_path = os.path.join(tmpdir, "probe_indices.json")
        assert os.path.isfile(probe_path)
        with open(probe_path) as f:
            probe_indices = json.load(f)
        assert isinstance(probe_indices, list)
        assert len(probe_indices) > 0

        # Check partitions.json (full mapping)
        full_path = os.path.join(tmpdir, "partitions.json")
        assert os.path.isfile(full_path)
        with open(full_path) as f:
            partitions = json.load(f)
        assert len(partitions) == 4
        # Verify per-node files match the full mapping
        for node_id in range(4):
            with open(os.path.join(tmpdir, f"node_{node_id}_indices.json")) as f:
                node_indices = json.load(f)
            assert set(node_indices) == set(partitions[str(node_id)])


def test_generate_partitions_no_overlap():
    from scripts.generate_partitions import generate_partition_files

    with tempfile.TemporaryDirectory() as tmpdir:
        generate_partition_files(
            dataset_name="fmnist",
            num_nodes=4,
            shard_size=15,
            shards_per_node=12,
            classes_per_node=4,
            probe_size=100,
            seed=42,
            output_dir=tmpdir,
        )
        all_indices = set()
        for node_id in range(4):
            with open(os.path.join(tmpdir, f"node_{node_id}_indices.json")) as f:
                indices = json.load(f)
            for idx in indices:
                assert idx not in all_indices, f"Index {idx} appears in multiple partitions"
                all_indices.add(idx)
