"""
Data Loader and Non-IID Partitioning for ShapeFL
=================================================
Implements data loading and non-IID partitioning following the paper's methodology.

Paper Method (Section V-A):
- Each training dataset is divided into shards of size 15
- Each computing node is distributed with s shards from k classes
- For FMNIST/CIFAR-10: s=12, k=4
- For CIFAR-100: s=100, k=20

Supported datasets:
- fmnist   : Fashion-MNIST (1×28×28, 10 classes)
- cifar10  : CIFAR-10      (3×32×32, 10 classes)
- cifar100 : CIFAR-100     (3×32×32, 100 classes)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Dict, Tuple
import pandas as pd


# ── Dataset metadata ─────────────────────────────────────────────────────────
DATASET_INFO = {
    "fmnist": {
        "num_classes": 10,
        "input_channels": 1,
        "input_size": (28, 28),
        "shards_per_node": 12,
        "classes_per_node": 4,
    },
    "cifar10": {
        "num_classes": 10,
        "input_channels": 3,
        "input_size": (32, 32),
        "shards_per_node": 12,
        "classes_per_node": 4,
    },
    "cifar100": {
        "num_classes": 100,
        "input_channels": 3,
        "input_size": (32, 32),
        "shards_per_node": 100,
        "classes_per_node": 20,
    },
}

DATASET_DEFAULT_MODEL = {
    "fmnist": "lenet5",
    "cifar10": "mobilenetv2",
    "cifar100": "resnet18",
}


class FMNISTDataset(Dataset):
    """Fashion-MNIST dataset wrapper."""

    def __init__(self, data: np.ndarray, targets: np.ndarray, transform=None):
        self.data = torch.tensor(data, dtype=torch.float32).unsqueeze(1) / 255.0
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target


def _default_dataset_dir() -> str:
    """Return the default dataset directory (``<project>/dataset``)."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "dataset",
    )


def load_fmnist_data(
    data_dir: str = None, use_csv: bool = True
) -> Tuple[Dataset, Dataset]:
    """Load Fashion-MNIST dataset."""
    if data_dir is None:
        data_dir = _default_dataset_dir()

    csv_path = os.path.join(data_dir, "fashion-mnist_train.csv")
    if use_csv and os.path.exists(csv_path):
        print("Loading Fashion-MNIST from CSV files...")
        train_df = pd.read_csv(csv_path)
        train_labels = train_df.iloc[:, 0].values
        train_images = train_df.iloc[:, 1:].values.reshape(-1, 28, 28)

        test_df = pd.read_csv(os.path.join(data_dir, "fashion-mnist_test.csv"))
        test_labels = test_df.iloc[:, 0].values
        test_images = test_df.iloc[:, 1:].values.reshape(-1, 28, 28)

        train_dataset = FMNISTDataset(train_images, train_labels)
        test_dataset = FMNISTDataset(test_images, test_labels)
    else:
        print("Loading Fashion-MNIST from torchvision...")
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=transform
        )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    return train_dataset, test_dataset


def load_cifar10_data(data_dir: str = None, augment: bool = False) -> Tuple[Dataset, Dataset]:
    """Load CIFAR-10 dataset via torchvision.

    Args:
        data_dir: Directory for downloading / caching the dataset.
        augment: If True, apply RandomCrop and RandomHorizontalFlip to training
                 data.  The paper does not mention data augmentation, so the
                 paper-faithful default is False.
    """
    if data_dir is None:
        data_dir = _default_dataset_dir()

    print(f"Loading CIFAR-10 from torchvision... (augment={augment})")
    train_transforms = []
    if augment:
        train_transforms += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
    transform_train = transforms.Compose(train_transforms)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test,
    )
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    return train_dataset, test_dataset


def load_cifar100_data(data_dir: str = None, augment: bool = False) -> Tuple[Dataset, Dataset]:
    """Load CIFAR-100 dataset via torchvision.

    Args:
        data_dir: Directory for downloading / caching the dataset.
        augment: If True, apply RandomCrop and RandomHorizontalFlip to training
                 data.  The paper does not mention data augmentation, so the
                 paper-faithful default is False.
    """
    if data_dir is None:
        data_dir = _default_dataset_dir()

    print(f"Loading CIFAR-100 from torchvision... (augment={augment})")
    train_transforms = []
    if augment:
        train_transforms += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    transform_train = transforms.Compose(train_transforms)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train,
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test,
    )
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    return train_dataset, test_dataset


def load_data(
    dataset_name: str = "fmnist",
    data_dir: str = None,
    augment: bool = False,
) -> Tuple[Dataset, Dataset]:
    """Unified data-loading entry point.

    Args:
        dataset_name: One of ``"fmnist"``, ``"cifar10"``, ``"cifar100"``.
        data_dir: Override default dataset cache directory.
        augment: Whether to apply training-time data augmentation (RandomCrop,
                 RandomHorizontalFlip) for CIFAR datasets. Has no effect on
                 Fashion-MNIST. Default: False to match the paper.
    """
    name = dataset_name.lower()
    if name == "fmnist":
        return load_fmnist_data(data_dir=data_dir)
    elif name == "cifar10":
        return load_cifar10_data(data_dir=data_dir, augment=augment)
    elif name == "cifar100":
        return load_cifar100_data(data_dir=data_dir, augment=augment)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Choose from: fmnist, cifar10, cifar100"
        )


def create_non_iid_partitions(
    dataset: Dataset,
    num_nodes: int,
    shard_size: int = 15,
    shards_per_node: int = 12,
    classes_per_node: int = 4,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Create non-IID data partitions following the paper's methodology.

    Paper Method:
    - Divide dataset into shards of ``shard_size``
    - Each node gets ``shards_per_node`` shards from ``classes_per_node`` classes
    """
    np.random.seed(seed)

    if hasattr(dataset, "targets"):
        if isinstance(dataset.targets, torch.Tensor):
            labels = dataset.targets.numpy()
        else:
            labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])

    class_shards = {}
    for c in range(num_classes):
        indices = class_indices[c]
        num_shards = len(indices) // shard_size
        shards = [indices[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]
        class_shards[c] = shards

    print(f"Created shards per class: {[len(class_shards[c]) for c in range(num_classes)]}")

    partitions = {n: [] for n in range(num_nodes)}

    for node_id in range(num_nodes):
        available_classes = [c for c in range(num_classes) if len(class_shards[c]) > 0]
        if len(available_classes) < classes_per_node:
            selected_classes = available_classes
        else:
            selected_classes = np.random.choice(
                available_classes, classes_per_node, replace=False
            )

        shards_per_class = shards_per_node // len(selected_classes)
        extra_shards = shards_per_node % len(selected_classes)

        for i, c in enumerate(selected_classes):
            n_shards = shards_per_class + (1 if i < extra_shards else 0)
            for _ in range(n_shards):
                if len(class_shards[c]) > 0:
                    shard = class_shards[c].pop()
                    partitions[node_id].extend(shard)

    print("\nPartition Statistics:")
    for node_id in range(num_nodes):
        node_labels = labels[partitions[node_id]]
        unique, counts = np.unique(node_labels, return_counts=True)
        print(
            f"  Node {node_id}: {len(partitions[node_id])} samples, "
            f"classes: {dict(zip(unique.tolist(), counts.tolist()))}"
        )

    return partitions


def get_node_dataloader(
    dataset: Dataset, indices: List[int], batch_size: int = 32, shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader for a specific node's data partition."""
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def save_partitions(partitions: Dict[int, List[int]], filepath: str):
    """Save partitions to a JSON file."""
    partitions_str = {str(k): v for k, v in partitions.items()}
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(partitions_str, f)
    print(f"Partitions saved to {filepath}")


def load_partitions(filepath: str) -> Dict[int, List[int]]:
    """Load partitions from a JSON file."""
    with open(filepath, "r") as f:
        partitions_str = json.load(f)
    partitions = {int(k): v for k, v in partitions_str.items()}
    print(f"Partitions loaded from {filepath}")
    return partitions


def get_data_distribution(dataset: Dataset, indices: List[int]) -> Dict[int, int]:
    """Get the class distribution for a subset of the dataset."""
    if hasattr(dataset, "targets"):
        if isinstance(dataset.targets, torch.Tensor):
            labels = dataset.targets[indices].numpy()
        else:
            labels = np.array(dataset.targets)[indices]
    else:
        labels = np.array([dataset[i][1] for i in indices])

    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def get_partition_label_counts(
    dataset: Dataset,
    partitions: Dict[int, List[int]],
    num_classes: int,
) -> Dict[int, np.ndarray]:
    """Return per-node label-count vectors for simulation-side baseline planning."""
    if hasattr(dataset, "targets"):
        if isinstance(dataset.targets, torch.Tensor):
            labels = dataset.targets.numpy()
        else:
            labels = np.array(dataset.targets)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    label_counts: Dict[int, np.ndarray] = {}
    for node_id, indices in partitions.items():
        node_labels = labels[indices]
        label_counts[node_id] = np.bincount(node_labels, minlength=num_classes).astype(np.int64)
    return label_counts


def create_client_eval_partitions(
    test_dataset: Dataset,
    reference_label_counts: Dict[int, np.ndarray],
    num_classes: int,
    samples_per_client: int | None = None,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Create deterministic held-out test splits matched to train-label skew.

    The allocation is label-aware and non-overlapping: each client's held-out
    split mirrors the class proportions in its training shard as closely as the
    shared test set permits.
    """
    if hasattr(test_dataset, "targets"):
        if isinstance(test_dataset.targets, torch.Tensor):
            labels = test_dataset.targets.numpy()
        else:
            labels = np.asarray(test_dataset.targets)
    else:
        labels = np.asarray([test_dataset[idx][1] for idx in range(len(test_dataset))])

    rng = np.random.RandomState(seed)
    class_indices: Dict[int, List[int]] = {}
    for class_id in range(num_classes):
        indices = np.where(labels == class_id)[0].tolist()
        rng.shuffle(indices)
        class_indices[class_id] = indices

    node_ids = sorted(reference_label_counts.keys())
    if samples_per_client is None:
        samples_per_client = max(1, len(labels) // max(len(node_ids), 1))

    partitions: Dict[int, List[int]] = {node_id: [] for node_id in node_ids}
    remaining_pool = set(range(len(labels)))

    for node_id in node_ids:
        train_counts = np.asarray(reference_label_counts[node_id], dtype=np.float64)
        total_train = float(train_counts.sum())
        if total_train <= 0.0:
            continue

        proportions = train_counts / total_train
        target_counts = np.floor(proportions * samples_per_client).astype(int)
        remainder = int(samples_per_client - target_counts.sum())
        if remainder > 0:
            fractional = proportions * samples_per_client - target_counts
            for class_id in np.argsort(-fractional)[:remainder]:
                target_counts[int(class_id)] += 1

        assigned: List[int] = []
        for class_id in range(num_classes):
            take = min(target_counts[class_id], len(class_indices[class_id]))
            if take <= 0:
                continue
            chosen = class_indices[class_id][:take]
            class_indices[class_id] = class_indices[class_id][take:]
            assigned.extend(chosen)
            remaining_pool.difference_update(chosen)

        shortfall = max(0, samples_per_client - len(assigned))
        if shortfall > 0 and remaining_pool:
            pool_list = sorted(remaining_pool)
            fill = rng.choice(
                pool_list,
                size=min(shortfall, len(pool_list)),
                replace=False,
            ).tolist()
            assigned.extend(fill)
            remaining_pool.difference_update(fill)
            for class_id in range(num_classes):
                if not class_indices[class_id]:
                    continue
                class_indices[class_id] = [
                    index for index in class_indices[class_id] if index in remaining_pool
                ]

        partitions[node_id] = sorted(assigned)

    return partitions
