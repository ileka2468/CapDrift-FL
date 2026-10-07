"""
Dirichlet-based Non-IID Dataset Partitioner for OmniFed.
Implements standard statistical heterogeneity partitioning (Hsu et al., 2019; Li et al., 2020)
using Dirichlet distribution Dir(alpha) to induce controlled label skew.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder


def partition_dataset_dirichlet(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42
) -> List[List[int]]:
    """
    Partitions a dataset across `num_clients` using a Dirichlet distribution Dir(alpha) over class labels.
    - alpha -> inf (e.g. 100.0): Uniform IID distribution across clients.
    - alpha = 0.5: Moderate Non-IID label skew (realistic hospital specialization).
    - alpha = 0.1: Severe Non-IID label skew (extreme client bias).
    """
    rng = np.random.RandomState(seed)

    # Extract targets from dataset
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, "samples"):
        targets = np.array([s[1] for s in dataset.samples])
    else:
        # Fallback extraction
        targets = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(targets))
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    # Partition each class among clients according to Dirichlet distribution
    for c in range(num_classes):
        c_indices = np.where(targets == c)[0]
        rng.shuffle(c_indices)

        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Normalize proportions to partition size
        proportions = proportions / proportions.sum()
        split_points = (np.cumsum(proportions) * len(c_indices)).astype(int)[:-1]

        client_splits = np.split(c_indices, split_points)
        for client_id, split in enumerate(client_splits):
            client_indices[client_id].extend(split.tolist())

    # Ensure every client has at least min_samples to prevent zero-sample RandomSampler error
    min_samples = max(2, min(4, len(targets) // (num_clients * 2)))
    for c_id in range(num_clients):
        while len(client_indices[c_id]) < min_samples:
            # Borrow sample from the client with the largest partition
            richest_client = max(range(num_clients), key=lambda i: len(client_indices[i]))
            if len(client_indices[richest_client]) <= min_samples:
                break
            borrowed = client_indices[richest_client].pop()
            client_indices[c_id].append(borrowed)

    for c_id in range(num_clients):
        rng.shuffle(client_indices[c_id])

    return client_indices


def create_client_subsets(
    dataset: Dataset,
    client_indices: List[List[int]]
) -> List[Subset]:
    """Wraps index lists into PyTorch Subsets."""
    return [Subset(dataset, indices) for indices in client_indices]


def inspect_client_distribution(
    dataset: Dataset,
    client_indices: List[List[int]]
) -> Dict[int, Dict[int, int]]:
    """Inspects class histogram per client for telemetry and reporting."""
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    elif hasattr(dataset, "samples"):
        targets = np.array([s[1] for s in dataset.samples])
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])

    dist = {}
    for c_id, indices in enumerate(client_indices):
        c_targets = targets[indices]
        classes, counts = np.unique(c_targets, return_counts=True)
        dist[c_id] = {int(cls): int(cnt) for cls, cnt in zip(classes, counts)}
    return dist
