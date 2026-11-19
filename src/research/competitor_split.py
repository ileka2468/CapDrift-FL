"""Deterministic, stratified holdouts of conservative filename/content groups."""

from bisect import bisect_left
from collections import defaultdict
from math import ceil, isfinite
from pathlib import Path
import re

import numpy as np


_PNEUMONIA = re.compile(r"person(\d+)_(?:bacteria|virus)_\d+", re.IGNORECASE)
_NORMAL = re.compile(r"((?:NORMAL2-)?IM)-(\d+)(?:-\d+)+", re.IGNORECASE)


def filename_group_ids(paths, content_hashes):
    """Infer conservative subject proxies, then merge exact image aliases.

    A pneumonia person number is grouped across bacteria/virus suffixes.
    Normal ''IM'' and ''NORMAL2-IM'' numbers remain distinct namespaces.
    Unrecognized filenames are only linked by exact-byte hashes.
    Return opaque integer group IDs and the count of recognized filenames.
    """
    if len(paths) != len(content_hashes):
        raise ValueError("paths and content_hashes must have the same length")
    parents = list(range(len(paths)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def merge(first, second):
        a, b = sorted((root(first), root(second)))
        parents[b] = a

    keys, digests = {}, {}
    recognized = 0
    for i, (path, digest) in enumerate(zip(paths, content_hashes)):
        stem = Path(path).stem
        person = _PNEUMONIA.fullmatch(stem)
        normal = _NORMAL.fullmatch(stem)
        if person or normal:
            key = ("person", int(person[1])) if person else (normal[1].lower(), int(normal[2]))
            recognized += 1
            merge(i, keys.setdefault(key, i))
        if digest is not None:
            merge(i, digests.setdefault(digest, i))
    return tuple(root(i) for i in range(len(paths))), recognized


def _reachable_counts(groups):
    """Return count -> (parent count, member IDs), with deterministic ties."""
    counts = {(0, 0): None}
    for members, counts_by_class in groups:
        a, b = counts_by_class
        for x, y in tuple(counts):
            counts.setdefault((x + a, y + b), ((x, y), members))
    return counts


def _recover(counts, target):
    while target != (0, 0):
        target, members = counts[target]
        yield from members


def split_indices(dataset, *, policy="filename-group", seed=42, holdout_fraction=0.2, indices=None):
    """Split eligible global image indices without splitting an inferred group.

    For binary classification, select the closest feasible holdout size and,
    within that size, minimize its deviation from the training population's
    class ratio. Require both classes in both sets for group-holdout evaluation.
    A nested split takes only the outer train indices via *indices*.
    """
    if not isfinite(holdout_fraction) or not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in (0, 1)")
    eligible = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
    if len(eligible) < 2 or len(eligible) != len(set(eligible)):
        raise ValueError("split needs at least two distinct eligible indices")
    if min(eligible) < 0 or max(eligible) >= len(dataset):
        raise ValueError("eligible index is outside the dataset")
    rng = np.random.RandomState(seed)
    target_size = min(len(eligible) - 1, ceil(len(eligible) * holdout_fraction))
    if policy == "image":
        shuffled = rng.permutation(eligible).tolist()
        return shuffled[:-target_size], shuffled[-target_size:]
    if policy != "filename-group":
        raise ValueError(f"unknown split policy: {policy}")
    groups = getattr(dataset, "group_ids", None)
    if groups is None or len(groups) != len(dataset):
        raise ValueError("filename-group split requires one group ID per image")
    labels = {i: int(dataset.targets[i]) for i in eligible}
    classes = sorted(set(labels.values()))
    if len(classes) != 2:
        raise ValueError("group stratification requires two classes")
    members = defaultdict(list)
    for i in eligible:
        members[groups[i]].append(i)
    ordered = list(members.values())
    populations = [sum(y == c for y in labels.values()) for c in classes]
    pure0, pure1, mixed = [], [], []
    for j in rng.permutation(len(ordered)):
        group = ordered[j]
        n0 = sum(labels[i] == classes[0] for i in group)
        n1 = len(group) - n0
        collection = pure0 if n1 == 0 else pure1 if n0 == 0 else mixed
        collection.append((group, (n0, n1)))
    reach0, reach1, combined = (_reachable_counts(g) for g in (pure0, pure1, mixed))
    reachable1 = sorted(y for _, y in reach1)
    best = selected = None
    total = len(eligible)
    for mx, my in combined:
        for x, _ in reach0:
            xsum = mx + x
            if not 0 < xsum < populations[0]:
                continue
            lower = bisect_left(reachable1, 1 - my)
            upper = bisect_left(reachable1, populations[1] - my)
            candidate = bisect_left(reachable1, target_size - xsum - my, lower, upper)
            for j in (candidate - 1, candidate):
                if not lower <= j < upper:
                    continue
                y = reachable1[j]
                n = xsum + my + y
                score = (abs(n - target_size), abs(xsum * total - n * populations[0]))
                if best is None or score < best:
                    best, selected = score, ((x, 0), (0, y), (mx, my))
    if selected is None:
        raise ValueError("no filename-group split can put both classes in both sets")
    holdout = set()
    for counts, key in zip((reach0, reach1, combined), selected):
        holdout.update(_recover(counts, key))
    shuffled = rng.permutation(eligible).tolist()
    return [i for i in shuffled if i not in holdout], [i for i in shuffled if i in holdout]
