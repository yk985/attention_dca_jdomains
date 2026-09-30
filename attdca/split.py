"""Identity-based clustering and a sequence-disjoint train/test split.

Greedy (CD-HIT style) clustering: sequences are visited from the least gapped to the
most gapped; each unassigned sequence becomes a representative and absorbs every
unassigned sequence within `thresh` normalised Hamming distance of it. Whole clusters
are then assigned to train or test, and any test sequence still within `thresh` of a
training sequence is dropped, which makes the separation exact.
"""
from __future__ import annotations

import numpy as np
import torch

from .data import one_hot, GAP
from .metrics import min_hamming


def greedy_cluster(Z: np.ndarray, thresh: float = 0.2, device: str = "cuda",
                   verbose: bool = True) -> np.ndarray:
    """Cluster labels (M,), ordered by decreasing cluster size."""
    M, L = Z.shape
    order = np.argsort((Z == GAP).sum(1), kind="stable")      # fewest gaps first
    X = one_hot(Z[order], device=device).flatten(1)
    free = torch.ones(M, dtype=torch.bool, device=device)
    lab = torch.full((M,), -1, dtype=torch.long, device=device)
    min_ident = L * (1 - thresh)
    c = 0
    for i in range(M):
        if not free[i]:
            continue
        ident = X[i] @ X.T                                     # (M,)
        hit = free & (ident >= min_ident)
        lab[hit] = c
        free &= ~hit
        c += 1
        if verbose and c % 2000 == 0:
            print(f"  {c} clusters, {int(free.sum())} sequences left", flush=True)
    out = np.empty(M, dtype=np.int64)
    out[order] = lab.cpu().numpy()
    # relabel by decreasing size
    sizes = np.bincount(out)
    rank = np.empty_like(sizes)
    rank[np.argsort(-sizes, kind="stable")] = np.arange(len(sizes))
    return rank[out]


def split_by_cluster(Z: np.ndarray, labels: np.ndarray, test_frac: float = 0.2,
                     thresh: float = 0.2, drop: str = "exact", seed: int = 0,
                     device: str = "cuda"):
    """Assign whole clusters to test until test_frac is reached, then clean up the test side.

    Clusters are shuffled, so the test set is a random sample of the family's
    diversity rather than its rarest corner.

    `drop` decides which test sequences are removed afterwards:
      "exact"     - only sequences that are byte-identical to a training sequence. A test
                    sequence that merely *resembles* a training one is kept, so the test set
                    still represents the family's density instead of only its isolated tail.
      "threshold" - everything within `thresh` of a training sequence, which forces a hard
                    margin but biases the test set towards the sparse regions.
    Returns (idx_train, idx_test, stats).
    """
    if drop not in ("exact", "threshold"):
        raise ValueError("drop must be 'exact' or 'threshold'")
    rng = np.random.default_rng(seed)
    sizes = np.bincount(labels)
    clusters = rng.permutation(len(sizes))
    target = test_frac * len(Z)
    take, tot = [], 0
    for c in clusters:
        if tot >= target:
            break
        take.append(c)
        tot += sizes[c]
    in_test = np.isin(labels, take)
    idx_tr = np.flatnonzero(~in_test)
    idx_te = np.flatnonzero(in_test)
    d = min_hamming(Z[idx_te], Z[idx_tr], device=device)
    keep = d > 0 if drop == "exact" else d >= thresh
    stats = {"n_clusters": int(len(sizes)), "n_test_clusters": len(take), "drop_rule": drop,
             "dropped_from_test": int((~keep).sum()), "nn_before": float(d.mean()),
             "nn_after": float(d[keep].mean()) if keep.any() else float("nan"),
             "frac_test_within_thresh": float((d[keep] < thresh).mean()) if keep.any() else 0.0}
    return idx_tr, idx_te[keep], stats
