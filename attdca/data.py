"""Reading aligned FASTA files, integer encoding and sequence reweighting."""
from __future__ import annotations

import numpy as np
import torch

ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"   # index 20 = gap
Q = len(ALPHABET)
GAP = ALPHABET.index("-")
_AA_TO_INT = {c: i for i, c in enumerate(ALPHABET)}  # anything else (X, B, Z, ...) -> gap


def read_fasta(path: str) -> tuple[list[str], list[str]]:
    names, seqs, cur = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                names.append(line[1:])
                cur = []
            else:
                cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return names, seqs


def encode(seqs: list[str]) -> np.ndarray:
    """Upper-case match columns only (lower-case / '.' are insert columns)."""
    keep = [k for k, c in enumerate(seqs[0]) if not (c.islower() or c == ".")]
    Z = np.empty((len(seqs), len(keep)), dtype=np.int64)
    for m, s in enumerate(seqs):
        if len(s) != len(seqs[0]):
            raise ValueError("sequences are not aligned")
        Z[m] = [_AA_TO_INT.get(s[k].upper(), GAP) for k in keep]
    return Z


def decode(Z: np.ndarray) -> list[str]:
    alpha = np.array(list(ALPHABET))
    return ["".join(alpha[row]) for row in Z]


def write_fasta(path: str, Z: np.ndarray, prefix: str = "gen") -> None:
    with open(path, "w") as f:
        for m, s in enumerate(decode(Z)):
            f.write(f">{prefix}_{m}\n{s}\n")


def load_msa(path: str, max_gap_frac: float = 0.9, dedup: bool = True) -> np.ndarray:
    """(M, L) int array; drops sequences with too many gaps and exact duplicates."""
    _, seqs = read_fasta(path)
    Z = encode(seqs)
    Z = Z[(Z == GAP).mean(1) <= max_gap_frac]
    if dedup:
        _, idx = np.unique(Z, axis=0, return_index=True)
        Z = Z[np.sort(idx)]
    return Z


def one_hot(Z, q: int = Q, device=None, dtype=torch.float32) -> torch.Tensor:
    Z = torch.as_tensor(Z, device=device)
    return torch.nn.functional.one_hot(Z.long(), q).to(dtype)


def identity_counts(Za, Zb, device="cuda", chunk: int = 4096) -> torch.Tensor:
    """Number of identical positions between every a in Za and b in Zb, (Ma, Mb) on device."""
    Xb = one_hot(Zb, device=device).flatten(1)
    out = []
    for s in range(0, len(Za), chunk):
        Xa = one_hot(Za[s:s + chunk], device=device).flatten(1)
        out.append(Xa @ Xb.T)
    return torch.cat(out)


def compute_theta(Z: np.ndarray, n_pairs: int = 200_000, seed: int = 0) -> float:
    """DCAUtils rule: theta = min(0.5, 0.38*0.32 / <fraction identity>)."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(Z), n_pairs)
    j = rng.integers(0, len(Z), n_pairs)
    keep = i != j
    meanfracid = (Z[i[keep]] == Z[j[keep]]).mean()
    return float(min(0.5, 0.38 * 0.32 / meanfracid))


def compute_weights(Z: np.ndarray, theta: float | str = "auto", device="cuda",
                    chunk: int = 2048) -> tuple[np.ndarray, float, float]:
    """w_m = 1 / #{n : d_H(m, n) < theta L} (n = m included). Returns (w, Meff, theta)."""
    if theta == "auto":
        theta = compute_theta(Z)
    M, L = Z.shape
    thresh = np.floor(theta * L)
    Xall = one_hot(Z, device=device).flatten(1)
    counts = torch.zeros(M, device=device)
    for s in range(0, M, chunk):
        dist = L - Xall[s:s + chunk] @ Xall.T
        counts[s:s + chunk] = (dist < thresh).sum(1).float()
    w = (1.0 / counts).cpu().numpy()
    return w, float(w.sum()), float(theta)
