"""Statistics used to compare generated and natural alignments."""
from __future__ import annotations

import numpy as np
import torch
from scipy import stats

from .data import Q, one_hot, identity_counts


def freqs(Z: np.ndarray, w: np.ndarray | None = None, device="cuda"):
    """Weighted one- and two-site frequencies.

    f1 (L, q), f2 (L, L, q, q); weights normalised to sum to 1 (uniform if w is None).
    """
    M, L = Z.shape
    w = np.full(M, 1.0 / M) if w is None else w / w.sum()
    X = one_hot(Z, device=device, dtype=torch.float64).reshape(M, L * Q)
    wt = torch.as_tensor(w, device=device, dtype=torch.float64)
    f1 = wt @ X
    f2 = (X * wt[:, None]).T @ X
    return f1.reshape(L, Q).cpu().numpy(), f2.reshape(L, Q, L, Q).permute(0, 2, 1, 3).cpu().numpy()


def connected_corr(f1, f2):
    """C_ij(a,b) = f_ij(a,b) - f_i(a) f_j(b), upper triangle i<j flattened."""
    L = f1.shape[0]
    C = f2 - f1[:, None, :, None] * f1[None, :, None, :]
    iu = np.triu_indices(L, k=1)
    return C[iu].reshape(-1)


def compare(x: np.ndarray, y: np.ndarray) -> dict:
    """Agreement of two statistic vectors (x = reference, y = model)."""
    r, _ = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    return {"pearson": float(r), "spearman": float(rho), "slope": float(slope),
            "intercept": float(intercept), "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
            "n": int(len(x))}


def min_hamming(Za, Zb, exclude_self: bool = False, device="cuda") -> np.ndarray:
    """For each a in Za: min normalised Hamming distance to the set Zb."""
    L = Za.shape[1]
    out = []
    for s in range(0, len(Za), 4096):
        ident = identity_counts(Za[s:s + 4096], Zb, device=device)
        if exclude_self:
            idx = torch.arange(s, min(s + 4096, len(Za)), device=device)
            ident[torch.arange(len(idx)), idx] = -1
        out.append((1 - ident.max(1).values / L).cpu())
    return torch.cat(out).numpy()


def pairwise_hamming(Za, Zb=None, n_max: int = 3000, seed: int = 0, device="cuda") -> np.ndarray:
    """All pairwise normalised Hamming distances (upper triangle if Zb is None), on subsets."""
    rng = np.random.default_rng(seed)
    Za = Za[rng.permutation(len(Za))[:n_max]]
    L = Za.shape[1]
    if Zb is None:
        D = 1 - identity_counts(Za, Za, device=device).cpu().numpy() / L
        return D[np.triu_indices(len(Za), k=1)]
    Zb = Zb[rng.permutation(len(Zb))[:n_max]]
    return (1 - identity_counts(Za, Zb, device=device).cpu().numpy() / L).ravel()


def autocorr_time(x: np.ndarray, c: float = 5.0) -> tuple[np.ndarray, float]:
    """Chain-averaged normalised autocorrelation of x (T, C) and Sokal-windowed tau_int."""
    x = x - x.mean(0, keepdims=True)
    T = x.shape[0]
    f = np.fft.rfft(x, n=2 * T, axis=0)
    acf = np.fft.irfft(f * np.conj(f), axis=0)[:T].mean(1)
    acf /= acf[0]
    taus = 2 * np.cumsum(acf) - 1
    m = np.arange(T) < c * taus
    win = int(np.argmin(m)) if not m.all() else T - 1
    return acf, float(taus[win])


def contact_ppv(J: np.ndarray, contact_file: str, cutoff: float = 8.0, min_sep: int = 5):
    """PPV curve of APC-corrected Frobenius scores (gauge-fixed, gap excluded)."""
    L = J.shape[0]
    Js = 0.5 * (J + J.transpose(1, 0, 3, 2))[:, :, :-1, :-1]
    Js = Js - Js.mean(2, keepdims=True) - Js.mean(3, keepdims=True) + Js.mean((2, 3), keepdims=True)
    F = np.sqrt((Js ** 2).sum((2, 3)))
    np.fill_diagonal(F, 0)
    Fi = F.sum(0) / (L - 1)                      # APC with off-diagonal means
    F = F - np.outer(Fi, Fi) / (F.sum() / (L * (L - 1)))
    dist = np.full((L, L), np.inf)
    for i, j, dd in np.loadtxt(contact_file):
        dist[int(i) - 1, int(j) - 1] = dist[int(j) - 1, int(i) - 1] = dd
    iu = [(i, j) for i in range(L) for j in range(i + min_sep, L)]
    s = np.array([F[i, j] for i, j in iu])
    c = np.array([dist[i, j] < cutoff for i, j in iu])
    order = np.argsort(-s)
    return np.cumsum(c[order]) / np.arange(1, len(order) + 1), float(c.mean())
