"""Pseudo-likelihood (Gibbs-like) sampling of the trained model, vectorised over chains.

One sweep = L single-site updates, sites visited in a fresh random order; at each
update every chain resamples x_i ~ p(. | x_-i) ∝ exp(beta * sum_j J_ij(., x_j)).

Because A^h is row-normalised, J_ij(a,b) != J_ji(b,a) in general, so these
conditionals are not exactly those of a single Boltzmann distribution. The
diagnostics therefore track two scores:
  energy  E(x) = -1/2 sum_{i != j} J_ij(x_i, x_j)   (Hamiltonian of the symmetrised couplings)
  PLL(x)  = sum_i log p(x_i | x_-i)                  (the quantity the model was trained on)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import AttentionPotts


@torch.no_grad()
def energy(Jf: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    B = X.shape[0]
    x = X.reshape(B, -1)
    return -0.5 * ((x @ Jf) * x).sum(1)


@torch.no_grad()
def pll(Jf: torch.Tensor, X: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    logp = torch.log_softmax(beta * AttentionPotts.logits_from_J(Jf, X), -1)
    return (logp * X).sum((1, 2))


@torch.no_grad()
def scores(J: torch.Tensor, Z, device="cuda", chunk: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """(energy, PLL) of integer sequences Z (M, L)."""
    Jf = AttentionPotts.flat(J.to(device))
    q = J.shape[-1]
    E, P = [], []
    for s in range(0, len(Z), chunk):
        X = F.one_hot(torch.as_tensor(Z[s:s + chunk], device=device).long(), q).float()
        E.append(energy(Jf, X).cpu()); P.append(pll(Jf, X).cpu())
    return torch.cat(E).numpy(), torch.cat(P).numpy()


@torch.no_grad()
def sample(J: torch.Tensor, init: np.ndarray, n_sweeps: int, beta: float = 1.0,
           record_every: int = 1, snapshots: tuple[int, ...] = (), seed: int = 0,
           device: str = "cuda"):
    """Run len(init) chains for n_sweeps sweeps.

    Returns dict with final states 'Z' (C, L), per-recorded-sweep 'sweeps',
    'energy' (T, C), 'pll' (T, C), 'ham_init' (T, C) (fraction of sites differing
    from the initial sequence), 'gap' (T, C) (gap fraction; gap = last symbol) and
    'snap' {sweep: states} for requested sweeps.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    J = J.to(device)
    L, _, q, _ = J.shape
    Jf = AttentionPotts.flat(J)
    # rows of Jf grouped by site: Jsite[i] (q, L*q)
    Jsite = Jf.reshape(L, q, L * q)
    Z0 = torch.as_tensor(init, device=device).long()
    Z = Z0.clone()
    X = F.one_hot(Z, q).float()
    C = len(Z)
    rec = {"sweeps": [], "energy": [], "pll": [], "ham_init": [], "gap": []}
    snaps = {}

    def record(t):
        rec["sweeps"].append(t)
        rec["energy"].append(energy(Jf, X).cpu())
        rec["pll"].append(pll(Jf, X, beta).cpu())
        rec["ham_init"].append((Z != Z0).float().mean(1).cpu())
        rec["gap"].append((Z == q - 1).float().mean(1).cpu())

    record(0)
    for t in range(1, n_sweeps + 1):
        for i in torch.randperm(L, generator=gen, device=device).tolist():
            logits = beta * (X.reshape(C, -1) @ Jsite[i].T)          # (C, q)
            new = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen).squeeze(1)
            Z[:, i] = new
            X[:, i] = F.one_hot(new, q).float()
        if t % record_every == 0 or t == n_sweeps:
            record(t)
        if t in snapshots:
            snaps[t] = Z.cpu().numpy().copy()
    out = {k: (np.array(v) if k == "sweeps" else torch.stack(v).numpy()) for k, v in rec.items()}
    out["Z"] = Z.cpu().numpy()
    out["snap"] = snaps
    return out
