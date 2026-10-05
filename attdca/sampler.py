"""Pseudo-likelihood (Gibbs-like) sampling of the trained model, vectorised over chains.

One sweep = L single-site updates, sites visited in a fresh random order; at each
update every chain resamples x_i ~ p(. | x_-i) ∝ exp(beta * sum_j J_ij(., x_j)).

Because A^h is row-normalised, J_ij(a,b) != J_ji(b,a) in general, so these
conditionals are not exactly those of a single Boltzmann distribution. The
diagnostics therefore track two scores:
  energy  E(x) = -1/2 sum_{i != j} J_ij(x_i, x_j)   (Hamiltonian of the symmetrised couplings)
  PLL(x)  = sum_i log p(x_i | x_-i)                  (the quantity the model was trained on)

Optional extras (both enter every site update):
  h          fields h_i(a), (L, q) shared or (C, L, q) per chain:  energy gains -sum_i h_i(x_i)
  restraint  harmonic restraint on linear coordinates (proposal C1):
               E_lam(x) = E(x) + lam/2 * sum_c (PC_c(x) - t_c)^2,   PC_c(x) = sum_i V[c, i, x_i] - m_c
             PC is linear in the one-hot sequence, so this is itself a Potts energy; the sampler
             keeps PC per chain and updates it by V[c, i, new] - V[c, i, old] after each move.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import AttentionPotts


@torch.no_grad()
def energy(Jf: torch.Tensor, X: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
    B = X.shape[0]
    x = X.reshape(B, -1)
    E = -0.5 * ((x @ Jf) * x).sum(1)
    if h is not None:
        E = E - (X * h).sum((1, 2))
    return E


@torch.no_grad()
def pll(Jf: torch.Tensor, X: torch.Tensor, beta: float = 1.0, h: torch.Tensor | None = None) -> torch.Tensor:
    logits = AttentionPotts.logits_from_J(Jf, X)
    if h is not None:
        logits = logits + h
    logp = torch.log_softmax(beta * logits, -1)
    return (logp * X).sum((1, 2))


@torch.no_grad()
def scores(J: torch.Tensor, Z, device="cuda", chunk: int = 20000, h=None) -> tuple[np.ndarray, np.ndarray]:
    """(energy, PLL) of integer sequences Z (M, L); h optional fields (L, q)."""
    Jf = AttentionPotts.flat(J.to(device))
    q = J.shape[-1]
    h = None if h is None else torch.as_tensor(h, dtype=torch.float32, device=device)
    E, P = [], []
    for s in range(0, len(Z), chunk):
        X = F.one_hot(torch.as_tensor(Z[s:s + chunk], device=device).long(), q).float()
        hb = h[s:s + chunk] if h is not None and h.dim() == 3 else h
        E.append(energy(Jf, X, hb).cpu()); P.append(pll(Jf, X, 1.0, hb).cpu())
    return torch.cat(E).numpy(), torch.cat(P).numpy()


@torch.no_grad()
def sample(J: torch.Tensor, init: np.ndarray, n_sweeps: int, beta: float = 1.0,
           record_every: int = 1, snapshots: tuple[int, ...] = (), seed: int = 0,
           device: str = "cuda", h=None, restraint: dict | None = None):
    """Run len(init) chains for n_sweeps sweeps.

    h: optional fields (L, q) or (C, L, q). restraint: optional dict with V (n, L, q), m (n,),
    t (n,) or (C, n) targets and lam (stiffness, scalar or (C,)), see module docstring.

    Returns dict with final states 'Z' (C, L), per-recorded-sweep 'sweeps',
    'energy' (T, C) (model energy incl. fields, excluding the restraint), 'pll' (T, C),
    'ham_init' (T, C) (fraction of sites differing from the initial sequence), 'gap' (T, C)
    (gap fraction; gap = last symbol), 'pc' (T, C, n) if restrained, and 'snap' {sweep: states}.
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
    if h is not None:
        h = torch.as_tensor(h, dtype=torch.float32, device=device)
        h = h.expand(C, L, q) if h.dim() == 2 else h
    if restraint is not None:
        Vr = torch.as_tensor(restraint["V"], dtype=torch.float32, device=device)      # (n, L, q)
        mr = torch.as_tensor(restraint["m"], dtype=torch.float32, device=device)
        tr = torch.as_tensor(restraint["t"], dtype=torch.float32, device=device)
        tr = tr.expand(C, len(mr)) if tr.dim() == 1 else tr                          # (C, n)
        lam = torch.as_tensor(restraint["lam"], dtype=torch.float32, device=device).expand(C)  # (C,)
        Vsite = Vr.permute(1, 2, 0)                                                   # (L, q, n)
        pc = Vsite[torch.arange(L, device=device)[None, :], Z].sum(1) - mr            # (C, n)
    rec = {"sweeps": [], "energy": [], "pll": [], "ham_init": [], "gap": []}
    if restraint is not None:
        rec["pc"] = []
    snaps = {}

    def record(t):
        rec["sweeps"].append(t)
        rec["energy"].append(energy(Jf, X, h).cpu())
        rec["pll"].append(pll(Jf, X, beta, h).cpu())
        rec["ham_init"].append((Z != Z0).float().mean(1).cpu())
        rec["gap"].append((Z == q - 1).float().mean(1).cpu())
        if restraint is not None:
            rec["pc"].append(pc.cpu())

    record(0)
    for t in range(1, n_sweeps + 1):
        for i in torch.randperm(L, generator=gen, device=device).tolist():
            logits = X.reshape(C, -1) @ Jsite[i].T                   # (C, q)
            if h is not None:
                logits = logits + h[:, i]
            if restraint is not None:
                pc_rest = pc - Vsite[i][Z[:, i]]                       # PC without site i's term
                cand = pc_rest[:, None, :] + Vsite[i][None]            # (C, q, n) PC for each letter
                logits = logits - 0.5 * lam[:, None] * (cand - tr[:, None, :]).pow(2).sum(-1)
            new = torch.multinomial(torch.softmax(beta * logits, -1), 1, generator=gen).squeeze(1)
            Z[:, i] = new
            X[:, i] = F.one_hot(new, q).float()
            if restraint is not None:
                pc = pc_rest + Vsite[i][new]
        if restraint is not None:                                    # drop float32 drift once a sweep
            pc = Vsite[torch.arange(L, device=device)[None, :], Z].sum(1) - mr
        if t % record_every == 0 or t == n_sweeps:
            record(t)
        if t in snapshots:
            snaps[t] = Z.cpu().numpy().copy()
    out = {k: (np.array(v) if k == "sweeps" else torch.stack(v).numpy()) for k, v in rec.items()}
    out["Z"] = Z.cpu().numpy()
    out["snap"] = snaps
    return out
