"""Factored-attention Potts model (AttentionDCA).

    J_ij(a, b) = sum_h A^h_ij V^h(a, b),     A^h = softmax_j( (Q^h)^T K^h ),  A^h_ii = 0

Conditional used by the pseudo-likelihood and by the sampler:

    p(x_i = a | x_-i) ∝ exp( sum_{j != i} J_ij(a, x_j) )

No fields: as in the original AttentionDCA, single-site conservation is absorbed
in the couplings (the gap / conserved-residue columns of V^h act as fields).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPotts(nn.Module):
    def __init__(self, L: int, q: int = 21, H: int = 32, d: int = 23, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.L, self.q, self.H, self.d = L, q, H, d
        self.Qm = nn.Parameter(torch.randn(H, d, L, generator=g))
        self.Km = nn.Parameter(torch.randn(H, d, L, generator=g))
        self.V = nn.Parameter(torch.randn(H, q, q, generator=g) * 0.1)
        self.register_buffer("diag", torch.eye(L, dtype=torch.bool))

    def attention(self) -> torch.Tensor:
        """(H, L, L), row-stochastic, zero diagonal."""
        e = torch.einsum("hdi,hdj->hij", self.Qm, self.Km)
        e = e.masked_fill(self.diag, float("-inf"))
        return torch.softmax(e, dim=-1)

    def couplings(self) -> torch.Tensor:
        """J as (L, L, q, q): J[i, j, a, b]."""
        return torch.einsum("hij,hab->ijab", self.attention(), self.V)

    @staticmethod
    def flat(J: torch.Tensor) -> torch.Tensor:
        """(L*q, L*q) matrix Jf[(i,a), (j,b)] = J_ij(a,b)."""
        L, _, q, _ = J.shape
        return J.permute(0, 2, 1, 3).reshape(L * q, L * q)

    @staticmethod
    def logits_from_J(Jf: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """X one-hot (B, L, q) -> conditional logits (B, L, q): sum_j J_ij(a, x_j)."""
        B, L, q = X.shape
        return (X.reshape(B, L * q) @ Jf.T).reshape(B, L, q)

    def neg_log_pl(self, Z: torch.Tensor, w: torch.Tensor, lambd: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Weighted negative log pseudo-likelihood + L2 on J.

        Z (B, L) int, w (B,) weights summing to 1 inside the batch.
        Returns (loss, nlpl) with nlpl = -sum_m w_m sum_i log p(x_i^m | x_-i^m).
        """
        J = self.couplings()
        X = nn.functional.one_hot(Z, self.q).float()
        logp = torch.log_softmax(self.logits_from_J(self.flat(J), X), dim=-1)
        ll = (logp * X).sum((1, 2))                      # (B,)
        nlpl = -(w * ll).sum()
        return nlpl + lambd * J.pow(2).sum(), nlpl
