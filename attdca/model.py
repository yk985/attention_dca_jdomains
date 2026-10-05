"""Factored-attention Potts model (AttentionDCA).

    J_ij(a, b) = sum_h A^h_ij V^h(a, b),     A^h = softmax_j( (Q^h)^T K^h ),  A^h_ii = 0

Conditional used by the pseudo-likelihood and by the sampler:

    p(x_i = a | x_-i) ∝ exp( sum_{j != i} J_ij(a, x_j) )

By default no fields: as in the original AttentionDCA, single-site conservation is
absorbed in the couplings (the gap / conserved-residue columns of V^h act as fields).

Optional fields (`field=`):
  "const"  h_i(a) = W_0,ia                                  (proposal M2)
  "quad"   h_i(a | z) = sum_f W_f,ia phi_f(z),  phi(z) = (1, u1, u2, u1^2, u2^2, u1 u2),
           u = (z - mu) / sd the standardised PCA coordinates   (proposal C3)
which enter the conditional as  p(x_i = a | x_-i [, z]) ∝ exp( h_i(a) + sum_{j != i} J_ij(a, x_j) ).
"""
from __future__ import annotations

import torch
import torch.nn as nn


N_FEAT = {"none": 0, "const": 1, "quad": 6}


class AttentionPotts(nn.Module):
    def __init__(self, L: int, q: int = 21, H: int = 32, d: int = 23, seed: int = 0,
                 field: str = "none", n_cond: int = 2):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.L, self.q, self.H, self.d = L, q, H, d
        self.Qm = nn.Parameter(torch.randn(H, d, L, generator=g))
        self.Km = nn.Parameter(torch.randn(H, d, L, generator=g))
        self.V = nn.Parameter(torch.randn(H, q, q, generator=g) * 0.1)
        self.register_buffer("diag", torch.eye(L, dtype=torch.bool))
        self.field = field
        if N_FEAT[field]:
            self.W = nn.Parameter(torch.zeros(N_FEAT[field], L, q))
        if field == "quad":                       # standardisation of the conditioning variable
            self.register_buffer("cond_mu", torch.zeros(n_cond))
            self.register_buffer("cond_sd", torch.ones(n_cond))

    def features(self, z: torch.Tensor | None, B: int) -> torch.Tensor:
        """phi (B, n_feat) for conditioning coordinates z (B, 2) (ignored unless field='quad')."""
        if self.field == "const":
            return torch.ones(B, 1, device=self.W.device)
        u = (z - self.cond_mu) / self.cond_sd
        u1, u2 = u[:, 0], u[:, 1]
        return torch.stack([torch.ones_like(u1), u1, u2, u1 * u1, u2 * u2, u1 * u2], 1)

    def fields(self, z: torch.Tensor | None = None, B: int = 1, kappa: float = 1.0) -> torch.Tensor | None:
        """h (B, L, q), or None for the field-free model. kappa scales the z-dependent part only."""
        if self.field == "none":
            return None
        phi = self.features(z, B)
        if kappa != 1.0:
            phi = torch.cat([phi[:, :1], kappa * phi[:, 1:]], 1)
        return torch.einsum("bf,flq->blq", phi, self.W)

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

    def neg_log_pl(self, Z: torch.Tensor, w: torch.Tensor, lambd: float, z: torch.Tensor | None = None,
                   lambd_h: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
        """Weighted negative log pseudo-likelihood + L2 on J (and on the field weights W).

        Z (B, L) int, w (B,) weights summing to 1 inside the batch, z (B, 2) conditioning
        coordinates for field='quad'. Returns (loss, nlpl) with
        nlpl = -sum_m w_m sum_i log p(x_i^m | x_-i^m [, z^m]).
        """
        J = self.couplings()
        X = nn.functional.one_hot(Z, self.q).float()
        logits = self.logits_from_J(self.flat(J), X)
        h = self.fields(z, len(Z))
        if h is not None:
            logits = logits + h
        logp = torch.log_softmax(logits, dim=-1)
        ll = (logp * X).sum((1, 2))                      # (B,)
        nlpl = -(w * ll).sum()
        loss = nlpl + lambd * J.pow(2).sum()
        if h is not None:
            loss = loss + lambd_h * self.W.pow(2).sum()
        return loss, nlpl
