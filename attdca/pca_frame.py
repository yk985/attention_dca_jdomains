"""PCA of one-hot sequences written as the exact affine map it is.

    PC_c(x) = sum_{i,a} V[c, i, a] x_ia - m_c,     m_c = sum_{i,a} V[c, i, a] mu_ia

so a coordinate of a sequence is a sum of per-site contributions V[c, i, x_i], and changing
site i from a to b moves it by V[c, i, b] - V[c, i, a]. This is what the restraint sampler
(proposal C1) and the conditional model (C3) rely on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from .data import Q


@dataclass
class LinearFrame:
    V: np.ndarray        # (C, L, q) loadings
    m: np.ndarray        # (C,) offset
    evr: np.ndarray      # explained variance ratio

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """Integer sequences (M, L) -> coordinates (M, C), by summing per-site contributions."""
        L = Z.shape[1]
        return self.V[:, np.arange(L)[None, :], Z].sum(-1).T - self.m

    @classmethod
    def fit(cls, Z: np.ndarray, n: int = 2) -> "LinearFrame":
        X = np.eye(Q, dtype=np.float32)[Z].reshape(len(Z), -1)
        p = PCA(n_components=n).fit(X)
        V = p.components_.astype(np.float64)
        return cls(V.reshape(n, Z.shape[1], Q), V @ p.mean_.astype(np.float64), p.explained_variance_ratio_)

    def save(self, path):
        np.savez(path, V=self.V, m=self.m, evr=self.evr)

    @classmethod
    def load(cls, path) -> "LinearFrame":
        d = np.load(path)
        return cls(d["V"], d["m"], d["evr"])
