"""Consistency checks: vectorised model/sampler agree with naive loops.

Run:  python -m pytest tests -q   (from TP4_AttDCA_remastered/)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attdca.data import compute_weights  # noqa: E402
from attdca.model import AttentionPotts  # noqa: E402
from attdca.sampler import energy, sample  # noqa: E402
from attdca.metrics import autocorr_time, freqs  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def small_model(L=7, H=3, d=4):
    torch.manual_seed(1)
    m = AttentionPotts(L, H=H, d=d, seed=1)
    with torch.no_grad():
        m.V.mul_(10)
    return m


def test_attention_row_stochastic_zero_diag():
    A = small_model().attention()
    assert torch.allclose(A.sum(-1), torch.ones_like(A.sum(-1)), atol=1e-6)
    assert torch.all(A.diagonal(dim1=1, dim2=2) == 0)


def test_couplings_match_definition():
    m = small_model()
    A, V, J = m.attention(), m.V, m.couplings()
    i, j, a, b = 2, 5, 3, 11
    assert torch.isclose(J[i, j, a, b], sum(A[h, i, j] * V[h, a, b] for h in range(m.H)), atol=1e-5)


def test_pl_matches_naive_loop():
    m = small_model()
    rng = np.random.default_rng(0)
    Z = torch.as_tensor(rng.integers(0, 21, (5, m.L)))
    w = torch.full((5,), 0.2)
    _, nlpl = m.neg_log_pl(Z, w, 0.0)
    J = m.couplings().detach()
    ref = 0.0
    for z in Z.numpy():
        for i in range(m.L):
            e = np.array([sum(J[i, j, a, z[j]].item() for j in range(m.L) if j != i) for a in range(21)])
            ref += 0.2 * (e[z[i]] - np.log(np.exp(e).sum()))
    assert abs(nlpl.item() + ref) < 1e-3


def test_energy_matches_naive():
    m = small_model()
    J = m.couplings().detach()
    z = np.random.default_rng(1).integers(0, 21, m.L)
    X = torch.nn.functional.one_hot(torch.as_tensor(z)[None], 21).float()
    ref = -0.5 * sum(J[i, j, z[i], z[j]].item() for i in range(m.L) for j in range(m.L) if i != j)
    assert abs(energy(AttentionPotts.flat(J), X).item() - ref) < 1e-3


def test_sampler_single_site_conditional():
    """With symmetric couplings the PL sampler is an exact Gibbs sampler: for L=2 the
    empirical joint of many chains must match the Boltzmann distribution exp(J_01(a,b))."""
    L, q = 2, 21
    torch.manual_seed(0)
    Jab = torch.randn(q, q) * 1.5
    J = torch.zeros(L, L, q, q)
    J[0, 1] = Jab
    J[1, 0] = Jab.T                                   # symmetric -> exact Boltzmann target
    out = sample(J, np.zeros((40000, L), dtype=np.int64), n_sweeps=20, device=DEV)
    emp = np.zeros((q, q))
    np.add.at(emp, (out["Z"][:, 0], out["Z"][:, 1]), 1)
    emp /= emp.sum()
    p = torch.softmax(Jab.flatten(), 0).reshape(q, q).numpy()
    assert np.abs(emp - p).sum() < 0.08              # total-variation-ish tolerance


def test_weights_simple():
    Z = np.array([[0] * 10, [0] * 10, [1] * 10])
    w, meff, _ = compute_weights(Z, theta=0.2, device=DEV)
    assert np.allclose(w, [0.5, 0.5, 1.0]) and abs(meff - 2) < 1e-9


def test_cluster_and_split_are_separated():
    """Three well-separated groups must give three clusters, and the split must leave no
    test sequence within the threshold of any training sequence."""
    from attdca.split import greedy_cluster, split_by_cluster
    from attdca.metrics import min_hamming
    rng = np.random.default_rng(0)
    L, thresh = 40, 0.25
    centres = rng.integers(0, 21, (3, L))
    Z = np.repeat(centres, 40, axis=0)
    flip = rng.random(Z.shape) < 0.05                      # ~2 mutations per sequence
    Z[flip] = rng.integers(0, 21, flip.sum())
    lab = greedy_cluster(Z, thresh=thresh, device=DEV, verbose=False)
    assert len(np.unique(lab)) == 3
    itr, ite, st = split_by_cluster(Z, lab, test_frac=0.34, thresh=thresh, seed=1, device=DEV)
    assert len(ite) > 0 and not set(itr) & set(ite)
    assert min_hamming(Z[ite], Z[itr], device=DEV).min() >= thresh


def test_freqs_and_autocorr():
    Z = np.array([[0, 1], [0, 2]])
    f1, f2 = freqs(Z, device=DEV)
    assert np.isclose(f1[0, 0], 1) and np.isclose(f1[1, 1], 0.5) and np.isclose(f2[0, 1, 0, 2], 0.5)
    x = np.random.default_rng(0).standard_normal((4000, 50))
    _, tau = autocorr_time(x)
    assert 0.7 < tau < 1.4
