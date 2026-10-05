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


def _toy_restraint(L=3, q=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    Jab = torch.randn(L, L, q, q, generator=g)
    J = 0.5 * (Jab + Jab.permute(1, 0, 3, 2))          # symmetric -> exact Boltzmann target
    J[torch.arange(L), torch.arange(L)] = 0
    h = torch.randn(L, q, generator=g)
    V = torch.randn(2, L, q, generator=g)
    return J, h, dict(V=V, m=torch.tensor([0.3, -0.2]), t=torch.tensor([1.0, -0.5]), lam=1.5)


def _all_states(L, q):
    return np.array(np.meshgrid(*[np.arange(q)] * L, indexing="ij")).reshape(L, -1).T


def test_restraint_sampler_targets_restrained_boltzmann():
    """Fields + harmonic PCA restraint: the empirical distribution of many chains must match
    exp(-E(x) - lam/2 |PC(x) - t|^2) enumerated over all q^L states."""
    L, q = 3, 3
    J, h, R = _toy_restraint(L, q)
    S = _all_states(L, q)
    X = torch.nn.functional.one_hot(torch.as_tensor(S), q).float()
    E = energy(AttentionPotts.flat(J), X, h.expand(len(S), L, q))
    pc = R["V"][:, np.arange(L)[None, :], S].sum(-1).T - R["m"]
    p = torch.softmax(-(E + 0.5 * R["lam"] * (pc - R["t"]).pow(2).sum(1)), 0).numpy()
    out = sample(J, np.zeros((60000, L), dtype=np.int64), n_sweeps=30, device=DEV, h=h, restraint=R)
    idx = (out["Z"] * q ** np.arange(L)[::-1]).sum(1)
    emp = np.bincount(idx, minlength=q ** L) / len(idx)
    assert np.abs(emp - p).sum() < 0.03
    assert np.allclose(out["pc"][-1], R["V"][:, np.arange(L)[None, :], out["Z"]].sum(-1).T - R["m"], atol=1e-5)


def test_restraint_equals_closed_form_potts_term():
    """lam/2 sum_c (PC_c - t_c)^2 = -1/2 x^T dJ x - dh.x + const with
    dJ = -lam V^T V, dh = lam (t + m) V (the identity behind proposal C1)."""
    L, q = 3, 3
    _, _, R = _toy_restraint(L, q, seed=1)
    V, m, t, lam = R["V"].reshape(2, -1).double(), R["m"].double(), R["t"].double(), R["lam"]
    S = _all_states(L, q)
    X = torch.nn.functional.one_hot(torch.as_tensor(S), q).double().reshape(len(S), -1)
    direct = 0.5 * lam * (X @ V.T - m - t).pow(2).sum(1)
    dJ, dh = -lam * V.T @ V, lam * (t + m) @ V
    potts = -0.5 * ((X @ dJ) * X).sum(1) - X @ dh
    assert torch.allclose(direct - potts, (direct - potts)[0].expand(len(S)), atol=1e-10)


def test_field_models_pl_match_naive_loop():
    for field in ("const", "quad"):
        m = AttentionPotts(6, H=3, d=4, seed=0, field=field)
        with torch.no_grad():
            m.W.normal_()
            if field == "quad":
                m.cond_mu.copy_(torch.tensor([0.5, -1.0])); m.cond_sd.copy_(torch.tensor([2.0, 0.5]))
        rng = np.random.default_rng(0)
        Z = torch.as_tensor(rng.integers(0, 21, (4, m.L)))
        z = torch.as_tensor(rng.normal(size=(4, 2)), dtype=torch.float32)
        w = torch.full((4,), 0.25)
        _, nlpl = m.neg_log_pl(Z, w, 0.0, z=z)
        J = m.couplings().detach()
        ll = 0.0
        for b in range(4):
            u = ((z[b] - m.cond_mu) / m.cond_sd).tolist() if field == "quad" else None
            phi = [1.0] if field == "const" else [1, u[0], u[1], u[0] ** 2, u[1] ** 2, u[0] * u[1]]
            for i in range(m.L):
                lg = sum(phi[f] * m.W[f, i].detach() for f in range(len(phi)))
                lg = lg + sum(J[i, j, :, Z[b, j]] for j in range(m.L) if j != i)
                ll += 0.25 * torch.log_softmax(lg, 0)[Z[b, i]].item()
        assert abs(nlpl.item() + ll) < 1e-4


def test_linear_frame_matches_sklearn():
    from sklearn.decomposition import PCA
    from attdca.pca_frame import LinearFrame
    Z = np.random.default_rng(0).integers(0, 21, (200, 8))
    fr = LinearFrame.fit(Z)
    X = np.eye(21)[Z].reshape(len(Z), -1)
    ref = PCA(2).fit(X).transform(X)
    assert np.allclose(np.abs(fr.transform(Z)), np.abs(ref), atol=1e-4)
