"""Generation for the conditioning / field experiments (proposals C1, C3, M2).

python scripts/sample_conditional.py --method c1     # baseline model + harmonic PCA restraint, lambda grid
python scripts/sample_conditional.py --method c3     # conditional-field models at z = target
python scripts/sample_conditional.py --method m2     # field model, unconditional
python scripts/sample_conditional.py --method base   # baseline, unconditional (same protocol as m2)

Targets are the k-means centres of the natural test set in the PCA plane (the clusters of the
main report), so target k is "make sequences of cluster C_k". Every chain starts from a random
training sequence, i.e. typically from another cluster, so it has to travel.
Output: outputs/cond/<method>.npz with final states Z and, per chain, the target index and a
configuration label (e.g. 'lam4' for C1, 's0.5_k1' for C3: training noise 0.5, gain kappa 1).
"""
import argparse
import json

import numpy as np
import torch

from common import OUT, resolve, load_frame
from attdca.data import load_msa
from attdca.model import AttentionPotts
from attdca.sampler import sample

p = argparse.ArgumentParser()
p.add_argument("--method", choices=["c1", "c3", "m2", "base"], required=True)
p.add_argument("--lams", default="0,0.25,1,4,16,64", help="C1 restraint stiffnesses")
p.add_argument("--c3_cfgs", default="s0.5:c3p:1,s0.5:c3p:2,s0.5:c3p:4,s0.5:c3p:8,"
                                    "s0.25:c3p_s0.25:1,s1:c3p_s1.0:1",
               help="C3 configurations label:model_tag:kappa (kappa = gain on the z-dependent field)")
p.add_argument("--chains", type=int, default=3000, help="chains per configuration and target")
p.add_argument("--n_uncond", type=int, default=10000, help="chains for unconditional runs")
p.add_argument("--sweeps", type=int, default=500)
p.add_argument("--record_every", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--base_tag", default="basep", help="field-free model used by c1 and base")
p.add_argument("--m2_tag", default="m2p")
p.add_argument("--out", default=None, help="output name (default: the method)")
a = p.parse_args()
OUTC = OUT / "cond"
OUTC.mkdir(exist_ok=True)

Ztr = load_msa(str(resolve("clean")[0]))
fr = load_frame("clean")
centres = np.array(json.load(open(OUT / "metrics_clean.json"))["pca_clusters"]["centres"])
K = len(centres)
rng = np.random.default_rng(a.seed)
kw = dict(n_sweeps=a.sweeps, record_every=a.record_every)


def load(tag):
    ck = torch.load(OUT / f"model_{tag}.pt")
    cfg = ck["cfg"]
    m = AttentionPotts(ck["L"], H=cfg["H"], d=cfg["d"], field=cfg.get("field", "none")).cuda()
    m.load_state_dict(ck["state"])
    return m.eval()


def track(t):
    """lam = 0 restraint: does not bias anything, only records the PCA trajectory."""
    return dict(V=fr.V, m=fr.m, t=t, lam=0.0)


runs = []        # (label, target index per chain, sampler output)
if a.method in ("c1", "c3"):
    tgt = np.repeat(np.arange(K), a.chains)
    t = centres[tgt]
    if a.method == "c1":
        J = torch.as_tensor(np.load(OUT / f"J_{a.base_tag}.npy"))
        cfgs = [(f"lam{float(v):g}", float(v)) for v in a.lams.split(",")]
    else:
        cfgs = [tuple(c.split(":")) for c in a.c3_cfgs.split(",")]
    for n, cfg in enumerate(cfgs):
        init = Ztr[rng.choice(len(Ztr), len(t))]
        if a.method == "c1":
            label, lam = cfg
            r = sample(J, init, restraint=dict(V=fr.V, m=fr.m, t=t, lam=lam), seed=a.seed + 1 + n, **kw)
        else:
            s, tag, kappa = cfg
            label = f"{s}_k{float(kappa):g}"
            m = load(tag)
            with torch.no_grad():
                Jm = m.couplings().cpu()
                h = m.fields(torch.as_tensor(t, dtype=torch.float32, device="cuda"), len(t), kappa=float(kappa))
            r = sample(Jm, init, h=h, restraint=track(t), seed=a.seed + 1 + n, **kw)
        runs.append((label, tgt, r))
        print(f"{a.method} {label}: done", flush=True)
else:
    init = Ztr[rng.choice(len(Ztr), a.n_uncond)]
    if a.method == "base":
        J, h = torch.as_tensor(np.load(OUT / f"J_{a.base_tag}.npy")), None
    else:
        m = load(a.m2_tag)
        with torch.no_grad():
            J, h = m.couplings().cpu(), m.fields(None, 1)[0]
    r = sample(J, init, h=h, restraint=track(np.zeros(2)), seed=a.seed + 1, **kw)
    runs.append(("uncond", np.full(a.n_uncond, -1), r))

Z = np.concatenate([r["Z"] for _, _, r in runs])
assert np.allclose(fr.transform(Z), np.concatenate([r["pc"][-1] for _, _, r in runs]), atol=1e-3)
np.savez_compressed(
    OUTC / f"{a.out or a.method}.npz", Z=Z, centres=centres, sweeps=runs[0][2]["sweeps"],
    target=np.concatenate([tg for _, tg, _ in runs]),
    label=np.concatenate([[lb] * len(tg) for lb, tg, _ in runs]),
    pc=np.concatenate([r["pc"] for _, _, r in runs], 1).astype(np.float32),
    energy=np.concatenate([r["energy"] for _, _, r in runs], 1).astype(np.float32))
print(f"{a.method}: {len(Z)} chains x {a.sweeps} sweeps saved")
