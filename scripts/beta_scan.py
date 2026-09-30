"""Inverse-temperature scan of the PL sampler: fidelity vs diversity vs gap collapse.

python scripts/beta_scan.py --tag main --sweeps 1000
Chains start from training sequences (and, separately, uniform random sequences).
"""
import argparse
import json

import numpy as np
import torch

from common import OUT, add_split_arg, resolve
from attdca.data import load_msa, compute_weights, GAP, Q
from attdca.metrics import freqs, connected_corr, compare, pairwise_hamming, min_hamming
from attdca.sampler import sample, scores

p = add_split_arg(argparse.ArgumentParser())
p.add_argument("--tag", default="main")
p.add_argument("--betas", default="0.9,1.0,1.05,1.1,1.15,1.2,1.3,1.5")
p.add_argument("--n", type=int, default=3000)
p.add_argument("--sweeps", type=int, default=1000)
a = p.parse_args()

J = torch.as_tensor(np.load(OUT / f"J_{a.tag}.npy"))
TRAIN, TEST = resolve(a.split)
Ztr, Zte = load_msa(str(TRAIN)), load_msa(str(TEST))
L = Zte.shape[1]
wte, _, _ = compute_weights(Zte)
f1, f2 = freqs(Zte, wte)
Cte = connected_corr(f1, f2)
Ete, Pte = scores(J, Zte)
rng = np.random.default_rng(0)
res = {"test": {"E": float((Ete * wte).sum() / wte.sum()),
                "gen_gen": float(pairwise_hamming(Zte).mean())}, "scan": []}
for init in ("train", "random"):
    for b in map(float, a.betas.split(",")):
        Z0 = Ztr[rng.choice(len(Ztr), a.n)] if init == "train" else rng.integers(0, Q, (a.n, L))
        r = sample(J, Z0, a.sweeps, beta=b, record_every=10, seed=1)
        Z = r["Z"]
        collapsed = (Z == GAP).mean(1) > 0.9
        Zk = Z[~collapsed]
        g1, g2 = freqs(Zk)
        E, _ = scores(J, Zk)
        cf, cc = compare(f1.ravel(), g1.ravel()), compare(Cte, connected_corr(g1, g2))
        row = {"init": init, "beta": b, "collapsed_frac": float(collapsed.mean()),
               "gap_trace": r["gap"].mean(1).tolist(), "E_trace": r["energy"].mean(1).tolist(),
               "E_mean": float(E.mean()), "r_f1": cf["pearson"], "slope_f1": cf["slope"],
               "r_C": cc["pearson"], "slope_C": cc["slope"],
               "gen_gen": float(pairwise_hamming(Zk, n_max=2000).mean()),
               "nn_train": float(min_hamming(Zk, Ztr).mean()),
               "unique_frac": float(len(np.unique(Zk, axis=0)) / max(len(Zk), 1))}
        res["scan"].append(row)
        print(f"{init:6s} beta={b:4.2f} collapsed {row['collapsed_frac']:.3f}  E {row['E_mean']:7.1f} "
              f"(test {res['test']['E']:.1f})  r_f1 {row['r_f1']:.3f}  r_C {row['r_C']:.3f} "
              f"slope_C {row['slope_C']:.2f}  gen-gen {row['gen_gen']:.3f}  nn-train {row['nn_train']:.3f}",
              flush=True)
res["sweeps"] = a.sweeps
res["trace_sweeps"] = r["sweeps"].tolist()
json.dump(res, open(OUT / f"beta_scan_{a.tag}.json", "w"), indent=1)
