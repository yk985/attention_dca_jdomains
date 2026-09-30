"""Decorrelation study + generation with the PL sampler at inverse temperature beta.

python scripts/sample_model.py --tag main --beta 1.0
 1. mixing study: chains started from natural training sequences and from uniform random
    sequences; energy / PLL / Hamming-to-start / gap fraction recorded every sweep;
 2. integrated autocorrelation time of the energy at stationarity (second half of the
    natural-start chains, collapsed chains excluded);
 3. generation: n_gen chains started from random training sequences, run for
    max(min_burn, 20 tau_int) sweeps; the final state of each chain is one sample.
    Samples with > 90% gaps (the filter applied to the natural MSA) are counted and removed.
"""
import argparse
import json

import numpy as np
import torch

from common import OUT, add_split_arg, resolve
from attdca.data import load_msa, write_fasta, Q, GAP
from attdca.sampler import sample
from attdca.metrics import autocorr_time

p = add_split_arg(argparse.ArgumentParser())
p.add_argument("--tag", default="main")
p.add_argument("--beta", type=float, default=1.0)
p.add_argument("--n_study", type=int, default=1000)
p.add_argument("--study_sweeps", type=int, default=2000)
p.add_argument("--n_gen", type=int, default=10000)
p.add_argument("--min_burn", type=int, default=500)
p.add_argument("--max_gap_frac", type=float, default=0.9)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
name = f"{a.tag}_b{a.beta:g}"

J = torch.as_tensor(np.load(OUT / f"J_{a.tag}.npy"))
L = J.shape[0]
rng = np.random.default_rng(a.seed)
TRAIN, _ = resolve(a.split)
Ztr = load_msa(str(TRAIN))

# ---- 1. mixing study --------------------------------------------------------------
study = {}
for kind, init in [("nat", Ztr[rng.choice(len(Ztr), a.n_study, replace=False)]),
                   ("rnd", rng.integers(0, Q, (a.n_study, L)))]:
    r = sample(J, init, a.study_sweeps, beta=a.beta, seed=a.seed + 1)
    study[kind] = r
    print(f"beta={a.beta:g} {kind} start: E {r['energy'][0].mean():8.2f} -> {r['energy'][-1].mean():8.2f}  "
          f"d(x_t,x_0) {r['ham_init'][-1].mean():.3f}  collapsed {(r['gap'][-1] > a.max_gap_frac).mean():.3f}")

# ---- 2. autocorrelation at stationarity --------------------------------------------
T = len(study["nat"]["sweeps"])
ok = ~(study["nat"]["gap"] > a.max_gap_frac).any(0)          # chains never collapsed
acf, tau = autocorr_time(study["nat"]["energy"][T // 2:, ok])
acf_p, tau_p = autocorr_time(study["nat"]["pll"][T // 2:, ok])
print(f"tau_int(E) = {tau:.1f} sweeps, tau_int(PLL) = {tau_p:.1f} sweeps ({ok.sum()} chains)")

# ---- 3. generation ----------------------------------------------------------------
n_burn = int(max(a.min_burn, np.ceil(20 * max(tau, tau_p))))
gen = sample(J, Ztr[rng.choice(len(Ztr), a.n_gen)], n_burn, beta=a.beta, record_every=10, seed=a.seed + 2)
collapsed = (gen["Z"] == GAP).mean(1) > a.max_gap_frac
Zg = gen["Z"][~collapsed]
write_fasta(str(OUT / f"generated_{name}.fasta"), Zg, prefix=f"attdca_{name}")
np.save(OUT / f"generated_{name}.npy", Zg)
print(f"generated {a.n_gen} chains x {n_burn} sweeps; {collapsed.sum()} collapsed removed -> {len(Zg)}")

np.savez_compressed(
    OUT / f"sampling_{name}.npz", sweeps=study["nat"]["sweeps"], acf_E=acf, acf_P=acf_p,
    **{f"{k}_{kind}": study[kind][v] for kind in ("nat", "rnd")
       for k, v in [("E", "energy"), ("P", "pll"), ("H", "ham_init"), ("G", "gap")]})
json.dump({"beta": a.beta, "tau_int_energy": tau, "tau_int_pll": tau_p, "n_tau_chains": int(ok.sum()),
           "n_burn": n_burn, "n_gen_chains": a.n_gen, "n_collapsed": int(collapsed.sum()),
           "n_generated": int(len(Zg)), "n_study": a.n_study, "study_sweeps": a.study_sweeps,
           "study_collapsed_nat": float((study["nat"]["gap"][-1] > a.max_gap_frac).mean()),
           "study_collapsed_rnd": float((study["rnd"]["gap"][-1] > a.max_gap_frac).mean())},
          open(OUT / f"sampling_{name}.json", "w"), indent=1)
