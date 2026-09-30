"""Compare generated sequences with the natural test set; write figures + metrics json.

python scripts/evaluate.py --tag main --betas 1.0,1.2
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA

from common import CONTACTS, FIG, OUT, add_split_arg, resolve
from attdca.data import load_msa, compute_weights, one_hot, Q
from attdca.metrics import (freqs, connected_corr, compare, min_hamming, pairwise_hamming,
                            contact_ppv)
from attdca.model import AttentionPotts
from attdca.sampler import scores

p = add_split_arg(argparse.ArgumentParser())
p.add_argument("--tag", default="main")
p.add_argument("--betas", default="1.0")
p.add_argument("--k_clusters", type=int, default=3, help="k-means clusters in PCA space")
p.add_argument("--cover", type=float, default=0.6,
               help="cluster is covered if (gen share / natural share) >= cover for every beta")
a = p.parse_args()
TRAIN, TEST = resolve(a.split)
SUF = "" if a.split == "orig" else f"_{a.split}"     # figure-name suffix
betas = [float(b) for b in a.betas.split(",")]
G = [f"b{b:g}" for b in betas]                       # generated-set keys

# palette (dataviz reference instance, light mode): test blue, beta=1 orange, beta* aqua, refs grey
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
GCOL = dict(zip(G, [ORANGE, AQUA]))
GCMAP = dict(zip(G, ["Oranges", "Greens"]))
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.3,
    "axes.edgecolor": "#b8b7b2", "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.dpi": 200, "font.family": "DejaVu Sans",
})
LAB = {"test": "natural test", "train": "natural train", "prof": "independent-site",
       **{g: rf"generated $\beta$={b:g}" for g, b in zip(G, betas)}}

rng = np.random.default_rng(0)
Ztr, Zte = load_msa(str(TRAIN)), load_msa(str(TEST))
L = Zte.shape[1]
Zs = {"test": Zte, "train": Ztr}
for g in G:
    Zs[g] = np.load(OUT / f"generated_{a.tag}_{g}.npy")
J = np.load(OUT / f"J_{a.tag}.npy")
wtr, meff_tr, th_tr = compute_weights(Ztr)
wte, meff_te, th_te = compute_weights(Zte)
d_te_tr = min_hamming(Zte, Ztr)
novel = d_te_tr > 0                                      # test sequences absent from train
Zs["test_novel"] = Zte[novel]
wnov, meff_nov, _ = compute_weights(Zte[novel], theta=th_te)
M = {"sizes": {"train": len(Ztr), "test": len(Zte), "test_novel": int(novel.sum()), "L": L,
               "meff_train": meff_tr, "meff_test": meff_te, "meff_test_novel": meff_nov,
               "theta_train": th_tr, "theta_test": th_te, **{f"gen_{g}": len(Zs[g]) for g in G}}}

# held-out pseudo-likelihood (generalisation)
ck = torch.load(OUT / f"model_{a.tag}.pt")
model = AttentionPotts(L, H=ck["cfg"]["H"], d=ck["cfg"]["d"]).cuda()
model.load_state_dict(ck["state"])
with torch.no_grad():
    for k, z, w in [("train", Ztr, wtr), ("test", Zte, wte), ("test_novel", Zte[novel], wnov)]:
        wt = torch.as_tensor(w / w.sum(), dtype=torch.float32, device="cuda")
        M.setdefault("nlpl_per_seq", {})[k] = model.neg_log_pl(torch.as_tensor(z, device="cuda"), wt, 0)[1].item()

# independent-site baseline: each column sampled from the weighted training frequencies
f1_tr, f2_tr = freqs(Ztr, wtr)
cum = np.cumsum(f1_tr, 1)
Zs["prof"] = (rng.random((10000, L, 1)) > cum[None]).sum(2).clip(0, Q - 1)

# ---- site frequencies and connected correlations -----------------------------------
F = {"test": freqs(Zte, wte), "train": (f1_tr, f2_tr), "test_novel": freqs(Zte[novel], wnov),
     "prof": freqs(Zs["prof"])}
for g in G:
    F[g] = freqs(Zs[g])
C = {k: connected_corr(*v) for k, v in F.items()}
top = np.abs(C["test"]) >= np.quantile(np.abs(C["test"]), 0.99)
M["f1"], M["Cij"], M["Cij_top1pct"], M["f1_vs_novel"], M["Cij_vs_novel"] = {}, {}, {}, {}, {}
for k in G + ["train", "prof"]:
    M["f1"][k] = compare(F["test"][0].ravel(), F[k][0].ravel())
    M["Cij"][k] = compare(C["test"], C[k])
    M["Cij_top1pct"][k] = compare(C["test"][top], C[k][top])
    M["f1_vs_novel"][k] = compare(F["test_novel"][0].ravel(), F[k][0].ravel())
    M["Cij_vs_novel"][k] = compare(C["test_novel"], C[k])
M["Cij_vs_novel"]["test_full"] = compare(C["test_novel"], C["test"])

# Reliability of the natural C_ij: correlate two independent halves of the test set and
# apply Spearman-Brown. Noise in the predictor attenuates any least-squares slope by this
# factor (regression dilution), so it sets the scale the fitted slopes should be read against.
_p = rng.permutation(len(Zte))
_h1, _h2 = _p[:len(_p) // 2], _p[len(_p) // 2:]
_r_hh = float(np.corrcoef(connected_corr(*freqs(Zte[_h1], wte[_h1])),
                          connected_corr(*freqs(Zte[_h2], wte[_h2])))[0, 1])
_rel = 2 * _r_hh / (1 + _r_hh)
M["Cij_reliability"] = {"half_half_r": _r_hh, "reliability": _rel,
                        "slope_disattenuated": {k: M["Cij"][k]["slope"] / _rel
                                                for k in G + ["train"]}}


def corr_figure(Fd, Cd, Md, fname, subtitle=""):
    """Scatter of generated vs natural f_i and C_ij (subsampled, top-1% highlighted)."""
    fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.9))
    ax[0].plot([0, 1], [0, 1], color=INK, lw=0.9, ls="--", zorder=1)
    for g in G:
        ax[0].scatter(Fd["test"][0].ravel(), Fd[g][0].ravel(), s=11, facecolor="none",
                      edgecolor=GCOL[g], linewidth=0.7, alpha=0.85, zorder=2,
                      label=LAB[g] + rf"  $r$={Md['f1'][g]['pearson']:.3f}")
    ax[0].set(xlabel=r"$f_i(a)$ natural", ylabel=r"$f_i(a)$ generated",
              title="One-site frequencies" + subtitle, xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax[0].legend(loc="upper left", handletextpad=0.1, borderpad=0.1)

    hi_q = np.quantile(np.abs(Cd["test"]), 0.99)
    lim = float(np.quantile(np.abs(np.concatenate([Cd["test"]] + [Cd[g] for g in G])), 0.9999))
    sub = rng.permutation(len(Cd["test"]))[:50000]          # thin the bulk, keep every strong pair
    for axx, g in zip(ax[1:], G):
        big = np.abs(Cd["test"]) >= hi_q
        axx.scatter(Cd["test"][sub], Cd[g][sub], s=2.5, color=GREY, alpha=0.30, lw=0, zorder=1,
                    rasterized=True, label="all pairs (thinned)")
        axx.scatter(Cd["test"][big], Cd[g][big], s=6, color=GCOL[g], alpha=0.75, lw=0, zorder=2,
                    rasterized=True, label=r"strongest 1% of $|C^{\rm nat}|$")
        axx.plot([-lim, lim], [-lim, lim], color=INK, lw=0.9, ls="--", zorder=3)
        s = Md["Cij"][g]
        axx.set(xlabel=r"$C_{ij}(a,b)$ natural", title=r"$C_{ij}$, " + LAB[g] + subtitle,
                xlim=(-lim, lim), ylim=(-lim, lim))
        axx.text(0.035, 0.965, rf"$r$ = {s['pearson']:.3f}" + "\n" + rf"slope = {s['slope']:.2f}",
                 va="top", transform=axx.transAxes, color=INK, fontsize=8)
    ax[1].set_ylabel(r"$C_{ij}(a,b)$ generated")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(*ax[2].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=7,
               markerscale=2.6, handletextpad=0.2)
    fig.savefig(FIG / fname); plt.close(fig)


corr_figure(F, C, M, f"freqs_corr{SUF}.pdf")

# ---- PCA ---------------------------------------------------------------------------
oh = lambda z: one_hot(z).reshape(len(z), -1).numpy()
pca = PCA(n_components=2).fit(oh(Zte))
P = {k: pca.transform(oh(Zs[k])) for k in ["test", "train", "prof"] + G}
ev = pca.explained_variance_ratio_
M["pca_explained_var"] = ev.tolist()
M["pca_wasserstein"] = {k: [float(stats.wasserstein_distance(P["test"][:, c], P[k][:, c], u_weights=wte))
                            for c in (0, 1)] for k in G + ["train", "prof"]}
M["pca_wasserstein_unweighted"] = {k: [float(stats.wasserstein_distance(P["test"][:, c], P[k][:, c]))
                                       for c in (0, 1)] for k in G + ["train", "prof"]}
# ---- PCA clusters: which modes of the family do the samples actually cover? ---------
from sklearn.cluster import KMeans                                          # noqa: E402

km = KMeans(a.k_clusters, n_init=20, random_state=0).fit(P["test"][:, :2])
CL = {k: km.predict(P[k][:, :2]) for k in P}
wof = {"test": wte, "train": wtr}
share = {k: np.array([(wof.get(k, np.ones(len(CL[k])))[CL[k] == c]).sum() /
                      wof.get(k, np.ones(len(CL[k]))).sum() for c in range(a.k_clusters)])
         for k in P}
ratio = {g: share[g] / np.maximum(share["test"], 1e-12) for g in G}   # gen share / natural share
covered = sorted(c for c in range(a.k_clusters)
                 if min(ratio[g][c] for g in G) >= a.cover)
M["pca_clusters"] = {"k": a.k_clusters, "centres": km.cluster_centers_.tolist(),
                     "share": {k: v.tolist() for k, v in share.items()},
                     "ratio": {g: ratio[g].tolist() for g in G},
                     "covered": covered, "cover_rule": a.cover,
                     "covered_natural_weight": float(share["test"][covered].sum())}
missed = [c for c in range(a.k_clusters) if c not in covered]
M["pca_minor_cluster"] = {k: float(share[k][missed].sum()) for k in ["test", "train"] + G}
M["pca_clusters"]["missed"] = missed
print("PCA cluster weight shares:", {k: np.round(v, 3).tolist() for k, v in share.items()})
print("covered clusters:", covered, "holding", round(float(share['test'][covered].sum()), 3),
      "of the natural test weight")

# ---- is the missed cluster simply high in energy? -----------------------------------
# A cluster's stationary weight is set by its free energy F_k = <E>_k - S_k, not by <E>_k
# alone, so we report the mean energy of the *natural* sequences of each cluster (what the
# model thinks of real sequences there) next to the weight the sampler actually gives it.
# delta_F = -ln(share_gen / share_nat) is the model's free-energy error for that cluster.
_Jt = torch.as_tensor(J)
_Ete, _Pte = scores(_Jt, Zte)
_Etr, _Ptr = scores(_Jt, Ztr)
_Egen = {g: scores(_Jt, Zs[g])[0] for g in G}
_wm = lambda v, w: float((v * w).sum() / w.sum()) if w.sum() > 0 else float("nan")
clu = []
for c in range(a.k_clusters):
    mt, mr = CL["test"] == c, CL["train"] == c
    row = {"cluster": c, "covered": c in covered,
           "share_test": float(share["test"][c]), "share_train": float(share["train"][c]),
           "n_test": int(mt.sum()), "n_train": int(mr.sum()),
           "E_test": _wm(_Ete[mt], wte[mt]), "E_train": _wm(_Etr[mr], wtr[mr]),
           "E_test_unw": float(_Ete[mt].mean()) if mt.any() else float("nan"),
           "PLL_test": _wm(_Pte[mt], wte[mt]), "PLL_train": _wm(_Ptr[mr], wtr[mr])}
    for g in G:
        mg = CL[g] == c
        row[f"share_{g}"] = float(share[g][c])
        row[f"E_{g}"] = float(_Egen[g][mg].mean()) if mg.any() else float("nan")
        row[f"dF_{g}"] = float(-np.log(max(share[g][c], 1e-9) / max(share["test"][c], 1e-12)))
    clu.append(row)
M["cluster_energy"] = clu
for r_ in clu:
    print(f"  cluster {r_['cluster']} {'covered' if r_['covered'] else 'MISSED ':>8s}  "
          f"nat share {r_['share_test']:.3f} -> gen {r_['share_' + G[0]]:.3f}  "
          f"<E>_nat(test) {r_['E_test']:8.2f}  <E>_nat(train) {r_['E_train']:8.2f}  "
          f"<E>_gen {r_['E_' + G[0]]:8.2f}  dF {r_['dF_' + G[0]]:+.2f}")

# statistics restricted to the covered clusters (natural and generated alike)
inC = {k: np.isin(CL[k], covered) for k in P}
wte_c = wte[inC["test"]]
Fc = {"test": freqs(Zte[inC["test"]], wte_c), "train": freqs(Ztr[inC["train"]], wtr[inC["train"]]),
      "prof": freqs(Zs["prof"][inC["prof"]])}
for g in G:
    Fc[g] = freqs(Zs[g][inC[g]])
Cc = {k: connected_corr(*v) for k, v in Fc.items()}
Mc = {"f1": {}, "Cij": {}, "Cij_top1pct": {}}
topc = np.abs(Cc["test"]) >= np.quantile(np.abs(Cc["test"]), 0.99)
for k in G + ["train", "prof"]:
    Mc["f1"][k] = compare(Fc["test"][0].ravel(), Fc[k][0].ravel())
    Mc["Cij"][k] = compare(Cc["test"], Cc[k])
    Mc["Cij_top1pct"][k] = compare(Cc["test"][topc], Cc[k][topc])
M["restricted"] = {**Mc, "n": {k: int(inC[k].sum()) for k in P},
                   "meff_test": float(compute_weights(Zte[inC["test"]], theta=th_te)[1])}
corr_figure(Fc, Cc, Mc, f"freqs_corr_clusters{SUF}.pdf", subtitle=", covered clusters")

# ---- PCA scatter --------------------------------------------------------------------
ext = np.concatenate([P["test"]] + [P[g] for g in G])
lo, hi = ext.min(0) - 0.4, ext.max(0) + 0.4
keys = ["test"] + G
NSC = 4000                                        # points drawn per panel
fig, ax = plt.subplots(1, len(keys) + 1, figsize=(7.9, 2.5), sharex=True, sharey=True)
for axx, k in zip(ax, keys):
    # the natural panel is drawn by sampling with the reweighting weights, so the cloud
    # shown is the same distribution the statistics and the contours use
    pk = wte / wte.sum() if k == "test" else None
    idx = rng.choice(len(P[k]), min(NSC, len(P[k])), replace=False, p=pk)
    axx.scatter(*P[k][idx].T, s=3.5, color=BLUE if k == "test" else GCOL[k], alpha=0.35, lw=0,
                rasterized=True)
    axx.set_title(LAB[k] + (" (reweighted)" if k == "test" else "") + f", n={len(P[k])}", fontsize=8)
for c in range(a.k_clusters):
    col = INK if c in covered else "#c0392b"
    for axx in ax[:-1]:
        axx.plot(*km.cluster_centers_[c], marker="x", ms=6, mew=1.6, color=col, zorder=5)
    ax[0].annotate(f"C{c}" + ("" if c in covered else " (missed)"), km.cluster_centers_[c],
                   textcoords="offset points", xytext=(5, 5), fontsize=6.5, color=col, zorder=6)
xx, yy = np.mgrid[lo[0]:hi[0]:120j, lo[1]:hi[1]:120j]
PW = {"test": wte / wte.sum() * len(wte)}                 # natural test shown with its weights
for k in keys:
    c = BLUE if k == "test" else GCOL[k]
    kde = stats.gaussian_kde(P[k].T, weights=PW.get(k))(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax[-1].contour(xx, yy, kde, levels=np.quantile(kde, [0.8, 0.95, 0.99]), colors=c, linewidths=0.9)
    ax[-1].plot([], [], color=c, label=LAB[k] + (" (reweighted)" if k == "test" else ""))
ax[-1].set_title("density contours (80/95/99%)", fontsize=8)
for axx in ax:
    axx.set_xlabel(f"PC1 ({100 * ev[0]:.1f}%)")
ax[0].set_ylabel(f"PC2 ({100 * ev[1]:.1f}%)")
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.legend(*ax[-1].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7.5)
fig.savefig(FIG / f"pca{SUF}.pdf"); plt.close(fig)

# ---- Hamming distances -------------------------------------------------------------
H = {"test->train": d_te_tr, "test-test": pairwise_hamming(Zte),
     "test->test(nn)": min_hamming(Zte, Zte, exclude_self=True),
     "novel->train": d_te_tr[novel]}
for g in G:
    H[f"{g}->test"] = min_hamming(Zs[g], Zte)
    H[f"{g}->train"] = min_hamming(Zs[g], Ztr)
    H[f"{g}-{g}"] = pairwise_hamming(Zs[g])
    H[f"{g}-test"] = pairwise_hamming(Zs[g], Zte)
    H[f"{g}->{g}(nn)"] = min_hamming(Zs[g], Zs[g], exclude_self=True)
    M.setdefault("unique_frac", {})[g] = len(np.unique(Zs[g], axis=0)) / len(Zs[g])
M["hamming"] = {k: {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std()),
                    "q05": float(np.quantile(v, 0.05)), "frac_zero": float((v == 0).mean())}
                for k, v in H.items()}
bins = (np.arange(L + 2) - 0.5) / L              # one bin per k/L value
fig, ax = plt.subplots(1, 3, figsize=(7.8, 2.9))


def dhist(axx, key, color, label, fill=False, ls="-"):
    """Density of a distance set, with a tick at its mean on the top axis."""
    v = H[key]
    axx.hist(v, bins=bins, density=True, histtype="stepfilled" if fill else "step",
             lw=0 if fill else 1.5, ls=ls, color=color, label=label,
             facecolor=color + "33" if fill else "none", edgecolor=color)
    if fill:                                     # redraw the outline over the fill
        axx.hist(v, bins=bins, density=True, histtype="step", lw=1.5, color=color)
    axx.plot(v.mean(), 0, marker="v", ms=4.5, color=color, clip_on=False, zorder=6)


# panel 1 - how close is each sequence to the natural TEST set
for g in G:
    dhist(ax[0], f"{g}->test", GCOL[g], LAB[g] + r" $\to$ test")
dhist(ax[0], "test->test(nn)", BLUE, r"test $\to$ other test", fill=True)
# panel 2 - how close is each sequence to the TRAINING set (memorisation check)
for g in G:
    dhist(ax[1], f"{g}->train", GCOL[g], LAB[g] + r" $\to$ train")
dhist(ax[1], "test->train", BLUE, r"test $\to$ train", fill=True)
# panel 3 - spread inside each set
for g in G:
    dhist(ax[2], f"{g}-{g}", GCOL[g], LAB[g] + ", within set")
dhist(ax[2], "test-test", BLUE, "test, within set", fill=True)
dhist(ax[2], f"{G[0]}-test", GREY, "generated vs test", ls=":")

ax[0].set(title="(a) to nearest test sequence", xlabel="normalised Hamming distance",
          ylabel="density")
ax[1].set(title="(b) to nearest training sequence", xlabel="normalised Hamming distance")
ax[2].set(title="(c) all pairs within a set", xlabel="normalised Hamming distance")
for axx in ax:
    axx.legend(loc="upper left", fontsize=6.4, borderpad=0.2, labelspacing=0.25)
    axx.set_ylim(bottom=0, top=axx.get_ylim()[1] * 1.45)
    axx.set_xlim(0, 1)
fig.tight_layout(); fig.savefig(FIG / f"hamming{SUF}.pdf"); plt.close(fig)

# ---- energy / PLL distributions ----------------------------------------------------
Jt = torch.as_tensor(J)
EP = {k: scores(Jt, Zs[k]) for k in ["test", "train", "prof"] + G}
EP["random"] = scores(Jt, rng.integers(0, Q, (5000, L)))
wmean = lambda v, w: float((v * w).sum() / w.sum())
M["energy"] = {k: {"mean": float(e.mean()), "std": float(e.std()), "pll_mean": float(pl.mean()),
                   "pll_std": float(pl.std())} for k, (e, pl) in EP.items()}
M["energy"]["test"]["mean_w"] = wmean(EP["test"][0], wte)
M["energy"]["train"]["mean_w"] = wmean(EP["train"][0], wtr)
M["energy"]["test"]["pll_mean_w"] = wmean(EP["test"][1], wte)
M["wasserstein_energy"] = {k: float(stats.wasserstein_distance(EP["test"][0], EP[k][0], u_weights=wte,
                                                               v_weights=wtr if k == "train" else None))
                           for k in G + ["train", "prof"]}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
for j, xl in enumerate([r"energy $E(x)$", r"$\log\mathrm{PL}(x)/L$"]):
    for k in ["test", "prof"] + G:
        v = EP[k][j] / (L if j == 1 else 1)
        c = BLUE if k == "test" else GREY if k == "prof" else GCOL[k]
        ax[j].hist(v, bins=70, density=True, histtype="step", lw=1.3, color=c, label=LAB[k],
                   weights=wte if k == "test" else None)
    ax[j].set(xlabel=xl)
ax[0].set_ylabel("density")
ax[0].set_title("Energy (symmetrised Hamiltonian)"); ax[1].set_title("Pseudo-log-likelihood per site")
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7.5)
fig.savefig(FIG / f"energy_dist{SUF}.pdf"); plt.close(fig)

# ---- mixing / decorrelation --------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.0))
ax = ax.ravel()
M["sampling"] = {}
for g in G:
    S = np.load(OUT / f"sampling_{a.tag}_{g}.npz")
    sj = json.load(open(OUT / f"sampling_{a.tag}_{g}.json"))
    t = S["sweeps"] + 1
    c = GCOL[g]
    for kind, ls, lab in [("nat", "-", "train start"), ("rnd", ":", "random start")]:
        E, Hd, Gp = S[f"E_{kind}"], S[f"H_{kind}"], S[f"G_{kind}"]
        ax[0].plot(t, np.median(E, 1), color=c, ls=ls, lw=1.4)
        if kind == "nat":
            ax[0].fill_between(t, *np.quantile(E, [0.1, 0.9], axis=1), color=c, alpha=0.12, lw=0)
            ax[1].fill_between(t, *np.quantile(Hd, [0.1, 0.9], axis=1), color=c, alpha=0.12, lw=0)
        ax[1].plot(t, Hd.mean(1), color=c, ls=ls, lw=1.4)
    if g == G[0]:
        for i in range(2):
            ax[2].plot(t[:301] - 1, S["E_nat"][:301, i], color=c, lw=0.7, alpha=0.9)
            ax[2].plot(t[:301] - 1, S["E_rnd"][:301, i], color=c, lw=0.7, ls=":", alpha=0.9)
    n = min(len(S["acf_E"]), 80)
    ax[3].plot(np.arange(n), S["acf_E"][:n], color=c, lw=1.4)
    yl = 0.9 - 0.1 * G.index(g)
    ax[3].text(0.42, yl, rf"$\beta$={sj['beta']:g}: $\tau_{{int}}$ = {sj['tau_int_energy']:.1f} sweeps",
               color=INK, ha="left", va="center", transform=ax[3].transAxes, fontsize=7.5)
    ax[3].plot([0.30, 0.39], [yl, yl], transform=ax[3].transAxes, color=c, lw=1.6)
    hm = S["H_nat"].mean(1)
    sj["sweeps_to_90pct_ham"] = int(S["sweeps"][np.argmax(hm >= 0.9 * hm[-1])])
    sj["ham_end_nat"] = float(hm[-1])
    sj["E_med_end_nat"] = float(np.median(S["E_nat"][-1]))
    sj["E_med_end_rnd"] = float(np.median(S["E_rnd"][-1]))
    sj["E_start_nat"] = float(S["E_nat"][0].mean())
    M["sampling"][g] = sj
ax[0].axhline(M["energy"]["test"]["mean_w"], color=BLUE, ls="--", lw=0.9)
ax[0].set(xscale="log", xlabel="sweeps + 1", ylabel=r"median $E(x_t)$", title="Energy along the chains")
ax[1].axhline(M["hamming"]["test-test"]["mean"], color=BLUE, ls="--", lw=0.9)
ax[1].text(1.2, M["hamming"]["test-test"]["mean"] + 0.02, "mean test–test distance", color=MUTED, fontsize=7)
ax[1].set(xscale="log", xlabel="sweeps + 1", ylabel=r"$d_H(x_t, x_0)/L$", title="Distance from the start", ylim=(0, 1))
ax[2].axhline(M["energy"]["test"]["mean_w"], color=BLUE, ls="--", lw=0.9)
ax[2].set(xlabel="sweeps", ylabel=r"$E(x_t)$", title=r"Single chains ($\beta$=1, first 300 sweeps)", ylim=(-185, -60))
ax[3].axhline(0, color=GREY, lw=0.6)
ax[3].set(xlabel="lag (sweeps)", ylabel=r"$\rho_E$(lag)", title="Energy autocorrelation at stationarity")
from matplotlib.lines import Line2D
hand = [Line2D([], [], color=GCOL[g], lw=1.6, label=LAB[g]) for g in G] + [
    Line2D([], [], color=INK, lw=1.2, ls="-", label="start: training sequence"),
    Line2D([], [], color=INK, lw=1.2, ls=":", label="start: uniform random"),
    Line2D([], [], color=BLUE, lw=0.9, ls="--", label="natural test (mean)")]
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.legend(handles=hand, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, 0.0))
fig.savefig(FIG / f"mixing{SUF}.pdf"); plt.close(fig)

# ---- beta scan ---------------------------------------------------------------------
bs_path = OUT / f"beta_scan_{a.tag}.json"
if bs_path.exists():
    bs = json.load(open(bs_path))
    M["beta_scan"] = [{k: v for k, v in r.items() if not k.endswith("trace")} for r in bs["scan"]]
    rows = [r for r in bs["scan"] if r["init"] == "train"]
    b = np.array([r["beta"] for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.3))
    ax[0].plot(b, [r["E_mean"] for r in rows], "o-", color=ORANGE, ms=3.5, lw=1.3)
    ax[0].axhline(M["energy"]["train"]["mean_w"], color=BLUE, ls="--", lw=0.9, label="natural (train)")
    ax[0].set(xlabel=r"$\beta$", ylabel=r"$\langle E\rangle$ generated", title="Mean energy"); ax[0].legend(fontsize=6.5)
    ax[1].plot(b, [r["r_C"] for r in rows], "o-", color=ORANGE, ms=3.5, lw=1.3, label=r"Pearson $r$")
    ax[1].plot(b, [r["slope_C"] for r in rows], "s--", color=GREY, ms=3.5, lw=1.1, label="slope")
    ax[1].set(xlabel=r"$\beta$", title=r"$C_{ij}$ vs test"); ax[1].legend(fontsize=6.5)
    ax[2].plot(b, [r["gen_gen"] for r in rows], "o-", color=ORANGE, ms=3.5, lw=1.3, label="gen–gen")
    ax[2].plot(b, [r["nn_train"] for r in rows], "s--", color=GREY, ms=3.5, lw=1.1, label="to nearest train")
    ax[2].axhline(bs["test"]["gen_gen"], color=BLUE, ls="--", lw=0.9, label="test–test")
    ax[2].set(xlabel=r"$\beta$", title="Mean Hamming distance"); ax[2].set_ylim(0.15, None); ax[2].legend(fontsize=6.3, loc="lower left")
    for axx in ax:
        for bb, g in zip(betas, G):
            axx.axvline(bb, color=GCOL[g], lw=0.6, alpha=0.6)
    fig.tight_layout(); fig.savefig(FIG / f"beta_scan{SUF}.pdf"); plt.close(fig)

# ---- training curve + contact sanity check ------------------------------------------
h = json.load(open(OUT / f"train_hist_{a.tag}.json"))
ppv, frac = contact_ppv(J, str(CONTACTS))
M["contacts"] = {"ppv_at_L": float(ppv[L - 1]), "ppv_at_L2": float(ppv[L // 2 - 1]),
                 "ppv_at_2L": float(ppv[2 * L - 1]), "base_rate": frac}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
ax[0].plot(h["epoch"], h["train_nlpl"], color=AQUA, lw=1.3, label="train")
ax[0].plot(h["epoch"], h["val_nlpl"], color=BLUE, lw=1.3, label="validation")
ax[0].set(xlabel="epoch", ylabel=r"$-\log$PL per sequence", title="Training", ylim=(70, 110))
ax[0].legend()
r = np.arange(1, len(ppv) + 1)
ax[1].plot(r, ppv, color=BLUE, lw=1.3, label="AttentionDCA couplings (APC)")
ax[1].axhline(frac, color=GREY, ls="--", lw=0.9, label="random pairs")
ax[1].axvline(L, color=GREY, lw=0.6, ls=":")
ax[1].set(xscale="log", xlabel="number of top-ranked pairs", ylabel="PPV", ylim=(0, 1.02),
          title=r"Contact prediction ($|i-j|>4$, $<8$ Å)")
ax[1].legend(loc=(0.02, 0.28))
fig.tight_layout(); fig.savefig(FIG / f"training_contacts{SUF}.pdf"); plt.close(fig)
M["train"] = {k: h[k] for k in ("best_val_nlpl", "cfg", "theta", "meff", "n_params")}
M["train"]["epochs_run"] = len(h["epoch"])
M["train"]["best_epoch"] = int(np.argmin(h["val_nlpl"]) + 1)

M["betas"] = betas
json.dump(M, open(OUT / f"metrics_{a.tag}.json", "w"), indent=1)
print(json.dumps({k: M[k] for k in ("sizes", "nlpl_per_seq", "f1", "Cij", "Cij_vs_novel", "unique_frac",
                                    "contacts", "sampling")}, indent=1))
