"""Evaluation of the conditioning (C1, C3) and field (M2) experiments -> outputs/metrics_cond.json + figures.

Conditional pools are judged against the natural *test* sequences of the target cluster
(weighted), with three references per target:
  natural ceiling   training sequences of the cluster vs test sequences of the cluster
  rejection (C2)    unconditional baseline samples that happen to fall in the cluster
  unconditional     the whole unconditional baseline pool
Energies of every pool are computed under one common judge, the field-free baseline model,
so that C1 (which uses it) and C3 (which has its own couplings) are on the same scale.
Every pool is evaluated at the same size N_EVAL (subsampled when larger; the natural ceiling is
a weight-proportional draw of N_EVAL training sequences of the cluster), because the sampling
noise of C_ij estimates depends strongly on pool size.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import OUT, FIG, resolve, load_frame  # noqa: E402
from attdca.data import load_msa, compute_weights, Q  # noqa: E402
from attdca.metrics import freqs, connected_corr, compare, min_hamming, pairwise_hamming  # noqa: E402
from attdca.model import AttentionPotts  # noqa: E402
from attdca.sampler import scores  # noqa: E402

BASE, M2 = "basep", "m2p"
N_EVAL = 3000
HPD = (28, 6, 12, 2)            # column of H; alphabet indices of H, P, D
OUTC = OUT / "cond"
BLUE, ORANGE, AQUA, GREY, INK, MUTED, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b",
                                                "#52514e", "#8a5cd6")
TCOL = ["#d6452a", "#2a78d6", "#1baf7a"]          # target / cluster colours C0, C1, C2
plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.3,
    "axes.edgecolor": "#b8b7b2", "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 200, "font.family": "DejaVu Sans",
})

Ztr, Zte = (load_msa(str(p)) for p in resolve("clean"))
L = Ztr.shape[1]
wtr, _, _ = compute_weights(Ztr)
wte, _, _ = compute_weights(Zte)
fr = load_frame("clean")
pools = {m: np.load(OUTC / f"{m}.npz") for m in ("c1", "c3", "base", "m2")}
centres = pools["c1"]["centres"]
K = len(centres)


def assign(P):
    return np.argmin(((P[:, None, :] - centres[None]) ** 2).sum(-1), 1)


P_te, P_tr = fr.transform(Zte), fr.transform(Ztr)
P_mean = wte @ P_te / wte.sum()          # weighted centre of the natural family
lab_te, lab_tr = assign(P_te), assign(P_tr)
J_judge = torch.as_tensor(np.load(OUT / f"J_{BASE}.npy"))
E_te, _ = scores(J_judge, Zte)
E_tr, _ = scores(J_judge, Ztr)
nat = {k: dict(f=freqs(Zte[lab_te == k], wte[lab_te == k])) for k in range(K)}
for k in range(K):
    nat[k]["C"] = connected_corr(*nat[k]["f"])


def hpd(Z):
    i, h, p_, d = HPD
    return float(((Z[:, i] == h) & (Z[:, i + 1] == p_) & (Z[:, i + 2] == d)).mean())


_rs = np.random.default_rng(1)


def quality(Z, k, w=None):
    """Statistics of a pool Z aimed at cluster k (w: weights of a natural pool -> weighted draw)."""
    if w is not None:
        Z, w = Z[_rs.choice(len(Z), N_EVAL, p=w / w.sum())], None
    elif len(Z) > N_EVAL:
        Z = Z[_rs.choice(len(Z), N_EVAL, replace=False)]
    f1, f2 = freqs(Z, w)
    P = fr.transform(Z)
    ww = np.full(len(Z), 1 / len(Z)) if w is None else w / w.sum()
    E, _ = scores(J_judge, Z)
    out = {"n": int(len(Z)),
           "dist": float(ww @ np.linalg.norm(P - centres[k], axis=1)),
           # signed miss along the direction from the family centre to the target: > 0 = overshoot
           "overshoot": float((ww @ P - centres[k]) @ (centres[k] - P_mean) / np.linalg.norm(centres[k] - P_mean)),
           "pc_mean": (ww @ P).tolist(),
           "hit": float(ww @ (assign(P) == k)),
           "f1": compare(nat[k]["f"][0].ravel(), f1.ravel()),
           "Cij": compare(nat[k]["C"], connected_corr(f1, f2)),
           "hpd": float(ww @ (((Z[:, 28] == 6) & (Z[:, 29] == 12) & (Z[:, 30] == 2)).astype(float))),
           "E_judge": float(ww @ E),
           "gap": float(ww @ (Z == Q - 1).mean(1)),
           "d_train": float(ww @ min_hamming(Z, Ztr)),
           "div": float(pairwise_hamming(Z, n_max=1000).mean())}
    return out


R = {"targets": centres.tolist(), "cond": {"c1": {}, "c3": {}}, "ref": {}}
# ---- references --------------------------------------------------------------------
Pb = fr.transform(pools["base"]["Z"])
lab_b = assign(Pb)
for k in range(K):
    R["ref"][k] = {
        "ceiling": quality(Ztr[lab_tr == k], k, wtr[lab_tr == k]),
        "natural_test": {"E_judge": float((wte[lab_te == k] @ E_te[lab_te == k]) / wte[lab_te == k].sum()),
                         "hpd": float((wte[lab_te == k] @ (((Zte[:, 28] == 6) & (Zte[:, 29] == 12) &
                                                             (Zte[:, 30] == 2))[lab_te == k])) / wte[lab_te == k].sum()),
                         "n": int((lab_te == k).sum())},
        "rejection": quality(pools["base"]["Z"][lab_b == k], k),
        "uncond": quality(pools["base"]["Z"], k),
    }
    print(f"target C{k}: ceiling r(Cij) {R['ref'][k]['ceiling']['Cij']['pearson']:.3f}  "
          f"rejection r(Cij) {R['ref'][k]['rejection']['Cij']['pearson']:.3f} (n={R['ref'][k]['rejection']['n']})")

# ---- conditional pools ---------------------------------------------------------------
for m in ("c1", "c3"):
    d = pools[m]
    for k in range(K):
        for v in dict.fromkeys(d["label"]):
            sel = (d["target"] == k) & (d["label"] == v)
            q_ = quality(d["Z"][sel], k)
            R["cond"][m].setdefault(str(k), {})[v] = q_
            print(f"{m} C{k} {v:>9}: dist {q_['dist']:.2f} over {q_['overshoot']:+.2f} hit {q_['hit']:.2f}  r(f) {q_['f1']['pearson']:.3f} "
                  f"r(C) {q_['Cij']['pearson']:.3f}  HPD {q_['hpd']:.2f}  E {q_['E_judge']:.1f}  "
                  f"d_tr {q_['d_train']:.2f} div {q_['div']:.2f} gap {q_['gap']:.3f}")

# ---- C1 convergence: mean distance to target vs sweep --------------------------------
d = pools["c1"]
R["c1_conv"] = {"sweeps": d["sweeps"].tolist(), "dist": {}}
for k in range(K):
    for v in dict.fromkeys(d["label"]):
        sel = (d["target"] == k) & (d["label"] == v)
        R["c1_conv"]["dist"][f"{k}_{v[3:]}"] = np.linalg.norm(d["pc"][:, sel] - centres[k], axis=-1).mean(1).tolist()

# ---- M2: fields vs no fields, unconditional ------------------------------------------
def load(tag):
    ck = torch.load(OUT / f"model_{tag}.pt")
    m = AttentionPotts(ck["L"], H=ck["cfg"]["H"], d=ck["cfg"]["d"], field=ck["cfg"].get("field", "none")).cuda()
    m.load_state_dict(ck["state"])
    return m.eval()


share_nat = np.array([wte[lab_te == k].sum() for k in range(K)]) / wte.sum()
R["m2"] = {"share_test": share_nat.tolist()}
f_te = freqs(Zte, wte)
C_te = connected_corr(*f_te)
for name, tag in (("base", BASE), ("m2", M2)):
    mdl = load(tag)
    Zg = pools[name]["Z"]
    lab_g = assign(fr.transform(Zg))
    share = np.bincount(lab_g, minlength=K) / len(lab_g)
    with torch.no_grad():
        J = mdl.couplings()
        h = mdl.fields(None, 1)
        h = None if h is None else h[0].cpu().numpy()
        nl = {}
        for split, Z, w in (("train", Ztr, wtr), ("test", Zte, wte)):
            wt = torch.as_tensor(w / w.sum(), dtype=torch.float32, device="cuda")
            nl[split] = mdl.neg_log_pl(torch.as_tensor(Z, device="cuda"), wt, 0)[1].item()
    E_own_tr, P_own_tr = scores(J.cpu(), Ztr, h=h)
    E_own_g, _ = scores(J.cpu(), Zg, h=h)
    Eg = np.array([E_own_g[lab_g == k].mean() for k in range(K)])
    Fg = -np.log(np.maximum(share, 1e-9))                       # F_k up to a common constant
    R.setdefault("thermo", {})[name] = {"E_gen": Eg.tolist(), "F": (Fg - Fg[1]).tolist(),
                                        "S": ((Eg - Eg[1]) - (Fg - Fg[1])).tolist()}
    fg = freqs(Zg)
    R["m2"][name] = {
        "share": share.tolist(), "dF": (-np.log(np.maximum(share, 1e-9) / share_nat)).tolist(),
        "nlpl": nl, "f1": compare(f_te[0].ravel(), fg[0].ravel()), "Cij": compare(C_te, connected_corr(*fg)),
        "hpd": hpd(Zg), "gap": float((Zg == Q - 1).mean()),
        "E_cluster_train": [float(wtr[lab_tr == k] @ E_own_tr[lab_tr == k] / wtr[lab_tr == k].sum()) for k in range(K)],
        "PLL_cluster_train": [float(wtr[lab_tr == k] @ P_own_tr[lab_tr == k] / wtr[lab_tr == k].sum()) for k in range(K)],
        "field_abs_mean": None if h is None else float(np.abs(h).mean()),
        "J_frob": float(J.pow(2).sum().sqrt()),
    }
    print(f"{name}: shares {np.round(share, 3)} vs nat {np.round(share_nat, 3)}  dF {np.round(R['m2'][name]['dF'], 2)}  "
          f"r(f) {R['m2'][name]['f1']['pearson']:.3f} r(C) {R['m2'][name]['Cij']['pearson']:.3f}  "
          f"test NLPL {nl['test']:.2f}  E/cluster {np.round(R['m2'][name]['E_cluster_train'], 1)}  "
          f"PLL/cluster {np.round(R['m2'][name]['PLL_cluster_train'], 1)}")
# ---- mode weights of field-free baselines that differ only in training length -----------
# base_clean: the main report's model (patience 50; retraining it with the current code gives
# bitwise-identical couplings); base (basep): patience 150. Same sampling protocol for both.
R["mode_fragility"] = {}
for f_, key in (("base_clean", "ShareClean"), ("base", "ShareBasep")):
    if (OUTC / f"{f_}.npz").exists():
        lab_f = assign(fr.transform(np.load(OUTC / f"{f_}.npz")["Z"]))
        sh = np.bincount(lab_f, minlength=K) / len(lab_f)
        R["mode_fragility"][key] = "/".join(f"{100 * x:.0f}" for x in sh)
        print(f"{f_}: shares {np.round(sh, 3)}")
json.dump(R, open(OUT / "metrics_cond.json", "w"), indent=1)

# =====================================================================================
# Figures
# =====================================================================================
rng = np.random.default_rng(0)
bg = rng.choice(len(P_te), 4000, replace=False)


def pca_panel(ax, P, col, title, n=350):
    ax.scatter(P_te[bg, 0], P_te[bg, 1], s=2, c="#cfcec9", lw=0, rasterized=True)
    for k in range(K):
        Pk = P[k]
        idx = rng.choice(len(Pk), min(n, len(Pk)), replace=False)
        ax.scatter(Pk[idx, 0], Pk[idx, 1], s=3, c=col[k], lw=0, alpha=0.75, rasterized=True)
        ax.scatter(*centres[k], marker="*", s=90, c=col[k], edgecolors=INK, linewidths=0.6, zorder=5)
    ax.set_title(title)


def pool(m, k, v):
    d = pools[m]
    return fr.transform(d["Z"][(d["target"] == k) & (d["label"] == v)])


panels = [("unconditional baseline", [fr.transform(pools["base"]["Z"])] * K, [INK] * K)]
panels += [(rf"C1  $\lambda$={v}", [pool("c1", k, f"lam{v}") for k in range(K)], TCOL) for v in (1, 4, 16)]
panels += [(rf"C3  $\sigma$=0.5, $\kappa$={v}", [pool("c3", k, f"s0.5_k{v}") for k in range(K)], TCOL)
           for v in (1, 4)]
fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.9), sharex=True, sharey=True)
for ax, (t, P, col) in zip(axs.ravel(), panels):
    pca_panel(ax, P, col, t)
for ax in axs[1]:
    ax.set_xlabel("PC1")
for ax in axs[:, 0]:
    ax.set_ylabel("PC2")
fig.tight_layout()
fig.savefig(FIG / "cond_pca.pdf"); plt.close(fig)

# trade-off: accuracy vs quality, one curve per (method, target)
SERIES = (("c1", [f"lam{v}" for v in ("0.25", "1", "4", "16", "64")], "-", "o", r"C1 ($\lambda$)"),
          ("c3", ["s1_k1", "s0.5_k1", "s0.25_k1"], "--", "s", r"C3 ($\sigma$, $\kappa$=1)"),
          ("c3", ["s0.5_k1", "s0.5_k2", "s0.5_k4", "s0.5_k8"], ":", "^", r"C3 ($\kappa$, $\sigma$=0.5)"))
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.7))
for k in range(K):
    for m, ks, ls, mk, lab in SERIES:
        rows = R["cond"][m][str(k)]
        x = [rows[v]["dist"] for v in ks]
        for ax, key in zip(axs, ("Cij", "hpd", "E_judge")):
            y = [rows[v][key]["pearson"] if key == "Cij" else rows[v][key] for v in ks]
            ax.plot(x, y, ls=ls, marker=mk, ms=3, c=TCOL[k], lw=1,
                    label=f"{lab} → C{k}" if ax is axs[0] else None)
    ref = R["ref"][k]
    for ax, key in zip(axs, ("Cij", "hpd", "E_judge")):
        yc = ref["ceiling"]["Cij"]["pearson"] if key == "Cij" else (ref["natural_test"][key])
        ax.axhline(yc, c=TCOL[k], lw=0.7, ls=":")
        yr = ref["rejection"]["Cij"]["pearson"] if key == "Cij" else ref["rejection"][key]
        ax.scatter(ref["rejection"]["dist"], yr, marker="D", s=22, facecolors="none", edgecolors=TCOL[k], zorder=5)
for ax, t in zip(axs, (r"$r(C_{ij})$ vs natural cluster", "HPD motif retained", "energy (baseline judge)")):
    ax.set_xscale("log"); ax.set_xlabel("mean distance to target (PC units)"); ax.set_title(t)
fig.legend(*axs[0].get_legend_handles_labels(), ncol=3, fontsize=6.3, loc="upper center",
           bbox_to_anchor=(0.5, 0.02))
fig.tight_layout()
fig.savefig(FIG / "cond_tradeoff.pdf"); plt.close(fig)

# C1 convergence
fig, ax = plt.subplots(figsize=(3.6, 2.4))
sw = np.array(R["c1_conv"]["sweeps"])
for key, y in R["c1_conv"]["dist"].items():
    k, v = key.split("_")
    if v in ("0", "1", "4", "64"):
        ax.plot(sw, y, c=TCOL[int(k)], lw=0.9, ls={"0": ":", "1": "-.", "4": "--", "64": "-"}[v])
ax.set_xlabel("sweep"); ax.set_ylabel("mean distance to target")
ax.set_xscale("symlog", linthresh=10)
ax.set_title(r"C1 convergence ($\lambda$ = 0 dotted, 1, 4, 64 solid)")
fig.tight_layout()
fig.savefig(FIG / "cond_convergence.pdf"); plt.close(fig)

# M2: cluster shares (incl. the main report's shorter-trained baseline) + free-energy decomposition
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
x = np.arange(K)
sh_clean = np.bincount(assign(fr.transform(np.load(OUTC / "base_clean.npz")["Z"])), minlength=K) / 10000
bars = (("natural (test, weighted)", share_nat, BLUE), ("baseline, 668 epochs (report)", sh_clean, "#c9c8c3"),
        ("baseline, 962 epochs", R["m2"]["base"]["share"], GREY), ("M2 (+ fields), 962 epochs", R["m2"]["m2"]["share"], ORANGE))
for j_, (lab, v, c) in enumerate(bars):
    axs[0].bar(x + (j_ - 1.5) * 0.2, v, 0.2, color=c, label=lab)
axs[0].set_xticks(x, [f"C{k}" for k in range(K)]); axs[0].set_ylabel("share of sequences $s_k$")
axs[0].legend(fontsize=6.3, loc="upper left", bbox_to_anchor=(0.0, 1.0))
axs[0].set_ylim(0, 1.05)
th = R["thermo"]
for j_, (lab, key, hatch) in enumerate(((r"$\langle E\rangle_k-\langle E\rangle_{C1}$", "E_gen", ""),
                                        (r"$F_k-F_{C1}$", "F", "//"), (r"$S_k-S_{C1}$", "S", ".."))):
    for n_, (name, c) in enumerate((("base", GREY), ("m2", ORANGE))):
        v = np.array(th[name][key]); v = v - v[1] if key == "E_gen" else v
        axs[1].bar(np.array([0, 2]) * 1.0 + j_ * 0.27 + n_ * 0.12 - 0.3, v[[0, 2]], 0.12, color=c,
                   hatch=hatch, edgecolor="white", lw=0, label=lab if n_ == 0 else None)
axs[1].set_xticks([0, 2], ["C0 vs C1", "C2 vs C1"]); axs[1].set_ylabel(r"nats (energy units at $\beta$=1)")
axs[1].axhline(0, c=MUTED, lw=0.6)
axs[1].legend(fontsize=6.5, title="grey: baseline, orange: M2", title_fontsize=6.3)
fig.tight_layout()
fig.savefig(FIG / "m2_shares.pdf"); plt.close(fig)
print("figures written")
