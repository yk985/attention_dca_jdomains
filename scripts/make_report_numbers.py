"""metrics JSON -> report/auto_numbers.tex (macros) + report/tab_*.tex (tables)."""
import argparse
import json

import numpy as np

from common import OUT, ROOT

p = argparse.ArgumentParser()
p.add_argument("--tag", default="main")
p.add_argument("--compare", default=None,
               help="second metrics tag; a few of its numbers are emitted with the prefix 'Cmp'")
a = p.parse_args()
M = json.load(open(OUT / f"metrics_{a.tag}.json"))
sweep = json.load(open(OUT / "sweep.json"))
betas = M["betas"]
G = [f"b{b:g}" for b in betas]
R = ROOT / "report"
L = M["sizes"]["L"]

mac = {}
def m(name, val):
    mac[name] = val
s = M["sizes"]
m("NL", L); m("Mtrain", f"{s['train']:,}".replace(",", "\\,")); m("Mtest", f"{s['test']:,}".replace(",", "\\,"))
m("Mnovel", s["test_novel"]); m("Mcopies", s["test"] - s["test_novel"])
m("PctCopies", f"{100 * (s['test'] - s['test_novel']) / s['test']:.0f}")
m("ThetaTrain", f"{s['theta_train']:.3f}"); m("MeffTrain", f"{s['meff_train']:.0f}")
m("ThetaTest", f"{s['theta_test']:.3f}"); m("MeffTest", f"{s['meff_test']:.0f}")
m("MeffNovel", f"{s['meff_test_novel']:.0f}")
m("NfOne", f"{L * 21:,}".replace(",", "\\,")); m("NCij", f"{L * (L - 1) // 2 * 441:,}".replace(",", "\\,"))
t = M["train"]
m("Hbest", t["cfg"]["H"]); m("dbest", t["cfg"]["d"])
lam = t["cfg"]["lambd"]; e = int(f"{lam:e}".split("e")[1]); mant = lam / 10 ** e
m("lambest", (f"{mant:g}\\cdot" if abs(mant - 1) > 1e-9 else "") + f"10^{{{e}}}")
m("Nparams", f"{t['n_params']:,}".replace(",", "\\,")); m("BestEpoch", t["best_epoch"])
m("ValNLPL", f"{t['best_val_nlpl']:.2f}")
for k, v in M["nlpl_per_seq"].items():
    m({"train": "NLPLtrain", "test": "NLPLtest", "test_novel": "NLPLnovel"}[k], f"{v:.2f}")
_gap = M["nlpl_per_seq"]["test"] - M["nlpl_per_seq"]["train"]
m("NLPLgap", f"{_gap:.1f}")
m("NLPLgapPct", f"{100 * _gap / M['nlpl_per_seq']['train']:.0f}")
m("NLPLtestSite", f"{M['nlpl_per_seq']['test'] / L:.2f}")
c = M["contacts"]
m("PPVL", f"{c['ppv_at_L']:.2f}"); m("PPVLhalf", f"{c['ppv_at_L2']:.2f}"); m("PPVbase", f"{c['base_rate']:.2f}")
m("PCone", f"{100 * M['pca_explained_var'][0]:.1f}"); m("PCtwo", f"{100 * M['pca_explained_var'][1]:.1f}")
m("Etest", f"{M['energy']['test']['mean_w']:.1f}"); m("Etrain", f"{M['energy']['train']['mean_w']:.1f}")
m("Erandom", f"{M['energy']['random']['mean']:.1f}")
m("HamTestTest", f"{M['hamming']['test-test']['mean']:.2f}")
m("HamTestNN", f"{M['hamming']['test->test(nn)']['mean']:.2f}")
m("HamNovelTrain", f"{M['hamming']['novel->train']['mean']:.2f}")
m("HamTestTrain", f"{M['hamming']['test->train']['mean']:.2f}")
m("rCtrain", f"{M['Cij']['train']['pearson']:.3f}"); m("rCprof", f"{M['Cij']['prof']['pearson']:.3f}")
m("sCtrain", f"{M['Cij']['train']['slope']:.2f}")
if "Cij_reliability" in M:
    RB = M["Cij_reliability"]
    m("Crel", f"{RB['reliability']:.2f}")
    m("sCdisTrain", f"{RB['slope_disattenuated']['train']:.2f}")
    for g, name in zip(G, ["One", "Star"]):
        m(f"sCdis{name}", f"{RB['slope_disattenuated'][g]:.2f}")
m("rCnovelFull", f"{M['Cij_vs_novel']['test_full']['pearson']:.3f}")
m("BetaStar", f"{betas[-1]:g}")
for g, name in zip(G, ["One", "Star"]):
    sj = M["sampling"][g]
    m(f"Tau{name}", f"{sj['tau_int_energy']:.1f}"); m(f"TauP{name}", f"{sj['tau_int_pll']:.1f}")
    m(f"Burn{name}", sj["n_burn"]); m(f"Ngen{name}", f"{sj['n_generated']:,}".replace(",", "\\,"))
    m(f"Ncol{name}", sj["n_collapsed"]); m(f"PctCol{name}", f"{100 * sj['n_collapsed'] / sj['n_gen_chains']:.1f}")
    m(f"StudyColNat{name}", f"{100 * sj['study_collapsed_nat']:.1f}")
    m(f"StudyColRnd{name}", f"{100 * sj['study_collapsed_rnd']:.1f}")
    m(f"HamEnd{name}", f"{sj['ham_end_nat']:.2f}"); m(f"SweepsNinety{name}", sj["sweeps_to_90pct_ham"])
    m(f"EendNat{name}", f"{sj['E_med_end_nat']:.1f}"); m(f"EendRnd{name}", f"{sj['E_med_end_rnd']:.1f}")
    m(f"Egen{name}", f"{M['energy'][g]['mean']:.1f}")
    m(f"rF{name}", f"{M['f1'][g]['pearson']:.3f}"); m(f"rC{name}", f"{M['Cij'][g]['pearson']:.3f}")
    m(f"sC{name}", f"{M['Cij'][g]['slope']:.2f}")
    m(f"rCnovel{name}", f"{M['Cij_vs_novel'][g]['pearson']:.3f}")
    m(f"HamGG{name}", f"{M['hamming'][f'{g}-{g}']['mean']:.2f}")
    m(f"HamGT{name}", f"{M['hamming'][f'{g}->test']['mean']:.2f}")
    m(f"HamGTr{name}", f"{M['hamming'][f'{g}->train']['mean']:.2f}")
    m(f"HamGTrMin{name}", f"{M['hamming'][f'{g}->train']['q05']:.2f}")
    m(f"Uniq{name}", f"{100 * M['unique_frac'][g]:.1f}")
    m(f"StudySweeps", sj["study_sweeps"])
    m(f"WPCa{name}", f"{M['pca_wasserstein'][g][0]:.2f}"); m(f"WPCb{name}", f"{M['pca_wasserstein'][g][1]:.2f}")
    m(f"WE{name}", f"{M['wasserstein_energy'][g]:.1f}")
m("WPCaTrain", f"{M['pca_wasserstein']['train'][0]:.2f}"); m("WPCbTrain", f"{M['pca_wasserstein']['train'][1]:.2f}")
m("WPCaProf", f"{M['pca_wasserstein']['prof'][0]:.2f}"); m("WPCbProf", f"{M['pca_wasserstein']['prof'][1]:.2f}")
m("WEtrain", f"{M['wasserstein_energy']['train']:.1f}"); m("WEprof", f"{M['wasserstein_energy']['prof']:.1f}")
m("Eunw", f"{M['energy']['test']['mean']:.1f}")
mc = M["pca_minor_cluster"]
m("ClTest", f"{100 * mc['test']:.1f}"); m("ClTrain", f"{100 * mc['train']:.1f}")
m("ClOne", f"{100 * mc[G[0]]:.1f}"); m("ClStar", f"{100 * mc[G[-1]]:.1f}")

# cluster-restricted statistics
if "restricted" in M:
    R_ = M["restricted"]
    pc = M["pca_clusters"]
    m("Kclust", pc["k"]); m("NcovClust", len(pc["covered"]))
    m("CovWeight", f"{100 * pc['covered_natural_weight']:.1f}")
    m("NtestCov", f"{R_['n']['test']:,}".replace(",", "\\,"))
    m("MeffTestCov", f"{R_['meff_test']:.0f}")
    m("rCtrainCov", f"{R_['Cij']['train']['pearson']:.3f}")
    for g, name in zip(G, ["One", "Star"]):
        m(f"rFcov{name}", f"{R_['f1'][g]['pearson']:.3f}")
        m(f"rCcov{name}", f"{R_['Cij'][g]['pearson']:.3f}")
        m(f"sCcov{name}", f"{R_['Cij'][g]['slope']:.2f}")
        m(f"rCcovTop{name}", f"{R_['Cij_top1pct'][g]['pearson']:.3f}")

# data audit + split construction
if (OUT / "data_audit.json").exists():
    A = json.load(open(OUT / "data_audit.json"))
    m("NsuperSet", f"{A['files']['filtered']['unique_seqs']:,}".replace(",", "\\,"))
    m("OrigCopies", f"{100 * A['pairs']['train2|test2']['test_copies']:.1f}")
    m("OrigNN", f"{A['pairs']['train2|test2']['nn_mean']:.3f}")
sp = OUT / "split" / "split_stats_c0.2.json"
if sp.exists():
    S_ = json.load(open(sp))
    m("SplitThresh", f"{S_['thresh']:g}"); m("SplitIdent", f"{100 * (1 - S_['thresh']):.0f}")
    m("NclustTot", f"{S_['n_clusters']:,}".replace(",", "\\,"))
    m("NsingleClust", f"{S_['n_singletons']:,}".replace(",", "\\,"))
    m("NtrainClean", f"{S_['n_train']:,}".replace(",", "\\,"))
    m("NtestClean", f"{S_['n_test']:,}".replace(",", "\\,"))
    m("NdropClean", f"{S_['dropped_from_test']:,}".replace(",", "\\,"))
    m("CleanNN", f"{S_['nn_mean']:.3f}"); m("CleanNNmin", f"{S_['nn_min']:.3f}")
    m("CleanNNminSub", f"{round(S_['nn_min'] * L)}")
    m("CleanNNmed", f"{S_.get('nn_median', float('nan')):.3f}")
    m("CleanNNqfive", f"{S_.get('nn_q05', float('nan')):.3f}")
    m("CleanPctNear", f"{100 * S_.get('frac_nn_lt_thresh', 0):.0f}")
    m("CleanDropRule", S_.get("drop_rule", "exact"))
# the other split, for the side-by-side paragraph
if a.compare:
    X = json.load(open(OUT / f"metrics_{a.compare}.json"))
    XG = [f"b{b:g}" for b in X["betas"]]
    m("CmpMtrain", f"{X['sizes']['train']:,}".replace(",", "\\,"))
    m("CmpMtest", f"{X['sizes']['test']:,}".replace(",", "\\,"))
    m("CmpPctCopies", f"{100 * (X['sizes']['test'] - X['sizes']['test_novel']) / X['sizes']['test']:.0f}")
    m("CmpValNLPL", f"{X['train']['best_val_nlpl']:.2f}")
    m("CmpNLPLtest", f"{X['nlpl_per_seq']['test']:.2f}")
    m("CmpNLPLnovel", f"{X['nlpl_per_seq']['test_novel']:.2f}")
    m("CmprFOne", f"{X['f1'][XG[0]]['pearson']:.3f}")
    m("CmprCOne", f"{X['Cij'][XG[0]]['pearson']:.3f}")
    m("CmprCtrain", f"{X['Cij']['train']['pearson']:.3f}")
    m("CmpHamGTrOne", f"{X['hamming'][f'{XG[0]}->train']['mean']:.2f}")
    m("CmpPPVLhalf", f"{X['contacts']['ppv_at_L2']:.2f}")
    if "restricted" in X:
        m("CmprCcovOne", f"{X['restricted']['Cij'][XG[0]]['pearson']:.3f}")

with open(R / "auto_numbers.tex", "w") as f:
    for k, v in mac.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---- tables ------------------------------------------------------------------------
def row(label, d):
    return f"{label} & {d['pearson']:.3f} & {d['spearman']:.3f} & {d['slope']:.2f} & {d['rmse']:.4f}"

lab = {G[0]: r"generated, $\beta=1$", G[-1]: rf"generated, $\beta^*={betas[-1]:g}$",
       "train": "natural train (ceiling)", "prof": "independent-site model"}
lines = [r"\begin{tabular}{llcccc}", r"\toprule",
         r"statistic & set vs.\ natural test & Pearson $r$ & Spearman $\rho$ & slope & RMSE \\", r"\midrule"]
for stat, key in [(r"$f_i(a)$", "f1"), (r"$C_{ij}(a,b)$", "Cij"), (r"$C_{ij}$, top 1\%", "Cij_top1pct")]:
    for j, k in enumerate(G + ["train", "prof"]):
        lines.append((stat if j == 0 else "") + " & " + row(lab[k], M[key][k]) + r" \\")
    lines.append(r"\midrule")
if M["sizes"]["test_novel"] < M["sizes"]["test"]:      # only meaningful on the species split
    for j, k in enumerate(G + ["train"]):
        lines.append((r"$C_{ij}$ vs.\ novel test" if j == 0 else "") + " & "
                     + row(lab[k], M["Cij_vs_novel"][k]) + r" \\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}"]
open(R / "tab_stats.tex", "w").write("\n".join(lines))

H = M["hamming"]
def hrow(label, k):
    d = H[k]
    return f"{label} & {d['mean']:.3f} & {d['median']:.3f} & {d['q05']:.3f} & {100 * d['frac_zero']:.1f}\\% \\\\"
lines = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"distance (normalised by $L$) & mean & median & 5\% quantile & $=0$ \\", r"\midrule"]
for g in G:
    b = lab[g].replace("generated, ", "")
    lines += [hrow(f"gen ({b}) $\\to$ nearest test", f"{g}->test"),
              hrow(f"gen ({b}) $\\to$ nearest train", f"{g}->train"),
              hrow(f"gen ({b}) $\\to$ nearest other gen", f"{g}->{g}(nn)"),
              hrow(f"gen--gen, all pairs ({b})", f"{g}-{g}"),
              hrow(f"gen--test, all pairs ({b})", f"{g}-test"), r"\midrule"]
lines += [hrow(r"test $\to$ nearest other test", "test->test(nn)"),
          hrow(r"test $\to$ nearest train", "test->train"),
          hrow(r"novel test $\to$ nearest train", "novel->train"),
          hrow("test--test, all pairs", "test-test"), r"\bottomrule", r"\end{tabular}"]
open(R / "tab_hamming.tex", "w").write("\n".join(lines))

lines = [r"\begin{tabular}{rrlrrr}", r"\toprule",
         r"$H$ & $d$ & $\lambda$ & params & best epoch & val.\ $-\log\mathrm{PL}$ \\", r"\midrule"]
for r_ in sweep:
    lines.append(f"{r_['H']} & {r_['d']} & $10^{{{int(round(__import__('math').log10(r_['lambd'])))}}}$ & "
                 f"{r_['n_params']:,} & {r_['best_epoch']} & {r_['val']:.2f} \\\\".replace(",", "\\,", 1))
lines += [r"\bottomrule", r"\end{tabular}"]
open(R / "tab_sweep.tex", "w").write("\n".join(lines))

if "beta_scan" in M:
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"start & $\beta$ & collapsed & $\langle E\rangle$ & $r(f_i)$ & $r(C_{ij})$ & slope $C_{ij}$ & gen--gen $d$ \\",
             r"\midrule"]
    for r_ in M["beta_scan"]:
        lines.append(f"{r_['init']} & {r_['beta']:g} & {100 * r_['collapsed_frac']:.1f}\\% & {r_['E_mean']:.1f} & "
                     f"{r_['r_f1']:.3f} & {r_['r_C']:.3f} & {r_['slope_C']:.2f} & {r_['gen_gen']:.3f} \\\\")
        if r_ is M["beta_scan"][len(M["beta_scan"]) // 2 - 1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(R / "tab_beta.tex", "w").write("\n".join(lines))
# cluster-restricted statistics table
if "restricted" in M:
    R_ = M["restricted"]
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"statistic & set vs.\ natural (covered clusters) & Pearson $r$ & Spearman $\rho$ & slope & RMSE \\",
             r"\midrule"]
    for stat, key in [(r"$f_i(a)$", "f1"), (r"$C_{ij}(a,b)$", "Cij"), (r"$C_{ij}$, top 1\%", "Cij_top1pct")]:
        for j, k in enumerate(G + ["train", "prof"]):
            lines.append((stat if j == 0 else "") + " & " + row(lab[k], R_[key][k]) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(R / "tab_clusters.tex", "w").write("\n".join(lines))

# per-cluster energy table
if "cluster_energy" in M:
    lines = [r"\begin{tabular}{crrrrrrr}", r"\toprule",
             r"cluster & \multicolumn{2}{c}{weight share} & \multicolumn{2}{c}{$\langle E\rangle$ natural}"
             r" & \multicolumn{2}{c}{$\langle\log\mathrm{PL}\rangle$ natural} & $\Delta F$ \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r" & natural & generated & test & train & test & train & (nats) \\", r"\midrule"]
    for r_ in M["cluster_energy"]:
        tag = "" if r_["covered"] else r"$^{\dagger}$"
        lines.append(f"C{r_['cluster']}{tag} & {r_['share_test']:.3f} & {r_['share_' + G[0]]:.3f} & "
                     f"{r_['E_test']:.1f} & {r_['E_train']:.1f} & {r_['PLL_test']:.1f} & "
                     f"{r_['PLL_train']:.1f} & {r_['dF_' + G[0]]:+.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(R / "tab_cluster_energy.tex", "w").write("\n".join(lines))
    mi = M["pca_clusters"]["missed"][0] if M["pca_clusters"]["missed"] else 0
    rm = next(r_ for r_ in M["cluster_energy"] if r_["cluster"] == mi)
    rc = [r_ for r_ in M["cluster_energy"] if r_["covered"]]
    m("MissCl", f"C{mi}")
    m("EmissTest", f"{rm['E_test']:.1f}"); m("EmissTrain", f"{rm['E_train']:.1f}")
    m("EcovTest", "/".join(f"{r_['E_test']:.1f}" for r_ in rc))
    m("EcovTrain", "/".join(f"{r_['E_train']:.1f}" for r_ in rc))
    m("EgapMiss", f"{abs(min(r_['E_test'] for r_ in rc) - rm['E_test']):.0f}")
    m("PLLmissTrain", f"{rm['PLL_train']:.1f}")
    m("PLLcovTrain", "/".join(f"{r_['PLL_train']:.1f}" for r_ in rc))
    m("dFmiss", f"{rm['dF_' + G[0]]:+.2f}")
    m("dFmissFac", f"{np.exp(-rm['dF_' + G[0]]):.2f}")
    m("RawShareMiss", f"{100 * rm.get('rawshare_train', float('nan')):.1f}")
    m("MeffMiss", f"{rm.get('meff_train', float('nan')):.0f}")
    m("MeffCov", "/".join(f"{r_.get('meff_train', float('nan')):.0f}" for r_ in rc))
    m("BoostMiss", f"{np.exp(rm['dF_' + G[0]]):.1f}")

# data-file audit table
if (OUT / "data_audit.json").exists():
    A = json.load(open(OUT / "data_audit.json"))
    SHOW = ["filtered", "filtered2", "train", "test", "train2", "test2", "test_4000"]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"file & sequences & unique & \% in \texttt{filtered} & \% in \texttt{train2} \\", r"\midrule"]
    for k in SHOW:
        f_ = A["files"][k]
        lines.append(f"\\texttt{{{k.replace('_', chr(92) + '_')}}} & {f_['n']:,} & {f_['unique_seqs']:,} & "
                     f"{100 * A['seq_overlap'][k + '|filtered']:.0f}\\% & "
                     f"{100 * A['seq_overlap'][k + '|train2']:.0f}\\% \\\\".replace(",", "\\,"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(R / "tab_audit.tex", "w").write("\n".join(lines))

# rewrite the macro file: the table blocks above define a few macros of their own
with open(R / "auto_numbers.tex", "w") as f:
    for k, v in mac.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

print(f"wrote {len(mac)} macros and tables to {R}")
