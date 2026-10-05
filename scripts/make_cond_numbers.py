"""outputs/metrics_cond.json -> report/cond_numbers.tex (macros) + report/tab_cond_*.tex (tables)."""
import json

import numpy as np

from common import OUT, ROOT

R = json.load(open(OUT / "metrics_cond.json"))
REP = ROOT / "report"
H = {t: json.load(open(OUT / f"train_hist_{t}.json")) for t in
     ("basep", "m2p", "c3p", "c3p_s0.25", "c3p_s1.0", "clean", "repro50") if (OUT / f"train_hist_{t}.json").exists()}
mac = {}


def f(x, n=2):
    return f"{x:.{n}f}"


def pct(x):
    return f"{100 * x:.0f}"


# ---- thermodynamics of the baseline's unconditional samples ------------------------------
th = R["thermo"]["base"]
mac["ThermoEgap"] = f(th["E_gen"][0] - th["E_gen"][1], 1)
mac["ThermoFgap"] = f(th["F"][0], 1)
mac["ThermoSgap"] = f(th["S"][0], 1)
mac["ThermoTable"] = ("$\\langle E\\rangle_k-\\langle E\\rangle_{C1}=("
                      + ",\\,".join(f(e - th["E_gen"][1], 1) for e in th["E_gen"]) + ")$, $F_k-F_{C1}=("
                      + ",\\,".join(f(x, 1) for x in th["F"]) + ")$ and therefore $S_k-S_{C1}=("
                      + ",\\,".join(f(x, 1) for x in th["S"]) + ")$ for $k=$ C0, C1, C2")

# ---- M2 ----------------------------------------------------------------------------------
m2 = R["m2"]
for name, key in (("base", "Base"), ("m2", "Mtwo")):
    r = m2[name]
    mac[f"Share{key}"] = "/".join(pct(s) for s in r["share"])
    mac[f"dFzero{key}"] = f(r["dF"][0])
    mac[f"rf{key}"] = f(r["f1"]["pearson"], 3)
    mac[f"rC{key}"] = f(r["Cij"]["pearson"], 3)
    mac[f"sC{key}"] = f(r["Cij"]["slope"])
    mac[f"NLPLtest{key}"] = f(r["nlpl"]["test"])
    mac[f"Jfrob{key}"] = f(r["J_frob"], 1)
mac["ShareNat"] = "/".join(pct(s) for s in m2["share_test"])
mac["FieldAbs"] = f(m2["m2"]["field_abs_mean"])
for t, key in (("basep", "Base"), ("m2p", "Mtwo"), ("c3p", "Cthree"), ("c3p_s0.25", "CthreeQ"),
               ("c3p_s1.0", "CthreeOne"), ("clean", "Clean"), ("repro50", "Repro")):
    if t in H:
        mac[f"Val{key}"] = f(H[t]["best_val_nlpl"])
        mac[f"Ep{key}"] = str(int(np.argmin(H[t]["val_nlpl"])) + 1)
for k, v in R.get("mode_fragility", {}).items():
    mac[k] = v

rows = [("natural test (weighted)", None, m2["share_test"], None)]
tab = ["\\begin{tabular}{lcccccc}", "\\toprule",
       " & val.\\ NLPL & test NLPL & $r(f_i)$ & $r(C_{ij})$ / slope & shares C0/C1/C2 (\\%) & $\\Delta F_{C0}$ \\\\",
       "\\midrule",
       f"natural & -- & -- & -- & -- & {mac['ShareNat']} & -- \\\\"]
for name, key, t in (("base", "Base", "no fields"), ("m2", "Mtwo", "M2 (+ fields)")):
    tab.append(f"{t} & {mac['Val' + key]} & {mac['NLPLtest' + key]} & {mac['rf' + key]} & "
               f"{mac['rC' + key]} / {mac['sC' + key]} & {mac['Share' + key]} & {mac['dFzero' + key]} \\\\")
tab += ["\\bottomrule", "\\end{tabular}"]
open(REP / "tab_cond_m2.tex", "w").write("\n".join(tab) + "\n")

# ---- conditional ------------------------------------------------------------------------
NAMES = {"lam0.25": r"C1 $\lambda{=}0.25$", "lam1": r"C1 $\lambda{=}1$", "lam4": r"C1 $\lambda{=}4$",
         "lam16": r"C1 $\lambda{=}16$", "lam64": r"C1 $\lambda{=}64$",
         "s1_k1": r"C3 $\sigma{=}1$", "s0.5_k1": r"C3 $\sigma{=}0.5$", "s0.25_k1": r"C3 $\sigma{=}0.25$",
         "s0.5_k2": r"C3 $\sigma{=}0.5$, $\kappa{=}2$", "s0.5_k4": r"C3 $\sigma{=}0.5$, $\kappa{=}4$",
         "s0.5_k8": r"C3 $\sigma{=}0.5$, $\kappa{=}8$"}
ORDER_C1 = ["lam1", "lam4", "lam16", "lam64"]
ORDER_C3 = ["s1_k1", "s0.5_k1", "s0.25_k1", "s0.5_k2", "s0.5_k4", "s0.5_k8"]


def line(lab, q):
    return (f"{lab} & {f(q['dist'])} & {q['overshoot']:+.2f} & {pct(q['hit'])} & {f(q['f1']['pearson'], 3)} & "
            f"{f(q['Cij']['pearson'], 3)} & {pct(q['hpd'])} & {f(q['E_judge'], 1)} & {f(q['d_train'])} & "
            f"{f(q['div'])} \\\\")


for k in range(len(R["targets"])):
    ref = R["ref"][str(k)]
    t = ["\\begin{tabular}{lccccccccc}", "\\toprule",
         " & dist. & overshoot & in $C_k$ (\\%) & $r(f_i)$ & $r(C_{ij})$ & HPD (\\%) & $E$ & $d_\\text{train}$ & div. \\\\",
         "\\midrule",
         line("natural (train, ceiling)", ref["ceiling"]),
         line(f"rejection from baseline (n={ref['rejection']['n']})", ref["rejection"]),
         line("unconditional baseline", ref["uncond"]),
         "\\midrule"]
    t += [line(NAMES[v], R["cond"]["c1"][str(k)][v]) for v in ORDER_C1]
    t += ["\\midrule"]
    t += [line(NAMES[v], R["cond"]["c3"][str(k)][v]) for v in ORDER_C3]
    t += ["\\bottomrule", "\\end{tabular}"]
    open(REP / f"tab_cond_C{k}.tex", "w").write("\n".join(t) + "\n")
    mac[f"NatE{'ABC'[k]}"] = f(ref["natural_test"]["E_judge"], 1)
    mac[f"NatN{'ABC'[k]}"] = str(ref["natural_test"]["n"])
    mac[f"Ceil{'ABC'[k]}"] = f(ref["ceiling"]["Cij"]["pearson"], 3)


# convenient per-cell macros: \CondC1A{lam4}{dist} is too fragile in LaTeX -> flat names
def key(s):
    return s.replace("lam", "L").replace("0.25", "Q").replace("0.5", "H").replace("_k", "K").replace(".", "")


DIG = {"0": "zero", "1": "one", "2": "two", "4": "four", "6": "six", "8": "eight", "Q": "Q", "H": "H"}
for m in ("c1", "c3"):
    for k, rows in R["cond"][m].items():
        for v, q in rows.items():
            nm = "".join(DIG.get(ch, ch) for ch in key(v))
            base = f"{m.upper().replace('1', 'one').replace('3', 'three')}{'ABC'[int(k)]}{nm}"
            mac[base + "dist"] = f(q["dist"])
            mac[base + "over"] = f"{q['overshoot']:+.2f}"
            mac[base + "rC"] = f(q["Cij"]["pearson"], 3)
            mac[base + "rf"] = f(q["f1"]["pearson"], 3)
            mac[base + "hpd"] = pct(q["hpd"])
            mac[base + "E"] = f(q["E_judge"], 1)
            mac[base + "div"] = f(q["div"])
            mac[base + "hit"] = pct(q["hit"])
            mac[base + "dtr"] = f(q["d_train"])

with open(REP / "cond_numbers.tex", "w") as fh:
    for k, v in mac.items():
        assert k.isalpha(), k
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"{len(mac)} macros")

# ---- compact summary table for report.tex / paper.tex: all three targets side by side --------
def trio(rows, fmt):
    return "/".join(fmt(rows[str(k)]) for k in range(len(R["targets"])))


SUMMARY = [("unconditional baseline", {str(k): R["ref"][str(k)]["uncond"] for k in range(3)}),
           ("rejection from baseline", {str(k): R["ref"][str(k)]["rejection"] for k in range(3)}),
           None,
           *[(NAMES[v], {k: R["cond"]["c1"][k][v] for k in R["cond"]["c1"]}) for v in ("lam1", "lam4", "lam16")],
           None,
           *[(NAMES[v], {k: R["cond"]["c3"][k][v] for k in R["cond"]["c3"]}) for v in ("s1_k1", "s0.5_k1", "s0.25_k1")],
           None,
           *[(NAMES[v], {k: R["cond"]["c3"][k][v] for k in R["cond"]["c3"]}) for v in ("s0.5_k4", "s0.5_k8")],
           None,
           ("natural (ceiling)", {str(k): R["ref"][str(k)]["ceiling"] for k in range(3)})]
t = [r"\begin{tabular}{lccccc}", r"\toprule",
     r" & in target (\%) & distance & $r(C_{ij})$ & HPD (\%) & diversity \\", r"\midrule"]
for row in SUMMARY:
    if row is None:
        t.append(r"\midrule")
        continue
    lab, rows = row
    t.append(f"{lab} & {trio(rows, lambda q: pct(q['hit']))} & {trio(rows, lambda q: f(q['dist'], 1))} & "
             f"{trio(rows, lambda q: f(q['Cij']['pearson']))} & {trio(rows, lambda q: pct(q['hpd']))} & "
             f"{trio(rows, lambda q: f(q['div']))} \\\\")
t += [r"\bottomrule", r"\end{tabular}"]
open(REP / "tab_cond_summary.tex", "w").write("\n".join(t) + "\n")
