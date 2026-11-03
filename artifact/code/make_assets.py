"""Render every number, table and figure of the paper from results/*.json (revision 4).

  python make_assets.py

Outputs (manuscript/generated/): numbers.tex, tab_main.tex, tab_audit.tex, fig_main.pdf

Optional inputs are rendered only when their result file exists (never hard-coded):
  lamp3_baselines_{qwen25_7b,qwen25_14b}.json   field-level baselines (Table 1 rows)
  {task}_heldout.json                           held-out third auditor (Table 2 column, macros)
  lamp2_analysis_qwen25-14b-ep18140.json         LaMP-2 extension (Table 1 group, macros)
  lamp2_pilot.json                              LaMP-2 pilot (macros)
  lamp3_analysis_300users.json                  7B reader on the 14B user set (macros)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paperfig as pf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
GEN = os.path.join(os.path.dirname(ROOT), "manuscript", "generated")
os.makedirs(GEN, exist_ok=True)
TASKS = ["lampqa", "lamp3", "lamp7"]
TNAME = {"lampqa": "LaMP-QA", "lamp3": "LaMP-3", "lamp7": "LaMP-7", "lamp2": "LaMP-2"}
TKEY = {"lampqa": "qa", "lamp3": "three", "lamp7": "seven", "lamp2": "two"}
RULES = ["sim", "sim_thr", "flt", "cert_sim"]
MNAME = {"sim": "Similarity top-$k$", "sim_thr": "Similarity threshold", "flt": "Detector filter",
         "cert_sim": "Cleared similarity"}
MSHORT = {"sim": "Sim", "sim_thr": "SimThr", "flt": "Flt", "cert_sim": "CSim", "cert_nec": "CNec",
          "nec_thr": "NecThr", "tok": "Tok"}
AN = {"0.1": "A", "0.2": "B", "0.3": "C", "0.5": "D", "0.75": "E", "1.0": "F"}
FIX_MED = "cert_fix2"   # cleared rule at the training median of the proxy score, never calibrated


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


A = {t: load(f"{t}_analysis.json") for t in TASKS}
A14 = load("lamp3_analysis_qwen25_14b.json")
ALLAMA = load("lamp3_analysis_llama31_8b.json")
A300 = load("lamp3_analysis_300users.json")
A2 = load("lamp2_analysis_qwen25-14b-ep18140.json")
BASE = {"three": load("lamp3_baselines_qwen25_7b.json"), "Fourteen": load("lamp3_baselines_qwen25_14b.json")}
HELD = {t: load(f"{t}_heldout.json") for t in TASKS}
PILOT2 = load("lamp2_pilot.json")


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def ci3(v):
    return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"


def pc(v):
    return f"{100*v[0]:.0f}\\% [{100*v[1]:.0f}, {100*v[2]:.0f}]"


def p0(x):
    return f"{100*x:.0f}"


def macros():
    L = []
    seen = {}

    def m(name, val):
        if name in seen:
            raise ValueError(f"Duplicate generated macro: {name}")
        seen[name] = str(val)
        L.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    mean = load("lamp3_mean_rating_summary.json")
    if mean:
        assert mean["n"] == 450
        m("MeanRatingUtility", f3(mean["util"]))
        m("MeanRatingKept", pc(mean["kept_90"]))
        m("MeanRatingDelta", ci3(mean["vs_top5_90"]))
    second_judge = load("lampqa_second_judge_summary.json")
    if second_judge:
        assert second_judge["users"] == 150 and second_judge["answers"] == 3184
        m("SecondJudgeGain", ci3(second_judge["gemma_judge"]["top5_minus_none_95"]))
        m("SecondJudgeQwenGain", ci3(second_judge["qwen_judge"]["top5_minus_none_95"]))
    cumulative = load("lamp3_cumulative_exposure.json")
    if cumulative:
        for name, cell in [("CumulativeSingleOne", "0.5|1|single_prompt_calibration"),
                           ("CumulativeSingleTen", "0.5|10|single_prompt_calibration"),
                           ("CumulativeCalibratedTen", "0.5|10|cumulative_calibration")]:
            m(name, f3(cumulative["cells"][cell]["mean_unique_flagged_records"]))
    for task in TASKS:
        sec = load(f"{task}_secondary_exposure.json")
        for loss, suffix in (("count", "Count"), ("severity2", "Weighted")):
            cell = sec["cells"][f"coarse|{loss}|0.2|all|pooled"]
            m(TKEY[task] + suffix + "Loss", f3(cell["mean_test_loss"]))
            m(TKEY[task] + suffix + "Size", f2(cell["mean_size"]))

    def xfit_macros(k, a, procs=("crc", "ltt"), kept_low=True):
        """Cross-fitted utility macros of one analysis file under prefix k (calibrated on its primary labels)."""
        x = a["xfit"][a["primary"]]
        for al in ("0.2", "0.5"):
            for r in RULES:
                for proc in procs:
                    q = x[proc][al][r]
                    nm = f"{k}{'Ltt' if proc == 'ltt' else ''}{MSHORT[r]}"
                    m(f"{nm}Expo{AN[al]}", f2(q["expo"])); m(f"{nm}Util{AN[al]}", f3(q["util"]))
                    m(f"{nm}Size{AN[al]}", f"{q['size']:.1f}"); m(f"{nm}VsTop{AN[al]}", ci3(q["vs_top5_90"]))
                    m(f"{nm}VsNone{AN[al]}", ci3(q["vs_none"]))
                    if q.get("kept"):
                        m(f"{nm}Kept{AN[al]}", pc(q["kept"]))
                        m(f"{nm}NonInf{AN[al]}", "yes" if q.get("noninferior") else "no")
                        if kept_low:
                            m(f"{nm}KeptLow{AN[al]}", f"{100*q['kept_90'][1]:.0f}\\%")
            m(f"{k}NSigLoss{AN[al]}", sum(x["crc"][al][r]["sig_loss"] for r in RULES))
            m(f"{k}NSigGain{AN[al]}", sum(x["crc"][al][r]["sig_gain_over_top5"] for r in RULES))
            m(f"{k}NNonInf{AN[al]}", sum(bool(x["crc"][al][r].get("noninferior")) for r in RULES))

    def fixed_macros(k, fx):
        m(k + "UNone", f3(fx["none"]["utility"][0])); m(k + "USimK", f3(fx["sim@K"]["utility"][0]))
        m(k + "ESimK", f2(fx["sim@K"]["exposure_union"])); m(k + "ESimKQwen", f2(fx["sim@K"]["exposure_qwen"]))
        m(k + "ESimKAny", p0(fx["sim@K"]["any_union"])); m(k + "ESimKSens", f2(fx["sim@K"]["sens_union"]))
        m(k + "TopFiveGain", ci3(fx["top5_minus_none"]))
        if "rand@K" in fx and "utility" in fx["rand@K"]:
            g = fx["sim@K"]["utility"][0] - fx["none"]["utility"][0]
            m(k + "RandKept", p0((fx["rand@K"]["utility"][0] - fx["none"]["utility"][0]) / g) if g > 1e-9 else "--")
        if FIX_MED in fx:
            q = fx[FIX_MED]
            m(k + "FixMedExpo", f2(q["exposure_union"])); m(k + "FixMedSize", f"{q['size']:.1f}")
            if "utility" in q:
                m(k + "FixMedUtil", f3(q["utility"][0]))

    above_any, above_sens_qwen = [], []
    for t in TASKS:
        a, k = A[t], TKEY[t]
        b, fx, v = a["base"], a["fixed"], a["valid"]
        m(k + "NEval", a["n_eval"]); m(k + "NReader", a["n_reader"])
        m(k + "RateDisc", p0(b["rate_disc"])); m(k + "RateGemma", p0(b["rate_disc_gemma"]))
        m(k + "RateUnion", p0(b["rate_disc_union"])); m(k + "KappaFull", f2(b["kappa_full"]))
        m(k + "RateSensUnion", p0(b["rate_sens_union"]))
        m(k + "QwenRecGemma", p0(b["qwen_recall_of_gemma"]))
        m(k + "UsersAnyUnion", p0(b["users_with_any_disc_union"]))
        m(k + "DetRecallUnion", p0(b["detector_recall_union"]))
        nec = b["gate"]["nec"]
        m(k + "NecSpearman", f2(nec["cv_spearman"][nec["chosen"]]))
        fixed_macros(k, fx)
        c = v["cells"]
        for cs in ("qwen", "union"):
            for ms in ("qwen", "gemma", "union"):
                q = c[f"{cs}|count|{ms}"]
                m(f"{k}Cal{cs.capitalize()}Meas{ms.capitalize()}", p0(q["frac_cells_above"]))
                m(f"{k}Cal{cs.capitalize()}Meas{ms.capitalize()}Worst", f"{q['worst_excess']:+.2f}")
            for loss, nm in (("any", "Any"), ("sens", "Sens")):
                q = c[f"{cs}|{loss}|union"]
                m(f"{k}Cal{cs.capitalize()}MeasUnion{nm}", p0(q["frac_cells_above"]))
                m(f"{k}Cal{cs.capitalize()}MeasUnion{nm}Worst", f"{q['worst_excess']:+.2f}")
            above_any.append(c[f"{cs}|any|union"]["frac_cells_above"])
        above_sens_qwen.append(c["qwen|sens|union"]["frac_cells_above"])
        # mean test loss of cleared similarity under the indicator and special-category losses (union calibration)
        for al in ("0.2", "0.5"):
            m(f"{k}AnyCSim{AN[al]}", f2(v["union"]["any"]["crc"][al]["cert_sim"]["union"]["mean"]))
            m(f"{k}SensCSim{AN[al]}", f2(v["union"]["sens"]["crc"][al]["cert_sim"]["union"]["mean"]))
        xfit_macros(k, a)
        # cost of calibrating on the first auditor only: utility and union-label exposure of what ships
        for al in ("0.2", "0.5"):
            q = a["xfit"]["qwen"]["crc"][al]["cert_sim"]
            m(f"{k}QcalCSimUtil{AN[al]}", f3(q["util"])); m(f"{k}QcalCSimExpo{AN[al]}", f2(q["expo"]))
            if q.get("kept"):
                m(f"{k}QcalCSimKept{AN[al]}", pc(q["kept"]))
        sh = a["shift"]
        m(k + "ShiftMain", f2(sh["main"]["worst_excess"])); m(k + "ShiftNec", f2(sh["necessity"]["worst_excess"]))
        m(k + "ShiftMainFrac", p0(sh["main"]["frac_cells_above"]))
        m(k + "ShiftRand", f2(-sh["main"]["worst_excess_random"]))
        m(k + "MonPooled", f"{sh['pooled_worst_bin_excess']:+.3f}"); m(k + "MonMondrian", f"{sh['mondrian_worst_bin_excess']:+.3f}")
        o = a["oracle"]
        m(k + "OrExpo", f2(o["oracle"]["exposure"][0])); m(k + "OrSuff", p0(o["oracle"]["sufficient"][0]))
        m(k + "OrEmpty", p0(o["oracle"]["empty_sufficient"][0])); m(k + "OrEmptyCI", pc(o["oracle"]["empty_sufficient"]))
        m(k + "OrSimKGap", f2(o["methods"]["sim@K"]["gap"][0]))
        m(k + "OrSimKGapCI", f"{o['methods']['sim@K']['gap'][0]:.2f} [{o['methods']['sim@K']['gap'][1]:.2f}, {o['methods']['sim@K']['gap'][2]:.2f}]")
        m(k + "OrNeeds", p0(o["oracle"]["needs_disclosing"][0])); m(k + "OrNeedsCI", pc(o["oracle"]["needs_disclosing"]))
        m(k + "OrN", o["n"])
        sm = load(f"{t}_shift_mech.json")
        m(k + "ShiftNecNoFeat", f2(sm["without_profile_feature"]["worst_excess"]))
        m(k + "OrSimKExpo", f2(o["methods"]["sim@K"]["exposure"][0]))
        fc = a["fixedcheck"]
        m(k + "FixLttDep", p0(max(fc["ltt"]["0.2"][f]["deploy_rate"] for f in ("sim_top1", "sim_top5", "flt_top5"))))
        m(k + "FixCrcDepTopOne", p0(fc["crc"]["0.2"]["sim_top1"]["deploy_rate"]))
        m(k + "FixCrcTopOneUtil", f3(fc["crc"]["0.2"]["sim_top1"]["util"]))
        m(k + "FixLttDepTopOne", p0(fc["ltt"]["0.2"]["sim_top1"]["deploy_rate"]))
    m("AnyCellsMax", p0(max(above_any)))
    m("SensQwenCellsMax", p0(max(above_sens_qwen)))
    m("CalUnionAllHold", "yes" if all(A[t]["valid"]["cells"][f"union|{l}|{ms}"]["frac_cells_above"] == 0
                                       for t in TASKS for l in ("count", "any", "sens") for ms in ("qwen", "gemma", "union")) else "no")
    lt = [A[t]["valid"]["union"]["count"]["ltt"][al][r]["union"]["frac_above"] for t in TASKS for al in A[t]["valid"]["union"]["count"]["ltt"] for r in RULES]
    m("LttAboveMax", f"{100*max(lt):.1f}")
    cr = [A[t]["valid"]["union"]["count"]["crc"][al][r]["union"]["frac_above"] for t in TASKS for al in A[t]["valid"]["union"]["count"]["crc"] for r in RULES]
    m("CrcAboveMax", p0(max(cr)))
    if A14:
        m("FourteenN", A14["n_reader"]); m("FourteenGain", ci3(A14["fixed"]["top5_minus_none"]))
        m("FourteenNone", f3(A14["fixed"]["none"]["utility"][0])); m("FourteenTop", f3(A14["fixed"]["sim@K"]["utility"][0]))
        fixed_macros("Fourteen", A14["fixed"])
        xfit_macros("Fourteen", A14, procs=("crc",))
    if A300:   # 7B reader on the same 300 users as the 14B reader
        m("SameN", A300["n_reader"]); m("SameGain", ci3(A300["fixed"]["top5_minus_none"]))
        m("SameNone", f3(A300["fixed"]["none"]["utility"][0]))
        q = A300["xfit"][A300["primary"]]["crc"]["0.5"]["cert_sim"]
        if q.get("kept"):
            m("SameCSimKeptD", pc(q["kept"])); m("SameCSimKeptLowD", f"{100*q['kept_90'][1]:.0f}\\%")
            m("SameCSimNonInfD", "yes" if q.get("noninferior") else "no")
    for k, bjs in BASE.items():
        if not bjs:
            continue
        m(k + "BaseN", bjs["n"])
        for v, nm in (("ratings", "Ratings"), ("redact_det", "RedDet"), ("redact_aud", "RedAud")):
            q = bjs[v]
            m(f"{k}Base{nm}Util", f3(q["util"])); m(f"{k}Base{nm}Expo", f2(q["exposure"]))
            m(f"{k}Base{nm}Kept", pc(q["kept"])); m(f"{k}Base{nm}KeptLow", f"{100*q['kept_90'][1]:.0f}\\%")
            m(f"{k}Base{nm}NonInf", "yes" if q.get("noninferior") else "no")
            m(f"{k}Base{nm}VsTop", ci3(q["vs_top5_90"]))
    for t in TASKS:
        h = HELD[t]
        if not h:
            continue
        k = TKEY[t]
        m(k + "HeldN", h["n_users"]); m(k + "HeldRateLlama", p0(h["rate"]["llama70"]))
        m(k + "HeldKappaQL", f2(h["kappa"]["llama70-qwen"] if "llama70-qwen" in h["kappa"] else h["kappa"]["qwen-llama70"]))
        m(k + "HeldKappaGL", f2(h["kappa"]["gemma-llama70"]))
        m(k + "HeldUnionRecLlama", p0(h["union_qg_recall_of_llama70"])); m(k + "HeldLlamaRecUnion", p0(h["llama70_recall_of_union_qg"]))
        r = h["heldout"]
        m(k + "HeldQwenToLlama", p0(r["qwen->llama70"]["frac_cells_above"])); m(k + "HeldQwenToLlamaWorst", f"{r['qwen->llama70']['worst_excess']:+.2f}")
        m(k + "HeldUnionToLlama", p0(r["qwen+gemma->llama70"]["frac_cells_above"])); m(k + "HeldUnionToLlamaWorst", f"{r['qwen+gemma->llama70']['worst_excess']:+.2f}")
        loo = [r[x]["frac_cells_above"] for x in ("qwen+gemma->llama70", "qwen+llama70->gemma", "gemma+llama70->qwen")]
        m(k + "HeldLooMax", p0(max(loo))); m(k + "HeldLooMaxWorst", f"{max(r[x]['worst_excess'] for x in ('qwen+gemma->llama70', 'qwen+llama70->gemma', 'gemma+llama70->qwen')):+.2f}")
    if PILOT2:
        m("twoPilotN", PILOT2["n"]); m("twoPilotGain", ci3(PILOT2["gain"]))
        m("twoPilotNone", f3(PILOT2["acc_none"])); m("twoPilotTop", f3(PILOT2["acc_top5"]))
    if A2:
        k, b, fx = "two", A2["base"], A2["fixed"]
        m(k + "NEval", A2["n_eval"]); m(k + "NReader", A2["n_reader"])
        m(k + "RateDisc", p0(b["rate_disc"])); m(k + "RateGemma", p0(b["rate_disc_gemma"]))
        m(k + "RateUnion", p0(b["rate_disc_union"])); m(k + "KappaFull", f2(b["kappa_full"]))
        m(k + "RateSensUnion", p0(b["rate_sens_union"]))
        fixed_macros(k, fx)
        c = A2["valid"]["cells"]
        for cs in ("qwen", "union"):
            for ms in ("qwen", "gemma", "union"):
                q = c[f"{cs}|count|{ms}"]
                m(f"{k}Cal{cs.capitalize()}Meas{ms.capitalize()}", p0(q["frac_cells_above"]))
                m(f"{k}Cal{cs.capitalize()}Meas{ms.capitalize()}Worst", f"{q['worst_excess']:+.2f}")
        xfit_macros(k, A2, procs=("crc", "ltt"))
    for extra in ("replication.json", "qa_pilot.json"):
        d = load(extra)
        if d:
            for name, val in d.get("macros", {}).items():
                m(name, val)
    if ALLAMA and "threeLlamaN" not in seen:
        m("threeLlamaN", ALLAMA["n_reader"])
    sim = load("sim_crc.json")
    m("SimConfigs", len(sim)); m("SimViol", sum(r["violated"] for r in sim))
    open(os.path.join(GEN, "numbers.tex"), "w").write("\n".join(L) + "\n")


def mark(q):
    s = ""
    if q.get("sig_loss"):
        s += "$^{-}$"
    elif q.get("sig_gain_over_top5"):
        s += "$^{+}$"
    if q.get("noninferior"):
        s += "$^{\\dagger}$"
    return s


def tab_main(supplement=False):
    if supplement:
        cols = [("lamp3_14b", A14, "LaMP-3, 14B", BASE["Fourteen"])]
    else:
        cols = [(t, A[t], TNAME[t], BASE["three"] if t == "lamp3" else None) for t in TASKS]
    n = len(cols)
    rows = ["\\begin{tabular}{l" + "rrr" * n + "}", "\\toprule"]
    rows.append(" & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{nm}}}" for _, _, nm, _ in cols) + " \\\\")
    rows.append(" ".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(n)))
    rows.append("Admission rule & " + " & ".join("Exp. & Rec. & Util." for _ in cols) + " \\\\")
    rows.append("\\midrule")
    rows.append(f"\\multicolumn{{{1+3*n}}}{{l}}{{\\emph{{Uncalibrated, whole records}}}} \\\\")
    for key, name in (("none", "No profile"), ("sim@K", "Similarity top-5"), ("flt@K", "Detector filter top-5")):
        line = [name]
        for _, a, _, _ in cols:
            fx = a["fixed"][key]
            line.append(f"{f2(fx['exposure_union'])} & {fx['size']:.1f} & {f3(fx['utility'][0])}")
        rows.append(" & ".join(line) + " \\\\")
    if any(bjs for _, _, _, bjs in cols):
        rows.append("\\midrule")
        rows.append(f"\\multicolumn{{{1+3*n}}}{{l}}{{\\emph{{Uncalibrated, rating summaries or text removal}}}} \\\\")
        for v, name in (("ratings", "Ratings only"), ("redact_det", "Redact detector-flagged text"),
                        ("redact_aud", "Redact auditor-flagged text")):
            line = [name]
            for _, _, _, bjs in cols:
                if bjs:
                    q = bjs[v]
                    line.append(f"{f2(q['exposure'])} & 5.0 & {f3(q['util'])}{mark(q)}")
                else:
                    line.append("-- & -- & --")
            rows.append(" & ".join(line) + " \\\\")
    mean = load("lamp3_mean_rating_summary.json")
    if mean and not supplement:
        assert mean["n"] == 450
        rows.append("Mean past rating only & -- & -- & -- & 0.00 & 0.0 & "
                    + f"{f3(mean['util'])}{mark(mean)}" + " & -- & -- & -- \\\\")
    for proc, title in (("crc", "Calibrated on union labels, $\\mathbb{E}$-bound at $\\alpha=0.2$ (Eq.~2)"),
                        ("ltt", "Calibrated on union labels, per-draw bound at $\\alpha=0.2$, $\\delta=0.1$")):
        if supplement:
            title = ("Union calibration: CRC, $\\alpha=0.2$" if proc == "crc" else
                     "Union calibration: LTT, $\\alpha=0.2$, $\\delta=0.1$")
        rows.append("\\midrule")
        rows.append(f"\\multicolumn{{{1+3*n}}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for r in RULES:
            line = [MNAME[r]]
            for _, a, _, _ in cols:
                q = a["xfit"][a["primary"]][proc]["0.2"][r]
                line.append(f"{f2(q['expo'])} & {q['size']:.1f} & {f3(q['util'])}{mark(q)}")
            rows.append(" & ".join(line) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(GEN, "tab_14b.tex" if supplement else "tab_main.tex"), "w").write("\n".join(rows) + "\n")


def tab_audit():
    held = any(HELD[t] for t in TASKS)
    ncol = 4 if held else 3
    rows = ["\\begin{tabular}{ll" + "r" * ncol + "}", "\\toprule"]
    rows.append(f" & Calibrated on & \\multicolumn{{{ncol}}}{{c}}{{Measured with}} \\\\")
    rows.append(f"\\cmidrule(lr){{3-{2+ncol}}}")
    rows.append("Task & & Qwen & Gemma & Union" + (" & Llama-70B" if held else "") + " \\\\")
    rows.append("\\midrule")
    for t in TASKS:
        c = A[t]["valid"]["cells"]
        for i, cs in enumerate(("qwen", "union")):
            vals = []
            for ms in ("qwen", "gemma", "union"):
                q = c[f"{cs}|count|{ms}"]
                vals.append(f"{100*q['frac_cells_above']:.0f}\\% ({q['worst_excess']:+.2f})")
            if held:
                h = HELD[t]
                if h:
                    q = h["heldout"]["qwen->llama70" if cs == "qwen" else "qwen+gemma->llama70"]
                    vals.append(f"{100*q['frac_cells_above']:.0f}\\% ({q['worst_excess']:+.2f})")
                else:
                    vals.append("--")
            rows.append(f"{TNAME[t] if i == 0 else ''} & {cs.capitalize()} & " + " & ".join(vals) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(GEN, "tab_audit.tex"), "w").write("\n".join(rows) + "\n")


def fig_main():
    pf.apply_style()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7})
    col = {"cert_sim": pf.BLUE, "sim": pf.ORANGE, "sim_thr": pf.VIOLET, "flt": pf.MAGENTA}
    mk = {"cert_sim": "s", "sim": "^", "sim_thr": "D", "flt": "P"}
    lab = {"cert_sim": "cleared similarity", "sim": "similarity top-k", "sim_thr": "similarity threshold",
           "flt": "detector filter"}
    n = len(TASKS)
    fig, ax = plt.subplots(1, 1 + n, figsize=(7.0, 1.55))
    a0 = ax[0]
    a0.plot([0, 1.0], [0, 1.0], color="black", lw=0.8, ls="--")
    for t, mkr, c in zip(TASKS, ("o", "s", "^"), (pf.BLUE, pf.ORANGE, pf.AQUA)):
        for cs, ls in (("qwen", ":"), ("union", "-")):
            d = A[t]["valid"][cs]["count"]["crc"]
            xs = [float(a) for a in d]
            hi = [max(d[a][r]["union"]["mean"] for r in RULES) for a in d]
            a0.plot(xs, hi, color=c, marker=mkr, lw=1.1, ms=3.2, ls=ls, label=TNAME[t] if cs == "union" else None)
    a0.set_xlabel(r"target $\alpha$"); a0.set_ylabel("exposure (union labels)")
    a0.set_title("(a) calibrated on Qwen (dotted)\nor on the union (solid)", fontsize=7.5)
    # Put task labels outside the data area: high-exposure curves occupy its top left.
    h0, l0 = a0.get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower left", ncol=1, bbox_to_anchor=(0.005, -0.19),
               handletextpad=0.3, handlelength=1.6, labelspacing=0.12, fontsize=6.5)
    a0.grid(True)
    for i, t in enumerate(TASKS):
        ai = ax[1 + i]
        x = A[t]["xfit"][A[t]["primary"]]["crc"]
        xs = [float(a) for a in x]
        for r in RULES:
            ai.plot(xs, [x[a][r]["util"] for a in x], color=col[r], marker=mk[r], label=lab[r],
                    lw=1.6 if r == "cert_sim" else 1.0, ms=3.2)
        ai.fill_between(xs, [x[a]["cert_sim"]["util_ci"][1] for a in x], [x[a]["cert_sim"]["util_ci"][2] for a in x],
                        color=col["cert_sim"], alpha=0.15, lw=0)
        fx = A[t]["fixed"]
        ai.axhline(fx["none"]["utility"][0], color="black", lw=0.8, ls=":")
        ai.axhline(fx["sim@K"]["utility"][0], color="black", lw=0.8, ls="--")
        ai.set_xlabel(r"target $\alpha$")
        if i == 0:
            ai.set_ylabel("utility")
        ai.set_title(f"({'bcd'[i]}) {TNAME[t]}"); ai.grid(True)
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.6, -0.08), handlelength=1.8)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(GEN, "fig_main.pdf"))


if __name__ == "__main__":
    macros(); tab_main(); tab_audit(); fig_main()
    if A14:
        tab_main(supplement=True)
    print("assets written; 14B:", bool(A14), "baselines:", {k: bool(v) for k, v in BASE.items()},
          "heldout:", {t: bool(HELD[t]) for t in TASKS}, "lamp2:", bool(A2), "7B@300:", bool(A300))
