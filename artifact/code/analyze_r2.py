"""Calibration, validity and matched-exposure utility analysis (revision 2). Pure numpy.

  python analyze.py TASK [READER_TAG]

Writes results/{task}_analysis[_READER].json. Sections:
  base     : auditor base rates, scorer quality, detector agreement
  fixed    : uncertified operating points on all reader users (bootstrap CIs)
  valid    : per loss / label source / rule / alpha, realized test loss over random half splits
             (calibrated on the primary auditor's labels), CRC and Learn-then-Test
  xfit     : 2-fold cross-fitted utility, exposure and records admitted per rule and alpha,
             with user-bootstrap 95% and 90% intervals for differences to top-5 and no profile
  union    : cross-fitted results when calibrating on the union of both auditors
  shift    : calibrate on short-profile users, test on long-profile users and the reverse
  oracle   : minimality gap on the oracle subsample
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from common import K, RES, SEED, SENSITIVE_CATS, load_proc
from crc import crc_index, hb_pvalue, ltt_index, monotonize

ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
CAL = ["cert_nec", "cert_sim", "sim", "nec", "tok", "flt", "nec_thr", "sim_thr"]
N_SPLITS = 1000
N_PART = 200
N_BOOT = 500
DELTA = 0.1
MARGIN = 0.03          # equivalence margin, declared in PLAN.md before the 300-user results
FLAG_KEYS = ["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "NRP",
             "lex_health", "lex_finance", "lex_family", "lex_belief", "lex_place"]
LOSSES = {"count": K, "any": 1, "sens": K}   # loss name -> bound B


def pct(v, lo=2.5, hi=97.5):
    v = np.asarray(v, dtype=float)
    return [float(np.mean(v)), float(np.percentile(v, lo)), float(np.percentile(v, hi))]


def main(task, rtag="qwen25_7b"):
    users = [u for u in load_proc(task)["users"] if u["split"] == "eval"]
    rules = json.load(open(os.path.join(RES, f"{task}_rules.json")))
    audit = {"qwen": json.load(open(os.path.join(RES, f"{task}_audit.json")))}
    pg = os.path.join(RES, f"{task}_audit_gemma2_9b_evaltop.json")
    if os.path.exists(pg):
        audit["gemma"] = json.load(open(pg))
    det = json.load(open(os.path.join(RES, f"{task}_detect.json")))
    util = json.load(open(os.path.join(RES, f"{task}_reader_utility_{rtag}.json")))
    users = [u for u in users if u["uid"] in rules and u["uid"] in audit["qwen"]]
    if "gemma" in audit:
        users = [u for u in users if u["uid"] in audit["gemma"]]
    grid = json.load(open(os.path.join(RES, f"{task}_thr_grid.json")))
    for u in users:
        r = rules[u["uid"]]
        P = len(r["g"])
        sims = [u["cands"][ci]["sim"] for ci in range(P)]
        gs = sorted(r["g"], reverse=True)
        r["sets"]["nec_thr"] = [r["sets"]["nec"][min(K, sum(x >= t for x in gs))] for t in grid["nec"]]
        r["sets"]["sim_thr"] = [r["sets"]["sim"][min(K, sum(x >= t for x in sims))] for t in grid["sim"]]
    N = len(users)
    has_u = np.array([u["uid"] in util for u in users])
    ridx = np.nonzero(has_u)[0]
    out = dict(task=task, reader=rtag, n_eval=N, n_reader=int(has_u.sum()), K=K, alphas=ALPHAS,
               labels=sorted(audit), n_part=N_PART, n_boot=N_BOOT, n_splits=N_SPLITS, delta=DELTA, margin=MARGIN)

    def key(ids):
        return ",".join(map(str, sorted(ids)))

    def flag(src, uid, ci, loss):
        if src == "union":
            return max(flag("qwen", uid, ci, loss), flag("gemma", uid, ci, loss))
        c = audit[src][uid][str(ci)]
        return int(any(x in SENSITIVE_CATS for x in c)) if loss == "sens" else int(len(c) > 0)

    def loss_of(src, uid, ids, loss):
        e = sum(flag(src, uid, i, loss) for i in ids)
        return min(e, 1) if loss == "any" else e

    srcs = list(audit) + (["union"] if "gemma" in audit else [])
    S = {m: [rules[u["uid"]]["sets"][m] for u in users] for m in CAL}
    L = {(src, loss, m): np.array([[loss_of(src, u["uid"], s, loss) for s in S[m][i]] for i, u in enumerate(users)], dtype=float)
         for src in srcs for loss in LOSSES for m in CAL}
    Lm = {k: monotonize(v) for k, v in L.items()}
    SIZE = {m: np.array([[len(s) for s in S[m][i]] for i in range(N)], dtype=float) for m in CAL}
    UT = {m: np.array([[util[u["uid"]][key(s)] if h else np.nan for s in S[m][i]] for i, (u, h) in enumerate(zip(users, has_u))])
          for m in CAL}
    u_none = UT["sim"][:, 0]
    u_top5 = UT["sim"][:, K]

    # ---------------- base rates and scorer quality ------------------------------------
    y, s, fl, ys, yg = [], [], [], [], []
    for u in users:
        for ci, sv in enumerate(rules[u["uid"]]["s"]):
            y.append(flag("qwen", u["uid"], ci, "count")); ys.append(flag("qwen", u["uid"], ci, "sens")); s.append(sv)
            fl.append(int(any(det[u["uid"]][str(ci)][k] > 0 for k in FLAG_KEYS)))
            if "gemma" in audit:
                yg.append(flag("gemma", u["uid"], ci, "count"))
    y, s, fl, ys = map(np.array, (y, s, fl, ys))
    base = dict(n_records=int(len(y)), rate_disc=float(y.mean()), rate_sens=float(ys.mean()),
                users_with_any_disc=float(np.mean([any(flag("qwen", u["uid"], ci, "count") for ci in range(len(rules[u["uid"]]["s"]))) for u in users])),
                proxy_auc=float(roc_auc_score(y, s)), detector_flag_rate=float(fl.mean()),
                detector_recall=float(fl[y == 1].mean()), detector_precision=float(y[fl == 1].mean()),
                gate=json.load(open(os.path.join(RES, f"{task}_gate_report.json"))))
    if yg:
        yg = np.array(yg)
        po = np.mean(y == yg); pe = y.mean() * yg.mean() + (1 - y.mean()) * (1 - yg.mean())
        base.update(rate_disc_gemma=float(yg.mean()), rate_disc_union=float(np.maximum(y, yg).mean()),
                    kappa_full=float((po - pe) / (1 - pe)), qwen_recall_of_gemma=float(y[yg == 1].mean()),
                    gemma_recall_of_qwen=float(yg[y == 1].mean()), proxy_auc_gemma=float(roc_auc_score(yg, s)),
                    detector_recall_union=float(fl[np.maximum(y, yg) == 1].mean()))
    out["base"] = base

    # ---------------- uncertified operating points (all reader users) ------------------
    rng = np.random.default_rng(SEED)

    def boot_mean(v):
        v = np.asarray(v, dtype=float)
        b = pct([np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)])
        return [float(np.mean(v)), b[1], b[2]]

    tokk = np.array([rules[u["uid"]]["sets"]["tok_native_k"] for u in users])
    fixed = {}

    def put(name, idx_sets, exp_src="qwen"):
        e = np.array([loss_of(exp_src, users[i]["uid"], idx_sets[i], "count") for i in ridx], dtype=float)
        a = np.array([loss_of(exp_src, users[i]["uid"], idx_sets[i], "any") for i in ridx], dtype=float)
        uu = np.array([util[users[i]["uid"]][key(idx_sets[i])] for i in ridx])
        fixed[name] = dict(exposure=boot_mean(e), any=float(a.mean()), utility=boot_mean(uu),
                           size=float(np.mean([len(idx_sets[i]) for i in ridx])))
        return uu

    put("none", [[] for _ in users])
    for m in ("sim", "nec", "flt", "rand"):
        put(m + "@K", [rules[u["uid"]]["sets"][m][K] for u in users])
    put("tok@native", [rules[u["uid"]]["sets"]["tok"][tokk[i]] for i, u in enumerate(users)])
    uf = np.array([util[users[i]["uid"]][key(rules[users[i]["uid"]]["sets"]["full"][0])] for i in ridx])
    fixed["full"] = dict(exposure=None, utility=boot_mean(uf), size=float(np.mean([len(rules[users[i]["uid"]]["sets"]["full"][0]) for i in ridx])))
    d = u_top5[ridx] - u_none[ridx]
    fixed["top5_minus_none"] = boot_mean(d)
    fixed["top5_minus_none_90"] = pct([np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)], 5, 95)
    if "gemma" in audit:
        for m in ("sim", "flt"):
            e = np.array([loss_of(src, users[i]["uid"], rules[users[i]["uid"]]["sets"][m][K], "count") for i in ridx for src in ["gemma"]])
            fixed[m + "@K"]["exposure_gemma"] = float(e.mean())
            e = np.array([loss_of("union", users[i]["uid"], rules[users[i]["uid"]]["sets"][m][K], "count") for i in ridx])
            fixed[m + "@K"]["exposure_union"] = float(e.mean())
    out["fixed"] = fixed

    # ---------------- validity over random half splits ---------------------------------
    rng = np.random.default_rng(SEED + 1)
    n_cal = N // 2
    valid = {}
    meas_srcs = srcs
    acc = {}
    for _ in range(N_SPLITS):
        perm = rng.permutation(N)
        cal, test = perm[:n_cal], perm[n_cal:]
        for loss, B in LOSSES.items():
            for m in CAL:
                for a in ALPHAS:
                    j = crc_index(Lm[("qwen", loss, m)][cal], a, B)
                    jl = ltt_index(Lm[("qwen", loss, m)][cal], a, B, DELTA) if loss != "sens" else None
                    for src in meas_srcs:
                        acc.setdefault((loss, m, a, src, "crc"), []).append(L[(src, loss, m)][test, j].mean())
                        if jl is not None:
                            acc.setdefault((loss, m, a, src, "ltt"), []).append(L[(src, loss, m)][test, jl].mean())
                    acc.setdefault((loss, m, a, "size", "crc"), []).append(SIZE[m][test, j].mean())
                    if jl is not None:
                        acc.setdefault((loss, m, a, "size", "ltt"), []).append(SIZE[m][test, jl].mean())
    for (loss, m, a, src, proc), v in acc.items():
        v = np.array(v)
        valid.setdefault(loss, {}).setdefault(proc, {}).setdefault(str(a), {}).setdefault(m, {})[src] = dict(
            mean=float(v.mean()), sd=float(v.std()), frac_above=float(np.mean(v > a)) if src != "size" else None)
    out["valid"] = valid

    # ---------------- what calibration adds over auditing one fixed rule --------------
    # A deployer audits n calibration users for a FIXED rule and deploys it only if a bound on
    # its loss passes (expectation check of Eq. 2 on a one-point grid, or a Hoeffding-Bentkus
    # test at level delta); otherwise it falls back to no profile. Compared on the same random
    # half splits with the calibrated threshold rule and the calibrated cleared rule.
    rng = np.random.default_rng(SEED + 6)
    fixed_sets = {"tok_native": [rules[u["uid"]]["sets"]["tok"][tokk[i]] for i, u in enumerate(users)],
                  "sim_top1": [rules[u["uid"]]["sets"]["sim"][1] for u in users],
                  "sim_top5": [rules[u["uid"]]["sets"]["sim"][K] for u in users]}
    FL = {(f, loss): np.array([loss_of("qwen", u["uid"], fixed_sets[f][i], loss) for i, u in enumerate(users)], dtype=float)
          for f in fixed_sets for loss in ("count", "any")}
    FU = {f: np.array([util[u["uid"]][key(fixed_sets[f][i])] if h else np.nan for i, (u, h) in enumerate(zip(users, has_u))])
          for f in fixed_sets}
    FS = {f: np.array([len(s) for s in fixed_sets[f]], dtype=float) for f in fixed_sets}
    fc = {}
    for _ in range(500):
        perm = rng.permutation(N)
        cal, test = perm[:n_cal], perm[n_cal:]
        tr = test[has_u[test]]
        for loss in ("count", "any"):
            B = LOSSES[loss]
            for a in ALPHAS:
                for f in fixed_sets:
                    r = FL[(f, loss)][cal].mean()
                    for proc, ok in (("exp", (n_cal * r + B) / (n_cal + 1) <= a),
                                     ("hb", hb_pvalue(r / B, n_cal, a / B) <= DELTA)):
                        u_ = np.nanmean(FU[f][tr]) if ok else np.nanmean(u_none[tr])
                        e_ = FL[(f, loss)][test].mean() if ok else 0.0
                        z_ = FS[f][test].mean() if ok else 0.0
                        fc.setdefault((loss, str(a), f, proc), []).append((float(ok), u_, e_, z_))
                for m in ("sim_thr", "cert_sim", "cert_nec"):
                    for proc in ("exp", "hb"):
                        j = crc_index(Lm[("qwen", loss, m)][cal], a, B) if proc == "exp" else ltt_index(Lm[("qwen", loss, m)][cal], a, B, DELTA)
                        fc.setdefault((loss, str(a), m, proc), []).append(
                            (1.0, np.nanmean(UT[m][tr, j]), L[("qwen", loss, m)][test, j].mean(), SIZE[m][test, j].mean()))
    fco = {}
    for (loss, a, f, proc), v in fc.items():
        v = np.array(v)
        fco.setdefault(loss, {}).setdefault(proc, {}).setdefault(a, {})[f] = dict(
            deploy_rate=float(v[:, 0].mean()), util=float(v[:, 1].mean()), loss=float(v[:, 2].mean()), size=float(v[:, 3].mean()))
    out["fixedcheck"] = fco

    # ---------------- cross-fitted utility ---------------------------------------------
    def crossfit(perm_users, labels_src="qwen", loss="count", rules_=CAL, alphas=ALPHAS):
        """perm_users: array of user indices (may contain duplicates); returns per-(m,a)
        arrays of per-user test utility / exposure / size under 2-fold cross-fitting."""
        uniq = np.unique(perm_users)
        side = {}
        sh = rng.permutation(len(uniq))
        for t, i in enumerate(uniq[sh]):
            side[i] = t % 2
        fold = np.array([side[i] for i in perm_users])
        res = {}
        for m in rules_:
            for a in alphas:
                jj = np.zeros(len(perm_users), dtype=int)
                for f in (0, 1):
                    cal = perm_users[fold != f]
                    jj[fold == f] = crc_index(Lm[(labels_src, loss, m)][cal], a, LOSSES[loss])
                idx = perm_users
                res[(m, a)] = (UT[m][idx, jj], L[(labels_src, "count", m)][idx, jj], SIZE[m][idx, jj],
                               L[(labels_src, "any", m)][idx, jj])
        return res

    rng = np.random.default_rng(SEED + 2)
    allu = np.arange(N)
    pt = {}
    for _ in range(N_PART):
        r = crossfit(allu)
        for (m, a), (uu, ee, zz, an) in r.items():
            pt.setdefault((m, a), []).append((np.nanmean(uu), ee.mean(), zz.mean(), an.mean()))
    xfit = {}
    for (m, a), v in pt.items():
        v = np.array(v)
        xfit.setdefault(str(a), {})[m] = dict(util=float(v[:, 0].mean()), expo=float(v[:, 1].mean()),
                                              size=float(v[:, 2].mean()), any=float(v[:, 3].mean()))
    # bootstrap: resample users, one random 2-fold partition per replicate
    bs = {}
    for _ in range(N_BOOT):
        b = rng.integers(0, N, N)
        r = crossfit(b)
        hb = has_u[b]
        un, ut = np.nanmean(u_none[b][hb]), np.nanmean(u_top5[b][hb])
        bs.setdefault("none", []).append(un); bs.setdefault("top5", []).append(ut)
        for (m, a), (uu, ee, zz, an) in r.items():
            um = np.nanmean(uu[hb])
            bs.setdefault((m, a), []).append(um)
    for a in ALPHAS:
        for m in CAL:
            v = np.array(bs[(m, a)])
            dt, dn = v - np.array(bs["top5"]), v - np.array(bs["none"])
            gain = np.array(bs["top5"]) - np.array(bs["none"])
            x = xfit[str(a)][m]
            x.update(util_ci=pct(v), vs_top5=pct(dt), vs_top5_90=pct(dt, 5, 95), vs_none=pct(dn),
                     vs_none_90=pct(dn, 5, 95),
                     equiv_top5=bool(pct(dt, 5, 95)[1] > -MARGIN and pct(dt, 5, 95)[2] < MARGIN),
                     noninferior_top5=bool(pct(dt, 5, 95)[1] > -MARGIN),
                     retained=pct(dn[gain > 1e-9] / gain[gain > 1e-9]) if np.any(gain > 1e-9) else None)
        for m1 in CAL:
            for m2 in CAL:
                if m1 < m2:
                    dd = np.array(bs[(m1, a)]) - np.array(bs[(m2, a)])
                    xfit[str(a)].setdefault("pairs", {})[f"{m1}-{m2}"] = pct(dd)
    out["xfit"] = xfit

    # ---------------- recalibration on the union of both auditors ----------------------
    if "gemma" in audit:
        rng = np.random.default_rng(SEED + 3)
        uni = {}
        for _ in range(N_PART // 2):
            r = crossfit(allu, labels_src="union", alphas=[0.2, 0.5])
            for (m, a), (uu, ee, zz, an) in r.items():
                uni.setdefault((m, a), []).append((np.nanmean(uu), ee.mean(), zz.mean()))
        out["union"] = {str(a): {m: dict(zip(("util", "expo_union", "size"), np.array(uni[(m, a)]).mean(0).tolist()))
                                 for m in CAL} for a in (0.2, 0.5)}

    # ---------------- shift: short-profile <-> long-profile users ----------------------
    npro = np.array([u["n_profile"] for u in users])
    short = npro <= np.median(npro)
    shift = {}
    rng = np.random.default_rng(SEED + 5)
    for name, cal, test in (("short_to_long", short, ~short), ("long_to_short", ~short, short),
                            ("random", None, None)):
        acc_s = {}
        for _ in range(300):
            if name == "random":     # same sizes without shift, for reference
                perm = rng.permutation(N)
                c_idx, t_idx = perm[:100], perm[N // 2:]
            else:
                c_idx = rng.choice(np.nonzero(cal)[0], 100, replace=False)
                t_idx = np.nonzero(test)[0]
            for m in CAL:
                for a_ in ALPHAS:
                    j = crc_index(Lm[("qwen", "count", m)][c_idx], a_, K)
                    acc_s.setdefault((m, a_), []).append(L[("qwen", "count", m)][t_idx, j].mean())
        for (m, a_), v in acc_s.items():
            shift.setdefault(name, {}).setdefault(str(a_), {})[m] = float(np.mean(v))
    shift["median_profile"] = float(np.median(npro))
    shift["worst_excess"] = max(v - float(a_) for nm in ("short_to_long", "long_to_short") for a_, d in shift[nm].items() for v in d.values())
    shift["worst_excess_random"] = max(v - float(a_) for a_, d in shift["random"].items() for v in d.values())
    shift["frac_cells_above"] = float(np.mean([v > float(a_) for nm in ("short_to_long", "long_to_short") for a_, d in shift[nm].items() for v in d.values()]))
    out["shift"] = shift

    # ---------------- oracle / minimality gap -------------------------------------------
    p = os.path.join(RES, f"{task}_oracle.json")
    if os.path.exists(p) and rtag == "qwen25_7b":
        orc = json.load(open(p))
        eps = orc["eps"]
        by = {r["uid"]: r for r in orc["users"]}
        idx = [i for i, u in enumerate(users) if u["uid"] in by and has_u[i]]
        o = [by[users[i]["uid"]] for i in idx]
        oe = np.array([r["exposure"] for r in o], dtype=float)
        uref = np.array([r["u_ref"] for r in o])
        rng = np.random.default_rng(SEED + 4)

        def ci2(v):
            v = np.asarray(v, dtype=float)
            b = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)]
            return [float(np.mean(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

        g = dict(n=len(o), eps=eps, max_records=orc.get("max_steps"),
                 oracle=dict(exposure=ci2(oe), size=ci2([len(r["S"]) for r in o]),
                             sufficient=ci2([r["sufficient"] for r in o]), utility=ci2([r["u"] for r in o]),
                             empty_sufficient=ci2([r["u_none"] >= r["u_ref"] - eps for r in o]),
                             needs_disclosing=ci2([r["exposure"] > 0 for r in o]),
                             mean_calls=float(np.mean([r["calls"] for r in o]))),
                 methods={})
        for m in ("sim", "nec", "flt"):
            e = L[("qwen", "count", m)][idx, K]
            g["methods"][m + "@K"] = dict(exposure=ci2(e), sufficient=ci2(UT[m][idx, K] >= uref - eps), gap=ci2(e - oe))
        rest = np.array([i for i in range(N) if i not in set(idx)])
        for m in ("cert_nec", "cert_sim", "sim", "nec_thr", "sim_thr", "tok", "flt"):
            for a in ALPHAS:
                j = crc_index(Lm[("qwen", "count", m)][rest], a, K)   # calibrated outside the oracle sample
                e = L[("qwen", "count", m)][idx, j]
                g["methods"][f"{m}@{a}"] = dict(exposure=ci2(e), idx=int(j), sufficient=ci2(UT[m][idx, j] >= uref - eps),
                                                utility=ci2(UT[m][idx, j]), gap=ci2(e - oe))
        out["oracle"] = g

    sfx = "" if rtag == "qwen25_7b" else "_" + rtag
    json.dump(out, open(os.path.join(RES, f"{task}_analysis{sfx}.json"), "w"), indent=1)
    print(task, "N", N, "reader", int(has_u.sum()), "labels", srcs)
    print("fixed top5-none", fixed["top5_minus_none"])
    for a in ("0.2", "0.5"):
        print(a, {m: (round(v["util"], 3), round(v["expo"], 2), round(v["size"], 1), [round(x, 3) for x in v["vs_top5_90"]])
                  for m, v in out["xfit"][a].items() if m != "pairs"})
    print("valid worst crc", max(v[m]["qwen"]["mean"] - float(a) for a, v in valid["count"]["crc"].items() for m in CAL))
    print("ltt", {a: (round(v["cert_nec"]["qwen"]["mean"], 3), round(v["cert_nec"]["size"]["mean"], 2), v["cert_nec"]["qwen"]["frac_above"])
                  for a, v in valid["count"]["ltt"].items()})
    print("shift worst excess", shift["worst_excess"])


if __name__ == "__main__":
    main(*sys.argv[1:])
