"""Calibration, validity and matched-exposure utility analysis (revision 3). Pure numpy.

  python analyze.py TASK [READER_TAG]

Writes results/{task}_analysis[_READER].json. Sections:
  base     : auditor base rates and agreement, scorer quality, detector recall
  fixed    : uncalibrated operating points on reader users (bootstrap CIs)
  valid    : realized test loss over random half splits, for every calibration label source
             (qwen, union) and measurement label source (qwen, gemma, union), CRC and LTT
  xfit     : 2-fold cross-fitted utility / exposure / records on reader users, calibrated on the
             primary labels (union), CRC and LTT; share of the top-5 gain kept; non-inferiority
             with a relative margin (25% of the gain) and significant losses
  fixedcheck: auditing one fixed rule with a bound versus calibrating a rule
  shift    : profile-length shift, Mondrian (per bin) calibration, random reference
  oracle   : minimality gap and how often the empty set suffices
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from common import K, RES, SEED, SENSITIVE_CATS, load_proc
from crc import crc_index, hb_pvalue, ltt_index, monotonize

ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
MAIN = ["sim", "sim_thr", "flt", "cert_sim"]          # rules in the paper's tables
EXTRA = ["cert_nec", "nec_thr", "tok"]                 # rules that use the near-random necessity score
N_SPLITS = 1000
N_PART = 200
N_BOOT = 500
DELTA = 0.1
REL_MARGIN = 0.25      # declared in PLAN.md (round-2 plan) before the 14B results
FLAG_KEYS = ["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "NRP",
             "lex_health", "lex_finance", "lex_family", "lex_belief", "lex_place"]
LOSSES = {"count": K, "any": 1, "sens": K}


def pct(v, lo=2.5, hi=97.5):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
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
    users = [u for u in users if u["uid"] in rules and all(u["uid"] in a for a in audit.values())]
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

    def key(ids):
        return ",".join(map(str, sorted(ids)))

    def avail(m):
        return all(key(s) in util[users[i]["uid"]] for i in ridx for s in rules[users[i]["uid"]]["sets"][m])

    CAL = [m for m in MAIN + EXTRA if avail(m)]
    PRIMARY = "union" if "gemma" in audit else "qwen"
    srcs = list(audit) + (["union"] if "gemma" in audit else [])
    out = dict(task=task, reader=rtag, n_eval=N, n_reader=int(has_u.sum()), K=K, alphas=ALPHAS, rules=CAL,
               primary=PRIMARY, labels=srcs, n_part=N_PART, n_boot=N_BOOT, n_splits=N_SPLITS, delta=DELTA,
               rel_margin=REL_MARGIN)

    def flag(src, uid, ci, special=False):
        if src == "union":
            return max(flag("qwen", uid, ci, special), flag("gemma", uid, ci, special))
        c = audit[src][uid][str(ci)]
        return int(any(x in SENSITIVE_CATS for x in c)) if special else int(len(c) > 0)

    def loss_of(src, uid, ids, loss):
        e = sum(flag(src, uid, i, loss == "sens") for i in ids)
        return min(e, 1) if loss == "any" else e

    S = {m: [rules[u["uid"]]["sets"][m] for u in users] for m in CAL}
    L = {(src, loss, m): np.array([[loss_of(src, u["uid"], s, loss) for s in S[m][i]] for i, u in enumerate(users)], dtype=float)
         for src in srcs for loss in LOSSES for m in CAL}
    Lm = {k: monotonize(v) for k, v in L.items()}
    SIZE = {m: np.array([[len(s) for s in S[m][i]] for i in range(N)], dtype=float) for m in CAL}
    UT = {m: np.array([[util[u["uid"]][key(s)] if h else np.nan for s in S[m][i]] for i, (u, h) in enumerate(zip(users, has_u))])
          for m in CAL}
    u_none = UT["sim"][:, 0]
    u_top5 = UT["sim"][:, K]

    # ---------------- base rates ---------------------------------------------------------
    y, s, fl, yg = [], [], [], []
    for u in users:
        for ci, sv in enumerate(rules[u["uid"]]["s"]):
            y.append(flag("qwen", u["uid"], ci)); s.append(sv)
            fl.append(int(any(det[u["uid"]][str(ci)][k] > 0 for k in FLAG_KEYS)))
            if "gemma" in audit:
                yg.append(flag("gemma", u["uid"], ci))
    y, s, fl = map(np.array, (y, s, fl))
    base = dict(n_records=int(len(y)), rate_disc=float(y.mean()),
                users_with_any_disc=float(np.mean([any(flag("qwen", u["uid"], ci) for ci in range(len(rules[u["uid"]]["s"]))) for u in users])),
                proxy_auc=float(roc_auc_score(y, s)), detector_recall=float(fl[y == 1].mean()),
                detector_precision=float(y[fl == 1].mean()),
                gate=json.load(open(os.path.join(RES, f"{task}_gate_report.json"))))
    ysn = np.array([flag("qwen", u["uid"], ci, True) for u in users for ci in range(len(rules[u["uid"]]["s"]))])
    base["rate_sens"] = float(ysn.mean())
    if yg:
        ysu = np.array([flag("union", u["uid"], ci, True) for u in users for ci in range(len(rules[u["uid"]]["s"]))])
        base["rate_sens_union"] = float(ysu.mean())
        yg = np.array(yg); yu = np.maximum(y, yg)
        po = np.mean(y == yg); pe = y.mean() * yg.mean() + (1 - y.mean()) * (1 - yg.mean())
        base.update(rate_disc_gemma=float(yg.mean()), rate_disc_union=float(yu.mean()),
                    kappa_full=float((po - pe) / (1 - pe)), qwen_recall_of_gemma=float(y[yg == 1].mean()),
                    gemma_recall_of_qwen=float(yg[y == 1].mean()), proxy_auc_union=float(roc_auc_score(yu, s)),
                    detector_recall_union=float(fl[yu == 1].mean()),
                    users_with_any_disc_union=float(np.mean([any(flag("union", u["uid"], ci) for ci in range(len(rules[u["uid"]]["s"]))) for u in users])))
    out["base"] = base

    # ---------------- uncalibrated operating points (reader users) ----------------------
    rng = np.random.default_rng(SEED)

    def boot(v):
        v = np.asarray(v, dtype=float)
        b = pct([np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)])
        return [float(np.mean(v)), b[1], b[2]]

    fixed = {}
    refs = {"none": [[] for _ in users], "sim@K": [rules[u["uid"]]["sets"]["sim"][K] for u in users],
            "flt@K": [rules[u["uid"]]["sets"]["flt"][K] for u in users]}
    if all(key(rules[users[i]["uid"]]["sets"]["rand"][K]) in util[users[i]["uid"]] for i in ridx):
        refs["rand@K"] = [rules[u["uid"]]["sets"]["rand"][K] for u in users]
    # MINIM-style fixed sensitivity thresholds: the cleared rule at a hand-set grid point (training
    # quantile of the proxy score), never calibrated; shows what one fixed threshold admits per task
    gq = base["gate"]["grid"]
    for j in range(1, len(gq) - 1):
        refs[f"cert_fix{j}"] = [rules[u["uid"]]["sets"]["cert_sim"][j] for u in users]
    for name, sets in refs.items():
        rec = dict(size=float(np.mean([len(sets[i]) for i in ridx])))
        if all(key(sets[i]) in util[users[i]["uid"]] for i in ridx):
            rec["utility"] = boot([util[users[i]["uid"]][key(sets[i])] for i in ridx])
        for src in srcs:
            try:
                rec["exposure_" + src] = float(np.mean([loss_of(src, users[i]["uid"], sets[i], "count") for i in ridx]))
                rec["any_" + src] = float(np.mean([loss_of(src, users[i]["uid"], sets[i], "any") for i in ridx]))
                rec["sens_" + src] = float(np.mean([loss_of(src, users[i]["uid"], sets[i], "sens") for i in ridx]))
            except KeyError:     # random records beyond the top 8 have primary-auditor labels only
                pass
        fixed[name] = rec
    d = u_top5[ridx] - u_none[ridx]
    fixed["top5_minus_none"] = boot(d)
    fixed["top5_gain_significant"] = bool(fixed["top5_minus_none"][1] > 0)
    out["fixed"] = fixed

    # ---------------- validity: every calibration source x measurement source ----------
    rng = np.random.default_rng(SEED + 1)
    n_cal = N // 2
    acc = {}
    cal_srcs = ["qwen"] + (["union"] if "gemma" in audit else [])
    for _ in range(N_SPLITS):
        perm = rng.permutation(N)
        cal, test = perm[:n_cal], perm[n_cal:]
        for csrc in cal_srcs:
            for loss, B in LOSSES.items():
                for m in CAL:
                    for a in ALPHAS:
                        for proc in ("crc", "ltt"):
                            j = crc_index(Lm[(csrc, loss, m)][cal], a, B) if proc == "crc" else \
                                ltt_index(Lm[(csrc, loss, m)][cal], a, B, DELTA)
                            for msrc in srcs:
                                acc.setdefault((csrc, loss, proc, a, m, msrc), []).append(L[(msrc, loss, m)][test, j].mean())
                            acc.setdefault((csrc, loss, proc, a, m, "size"), []).append(SIZE[m][test, j].mean())
    valid = {}
    for (csrc, loss, proc, a, m, msrc), v in acc.items():
        v = np.array(v)
        valid.setdefault(csrc, {}).setdefault(loss, {}).setdefault(proc, {}).setdefault(str(a), {}).setdefault(m, {})[msrc] = dict(
            mean=float(v.mean()), frac_above=float(np.mean(v > a)) if msrc != "size" else None)
    # share of (alpha, rule) cells whose mean realized loss exceeds alpha, per measurement source
    cells = {}
    for csrc in cal_srcs:
        for loss in LOSSES:
            for msrc in srcs:
                ex = [valid[csrc][loss]["crc"][str(a)][m][msrc]["mean"] - a for a in ALPHAS for m in CAL if m in MAIN]
                cells[f"{csrc}|{loss}|{msrc}"] = dict(frac_cells_above=float(np.mean(np.array(ex) > 0)),
                                                      worst_excess=float(max(ex)), n_cells=len(ex))
    valid["cells"] = cells
    out["valid"] = valid

    # ---------------- cross-fitted utility on the primary labels -----------------------
    def crossfit(pu, csrc, proc):
        uniq = np.unique(pu)
        side = {}
        for t, i in enumerate(uniq[rng.permutation(len(uniq))]):
            side[i] = t % 2
        fold = np.array([side[i] for i in pu])
        res = {}
        for m in CAL:
            for a in ALPHAS:
                jj = np.zeros(len(pu), dtype=int)
                for f in (0, 1):
                    c = pu[fold != f]
                    jj[fold == f] = crc_index(Lm[(csrc, "count", m)][c], a, K) if proc == "crc" else \
                        ltt_index(Lm[(csrc, "count", m)][c], a, K, DELTA)
                res[(m, a)] = (UT[m][pu, jj], L[(PRIMARY, "count", m)][pu, jj], SIZE[m][pu, jj])
        return res

    xfit = {}
    for csrc in cal_srcs:
        for proc in ("crc", "ltt"):
            rng = np.random.default_rng(SEED + 2)
            pt = {}
            allu = np.arange(N)
            for _ in range(N_PART):
                r = crossfit(allu, csrc, proc)
                for (m, a), (uu, ee, zz) in r.items():
                    pt.setdefault((m, a), []).append((np.nanmean(uu), ee[has_u].mean(), zz[has_u].mean()))
            bs = {}
            for _ in range(N_BOOT):
                b = rng.integers(0, N, N)
                r = crossfit(b, csrc, proc)
                hb = has_u[b]
                bs.setdefault("none", []).append(np.nanmean(u_none[b][hb]))
                bs.setdefault("top5", []).append(np.nanmean(u_top5[b][hb]))
                for (m, a), (uu, ee, zz) in r.items():
                    bs.setdefault((m, a), []).append(np.nanmean(uu[hb]))
            gain = np.array(bs["top5"]) - np.array(bs["none"])
            for (m, a), v in pt.items():
                v = np.array(v)
                ub = np.array(bs[(m, a)])
                dt, dn = ub - np.array(bs["top5"]), ub - np.array(bs["none"])
                rec = dict(util=float(v[:, 0].mean()), expo=float(v[:, 1].mean()), size=float(v[:, 2].mean()),
                           util_ci=pct(ub), vs_top5=pct(dt), vs_top5_90=pct(dt, 5, 95), vs_none=pct(dn), vs_none_90=pct(dn, 5, 95))
                rec["sig_loss"] = bool(rec["vs_top5_90"][2] < 0)
                rec["sig_gain_over_top5"] = bool(rec["vs_top5_90"][1] > 0)
                if fixed["top5_gain_significant"]:
                    kept = dn / np.where(gain > 1e-9, gain, np.nan)
                    plug = (v[:, 0].mean() - np.nanmean(u_none[has_u])) / (np.nanmean(u_top5[has_u]) - np.nanmean(u_none[has_u]))
                    rec["kept"] = [float(plug)] + pct(kept)[1:]
                    rec["kept_90"] = [float(plug)] + pct(kept, 5, 95)[1:]
                    rec["noninferior"] = bool(rec["kept_90"][1] > 1 - REL_MARGIN)
                xfit.setdefault(csrc, {}).setdefault(proc, {}).setdefault(str(a), {})[m] = rec
    out["xfit"] = xfit

    # ---------------- auditing one fixed rule versus calibrating -----------------------
    rng = np.random.default_rng(SEED + 6)
    fsets = {"sim_top1": [rules[u["uid"]]["sets"]["sim"][1] for u in users],
             "sim_top5": [rules[u["uid"]]["sets"]["sim"][K] for u in users],
             "flt_top5": [rules[u["uid"]]["sets"]["flt"][K] for u in users]}
    FL = {f: np.array([loss_of(PRIMARY, u["uid"], fsets[f][i], "count") for i, u in enumerate(users)], dtype=float) for f in fsets}
    FU = {f: np.array([util[u["uid"]][key(fsets[f][i])] if h else np.nan for i, (u, h) in enumerate(zip(users, has_u))]) for f in fsets}
    FS = {f: np.array([len(x) for x in fsets[f]], dtype=float) for f in fsets}
    fc = {}
    for _ in range(500):
        perm = rng.permutation(N)
        cal, test = perm[:n_cal], perm[n_cal:]
        tr = test[has_u[test]]
        for a in (0.2, 0.5):
            for f in fsets:
                r_ = FL[f][cal].mean()
                for proc, ok in (("crc", (n_cal * r_ + K) / (n_cal + 1) <= a), ("ltt", hb_pvalue(r_ / K, n_cal, a / K) <= DELTA)):
                    fc.setdefault((proc, str(a), f), []).append((float(ok), np.nanmean(FU[f][tr]) if ok else np.nanmean(u_none[tr]),
                                                                 FS[f][test].mean() if ok else 0.0))
            for m in ("sim_thr", "cert_sim"):
                for proc in ("crc", "ltt"):
                    j = crc_index(Lm[(PRIMARY, "count", m)][cal], a, K) if proc == "crc" else ltt_index(Lm[(PRIMARY, "count", m)][cal], a, K, DELTA)
                    fc.setdefault((proc, str(a), m), []).append((1.0, np.nanmean(UT[m][tr, j]), SIZE[m][test, j].mean()))
    out["fixedcheck"] = {}
    for (proc, a, f), v in fc.items():
        v = np.array(v)
        out["fixedcheck"].setdefault(proc, {}).setdefault(a, {})[f] = dict(deploy_rate=float(v[:, 0].mean()), util=float(v[:, 1].mean()),
                                                                           size=float(v[:, 2].mean()))

    # ---------------- shift by profile length and Mondrian calibration ------------------
    npro = np.array([u["n_profile"] for u in users])
    short = npro <= np.median(npro)
    shift = dict(median_profile=float(np.median(npro)))
    rng = np.random.default_rng(SEED + 5)
    for name in ("short_to_long", "long_to_short", "random"):
        acc_s = {}
        for _ in range(300):
            if name == "random":
                perm = rng.permutation(N)
                c_idx, t_idx = perm[:100], perm[N // 2:]
            else:
                cal_mask = short if name == "short_to_long" else ~short
                c_idx = rng.choice(np.nonzero(cal_mask)[0], 100, replace=False)
                t_idx = np.nonzero(~cal_mask)[0]
            for m in CAL:
                for a in ALPHAS:
                    j = crc_index(Lm[(PRIMARY, "count", m)][c_idx], a, K)
                    acc_s.setdefault((m, a), []).append(L[(PRIMARY, "count", m)][t_idx, j].mean())
        shift[name] = {f"{m}|{a}": float(np.mean(v)) for (m, a), v in acc_s.items()}
    for grp, ms in (("main", [m for m in CAL if m in MAIN]), ("necessity", [m for m in CAL if m in EXTRA])):
        if not ms:
            continue
        ex = [shift[nm][f"{m}|{a}"] - a for nm in ("short_to_long", "long_to_short") for m in ms for a in ALPHAS]
        exr = [shift["random"][f"{m}|{a}"] - a for m in ms for a in ALPHAS]
        shift[grp] = dict(worst_excess=float(max(ex)), frac_cells_above=float(np.mean(np.array(ex) > 0)),
                          worst_excess_random=float(max(exr)))
    # Mondrian: within random halves, calibrate pooled or per profile-length bin; measure per bin
    mon = {}
    for _ in range(300):
        perm = rng.permutation(N)
        cal, test = perm[:n_cal], perm[n_cal:]
        for m in [x for x in CAL if x in MAIN]:
            for a in (0.2, 0.5):
                jp = crc_index(Lm[(PRIMARY, "count", m)][cal], a, K)
                for b, mask in (("short", short), ("long", ~short)):
                    cb, tb = cal[mask[cal]], test[mask[test]]
                    jb = crc_index(Lm[(PRIMARY, "count", m)][cb], a, K)
                    mon.setdefault(("pooled", m, a, b), []).append(L[(PRIMARY, "count", m)][tb, jp].mean())
                    mon.setdefault(("mondrian", m, a, b), []).append(L[(PRIMARY, "count", m)][tb, jb].mean())
                    mon.setdefault(("mondrian_size", m, a, b), []).append(SIZE[m][tb, jb].mean())
                    mon.setdefault(("pooled_size", m, a, b), []).append(SIZE[m][tb, jp].mean())
    shift["mondrian"] = {"|".join(map(str, k)): float(np.mean(v)) for k, v in mon.items()}
    for kind in ("pooled", "mondrian"):
        shift[kind + "_worst_bin_excess"] = float(max(np.mean(v) - k[2] for k, v in mon.items() if k[0] == kind))
    out["shift"] = shift

    # ---------------- oracle / minimality gap -------------------------------------------
    p = os.path.join(RES, f"{task}_oracle.json")
    if os.path.exists(p) and rtag == "qwen25_7b":
        orc = json.load(open(p))
        eps = orc["eps"]
        by = {r["uid"]: r for r in orc["users"]}
        idx = [i for i, u in enumerate(users) if u["uid"] in by and has_u[i]]
        o = [by[users[i]["uid"]] for i in idx]
        # oracle exposure re-counted with the primary labels on the oracle's chosen set
        oe = np.array([loss_of(PRIMARY, users[i]["uid"], by[users[i]["uid"]]["S"], "count") for i in idx], dtype=float)
        uref = np.array([r["u_ref"] for r in o])
        rng = np.random.default_rng(SEED + 4)

        def ci2(v):
            v = np.asarray(v, dtype=float)
            b = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)]
            return [float(np.mean(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

        g = dict(n=len(o), eps=eps, max_records=orc.get("max_steps"),
                 oracle=dict(exposure=ci2(oe), sufficient=ci2([r["sufficient"] for r in o]),
                             empty_sufficient=ci2([r["u_none"] >= r["u_ref"] - eps for r in o]),
                             needs_disclosing=ci2(oe > 0)), methods={})
        e = L[(PRIMARY, "count", "sim")][idx, K]
        g["methods"]["sim@K"] = dict(exposure=ci2(e), sufficient=ci2(UT["sim"][idx, K] >= uref - eps), gap=ci2(e - oe))
        rest = np.array([i for i in range(N) if i not in set(idx)])
        for m in [x for x in CAL if x in MAIN]:
            for a in (0.2, 0.5):
                j = crc_index(Lm[(PRIMARY, "count", m)][rest], a, K)
                e = L[(PRIMARY, "count", m)][idx, j]
                g["methods"][f"{m}@{a}"] = dict(exposure=ci2(e), sufficient=ci2(UT[m][idx, j] >= uref - eps), gap=ci2(e - oe))
        out["oracle"] = g

    sfx = ("" if rtag == "qwen25_7b" else "_" + rtag) + os.environ.get("OUT_SUFFIX", "")
    json.dump(out, open(os.path.join(RES, f"{task}_analysis{sfx}.json"), "w"), indent=1)
    print(task, rtag, "N", N, "reader", int(has_u.sum()), "rules", CAL, "primary", PRIMARY)
    print("top5-none", fixed["top5_minus_none"])
    x = xfit[PRIMARY]["crc"]["0.2"]
    print("0.2 crc", {m: (round(v["util"], 3), round(v["expo"], 2), round(v["size"], 1), v.get("kept"), v["sig_loss"]) for m, v in x.items()})
    print("cells", json.dumps(cells))


if __name__ == "__main__":
    main(*sys.argv[1:])
