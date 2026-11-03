"""Calibration, validity and matched-exposure utility analysis. Pure numpy, no LLM calls.

  python analyze.py TASK [READER_TAG]

Writes results/{task}_analysis[_READER].json with
  base      : auditor base rates, scorer quality, detector agreement
  fixed     : exposure and utility of uncalibrated operating points
  curves    : exposure / utility at every grid index of every calibratable rule
  crc       : per alpha and rule, realized test exposure and utility over random splits
  diffs     : user-bootstrap CIs of utility differences at matched certified alpha
  oracle    : minimality-gap numbers (if the oracle search has been run)
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from common import K, RES, SEED, SENSITIVE_CATS, load_proc
from crc import crc_index, monotonize, ucb_index

ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
CAL = ["cert_nec", "cert_sim", "sim", "nec", "tok", "flt", "nec_thr", "sim_thr"]
N_SPLITS = 1000
N_BOOT = 1000
FLAG_KEYS = ["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "NRP",
             "lex_health", "lex_finance", "lex_family", "lex_belief", "lex_place"]


def label(cats, metric):
    return int(len(cats) > 0) if metric == "disc" else int(any(c in SENSITIVE_CATS for c in cats))


def main(task, rtag="qwen25_7b"):
    users = [u for u in load_proc(task)["users"] if u["split"] == "eval"]
    rules = json.load(open(os.path.join(RES, f"{task}_rules.json")))
    audit = json.load(open(os.path.join(RES, f"{task}_audit.json")))
    det = json.load(open(os.path.join(RES, f"{task}_detect.json")))
    util = json.load(open(os.path.join(RES, f"{task}_reader_utility_{rtag}.json")))
    users = [u for u in users if u["uid"] in rules and u["uid"] in audit]
    # adaptive threshold rules: admit the prefix of the necessity (similarity) ranking whose
    # score clears tau, capped at K; tau runs over a grid of training-score quantiles
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
    out = dict(task=task, reader=rtag, n_eval=N, n_reader=int(has_u.sum()), K=K, alphas=ALPHAS)

    def key(ids):
        return ",".join(map(str, sorted(ids)))

    def expo(u, ids, metric):
        return sum(label(audit[u["uid"]][str(i)], metric) for i in ids)

    def U(u, ids):
        return util[u["uid"]][key(ids)]

    # ---------------- base rates and scorer quality ------------------------------------
    y, s, fl, ys = [], [], [], []
    for u in users:
        for ci, sv in enumerate(rules[u["uid"]]["s"]):
            c = audit[u["uid"]][str(ci)]
            y.append(label(c, "disc")); ys.append(label(c, "sens")); s.append(sv)
            fl.append(int(any(det[u["uid"]][str(ci)][k] > 0 for k in FLAG_KEYS)))
    y, s, fl, ys = map(np.array, (y, s, fl, ys))
    cat_counts = {}
    for u in users:
        for ci in range(len(rules[u["uid"]]["s"])):
            for c in audit[u["uid"]][str(ci)]:
                cat_counts[c] = cat_counts.get(c, 0) + 1
    out["base"] = dict(n_records=int(len(y)), rate_disc=float(y.mean()), rate_sens=float(ys.mean()),
                       users_with_any_disc=float(np.mean([any(len(audit[u["uid"]][str(ci)]) > 0 for ci in range(len(rules[u["uid"]]["s"]))) for u in users])),
                       proxy_auc=float(roc_auc_score(y, s)),
                       detector_flag_rate=float(fl.mean()), detector_recall=float(fl[y == 1].mean()),
                       detector_precision=float(y[fl == 1].mean()), categories=cat_counts,
                       gate=json.load(open(os.path.join(RES, f"{task}_gate_report.json"))))

    for metric in ("disc", "sens"):
        E, Uu = {}, {}
        for m in CAL + ["rand"]:
            J = len(rules[users[0]["uid"]]["sets"][m])
            if m == "rand":
                E[m] = np.array([[expo(u, rules[u["uid"]]["sets"][m][j], metric) if h else np.nan for j in range(J)]
                                 for u, h in zip(users, has_u)], dtype=float)
            else:
                E[m] = np.array([[expo(u, rules[u["uid"]]["sets"][m][j], metric) for j in range(J)] for u in users], dtype=float)
            Uu[m] = np.array([[U(u, rules[u["uid"]]["sets"][m][j]) if h else np.nan for j in range(J)]
                              for u, h in zip(users, has_u)], dtype=float)
        size = {m: np.array([[len(rules[u["uid"]]["sets"][m][j]) for j in range(E[m].shape[1])] for u in users], dtype=float)
                for m in CAL}
        res = {}

        # ---------------- uncalibrated operating points -------------------------------
        rng = np.random.default_rng(SEED)
        ridx = np.nonzero(has_u)[0]

        def ci(v):
            v = np.asarray(v, dtype=float)
            bs = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)]
            return [float(np.mean(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

        u_none = Uu["sim"][ridx, 0]
        u_full = np.array([U(users[i], rules[users[i]["uid"]]["sets"]["full"][0]) for i in ridx])
        tokk = np.array([rules[u["uid"]]["sets"]["tok_native_k"] for u in users])
        fixed = {"none": dict(exposure=[0.0, 0.0, 0.0], utility=ci(u_none), size=0.0),
                 "full": dict(exposure=None, utility=ci(u_full),
                              size=float(np.mean([users[i]["n_pool"] for i in ridx])))}
        for m in ("sim", "nec", "flt", "rand"):
            fixed[m + "@K"] = dict(exposure=ci(E[m][ridx, K]), utility=ci(Uu[m][ridx, K]),
                                   exposure_all=float(np.nanmean(E[m][:, K])),
                                   size=float(np.mean([len(rules[users[i]["uid"]]["sets"][m][K]) for i in ridx])))
        fixed["tok@native"] = dict(exposure=ci(E["tok"][ridx, tokk[ridx]]), utility=ci(Uu["tok"][ridx, tokk[ridx]]),
                                   exposure_all=float(np.mean(E["tok"][np.arange(N), tokk])), size=float(tokk[ridx].mean()))
        res["fixed"] = fixed
        res["curves"] = {m: dict(exposure=np.nanmean(E[m], axis=0).tolist(),
                                 utility=np.nanmean(Uu[m][ridx], axis=0).tolist()) for m in CAL + ["rand"]}

        # ---------------- CRC over random calibration / test splits --------------------
        Lm = {m: monotonize(E[m]) for m in CAL}
        rng = np.random.default_rng(SEED + 1)
        n_cal = N // 2
        crc = {a: {m: dict(expo=[], util=[], size=[], idx=[]) for m in CAL} for a in ALPHAS}
        ucb = {a: dict(expo=[], idx=[]) for a in ALPHAS}
        for _ in range(N_SPLITS):
            perm = rng.permutation(N)
            cal, test = perm[:n_cal], perm[n_cal:]
            tr = test[has_u[test]]
            for a in ALPHAS:
                for m in CAL:
                    j = crc_index(Lm[m][cal], a, K)
                    d = crc[a][m]
                    d["expo"].append(E[m][test, j].mean()); d["util"].append(Uu[m][tr, j].mean())
                    d["size"].append(size[m][test, j].mean()); d["idx"].append(j)
                j = ucb_index(Lm["cert_nec"][cal], a, K, 0.1)
                ucb[a]["expo"].append(E["cert_nec"][test, j].mean()); ucb[a]["idx"].append(j)
        res["crc"] = {str(a): {m: dict(expo_mean=float(np.mean(d["expo"])), expo_sd=float(np.std(d["expo"])),
                                       expo_se=float(np.std(d["expo"]) / np.sqrt(N_SPLITS)),
                                       frac_splits_above=float(np.mean(np.array(d["expo"]) > a)),
                                       util_mean=float(np.mean(d["util"])), size_mean=float(np.mean(d["size"])),
                                       idx_mean=float(np.mean(d["idx"])))
                                   for m, d in crc[a].items()} for a in ALPHAS}
        res["ucb_cert_nec"] = {str(a): dict(expo_mean=float(np.mean(d["expo"])),
                                            frac_splits_above=float(np.mean(np.array(d["expo"]) > a)),
                                            idx_mean=float(np.mean(d["idx"]))) for a, d in ucb.items()}

        # ---------------- user bootstrap of matched-alpha utility differences ----------
        rng = np.random.default_rng(SEED + 2)
        diffs = {a: {m: [] for m in CAL if m != "cert_nec"} for a in ALPHAS}
        absu = {a: {m: [] for m in CAL} for a in ALPHAS}
        absu_none, absu_simk, expo_simk = [], [], []
        for _ in range(N_BOOT):
            uniq = rng.permutation(N)
            side = np.zeros(N, dtype=bool); side[uniq[:n_cal]] = True       # True = calibration
            bs = rng.integers(0, N, N)
            cal, test = bs[side[bs]], bs[~side[bs]]
            tr = test[has_u[test]]
            if len(tr) < 5 or len(cal) < 5:
                continue
            absu_none.append(Uu["sim"][tr, 0].mean())
            absu_simk.append(Uu["sim"][tr, K].mean())
            expo_simk.append(E["sim"][test, K].mean())
            for a in ALPHAS:
                um = {m: Uu[m][tr, crc_index(Lm[m][cal], a, K)].mean() for m in CAL}
                for m in CAL:
                    absu[a][m].append(um[m])
                    if m != "cert_nec":
                        diffs[a][m].append(um["cert_nec"] - um[m])
        pct = lambda v: [float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        res["diffs"] = {str(a): {m: pct(v) for m, v in d.items()} for a, d in diffs.items()}
        res["pair"] = {str(a): {f"{m1}-{m2}": pct(np.array(absu[a][m1]) - np.array(absu[a][m2]))
                                for m1 in CAL for m2 in CAL if m1 != m2} for a in ALPHAS}
        res["pair_simK"] = {str(a): {m: pct(np.array(absu[a][m]) - np.array(absu_simk)) for m in CAL} for a in ALPHAS}
        res["simK_none"] = pct(np.array(absu_simk) - np.array(absu_none))
        res["pair_none"] = {str(a): {m: pct(np.array(absu[a][m]) - np.array(absu_none)) for m in CAL} for a in ALPHAS}
        res["util_ci"] = {str(a): {m: pct(v) for m, v in d.items()} for a, d in absu.items()}
        res["util_none_ci"] = pct(absu_none)
        out[metric] = res

    # ---------------- oracle / minimality gap -------------------------------------------
    p = os.path.join(RES, f"{task}_oracle.json")
    if os.path.exists(p) and rtag == "qwen25_7b":
        orc = json.load(open(p))
        eps = orc["eps"]
        by = {r["uid"]: r for r in orc["users"]}
        idx = [i for i, u in enumerate(users) if u["uid"] in by and has_u[i]]
        o = [by[users[i]["uid"]] for i in idx]
        uref = np.array([r["u_ref"] for r in o])
        rng = np.random.default_rng(SEED + 3)

        def ci2(v):
            v = np.asarray(v, dtype=float)
            bs = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)]
            return [float(np.mean(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

        E = {m: np.array([[expo(users[i], rules[users[i]["uid"]]["sets"][m][j], "disc")
                           for j in range(len(rules[users[i]["uid"]]["sets"][m]))] for i in idx], dtype=float) for m in CAL}
        Um = {m: np.array([[U(users[i], rules[users[i]["uid"]]["sets"][m][j])
                            for j in range(len(rules[users[i]["uid"]]["sets"][m]))] for i in idx], dtype=float) for m in CAL}
        g = dict(n=len(o), eps=eps,
                 oracle=dict(exposure=ci2([r["exposure"] for r in o]), size=ci2([len(r["S"]) for r in o]),
                             sufficient=ci2([r["sufficient"] for r in o]), utility=ci2([r["u"] for r in o]),
                             empty_sufficient=ci2([r["u_none"] >= r["u_ref"] - eps for r in o]),
                             needs_disclosing=ci2([r["exposure"] > 0 for r in o]),
                             mean_calls=float(np.mean([r["calls"] for r in o]))),
                 methods={})
        for m in ("sim", "nec", "flt"):
            g["methods"][m + "@K"] = dict(exposure=ci2(E[m][:, K]), sufficient=ci2(Um[m][:, K] >= uref - eps),
                                          gap=ci2(E[m][:, K] - np.array([r["exposure"] for r in o])))
        # certified gate at the grid index most often selected for each alpha (deterministic operating point)
        rest = [i for i in range(N) if i not in set(idx)]
        for mm in ("cert_nec", "cert_sim", "sim", "nec_thr", "sim_thr", "tok", "flt"):
            Lall = monotonize(np.array([[expo(u, rules[u["uid"]]["sets"][mm][j], "disc")
                                         for j in range(E[mm].shape[1])] for u in users], dtype=float))
            for a in ALPHAS:
                j = crc_index(Lall[rest], a, K)       # calibrated on users outside the oracle subsample
                g["methods"][f"{mm}@{a}"] = dict(exposure=ci2(E[mm][:, j]), idx=j,
                                                 sufficient=ci2(Um[mm][:, j] >= uref - eps),
                                                 utility=ci2(Um[mm][:, j]),
                                                 gap=ci2(E[mm][:, j] - np.array([r["exposure"] for r in o])))
        out["oracle"] = g

    sfx = "" if rtag == "qwen25_7b" else "_" + rtag
    json.dump(out, open(os.path.join(RES, f"{task}_analysis{sfx}.json"), "w"), indent=1)
    d = out["disc"]
    print(json.dumps(out["base"], indent=0)[:1500])
    print("fixed", json.dumps(d["fixed"]))
    for a in ALPHAS:
        print(a, {m: (round(v["expo_mean"], 3), round(v["util_mean"], 4), round(v["size_mean"], 2)) for m, v in d["crc"][str(a)].items()})
        print("   diff cert_nec - m:", {m: [round(x, 4) for x in v] for m, v in d["diffs"][str(a)].items()})
    if "oracle" in out:
        print(json.dumps(out["oracle"], indent=0))


if __name__ == "__main__":
    main(*sys.argv[1:])
