"""Fit the two on-device scorers on training users and emit admission rules for evaluation users.

  necessity  g(q, r): predicted counterfactual utility gain of admitting record r alone
  sensitivity s(r):   predicted probability that the auditor flags r (cheap proxy, no LLM)

Both are fitted on training users only. For every evaluation user the script writes, per
method, the nested (or grid-indexed) family of admitted record sets, and the list of
distinct contexts the reader has to be run on.

  python gate.py TASK [READER_TAG]
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from common import K, PROC, RES, SEED, load_proc, record_text

M_GATE, M_CF = 8, 6
N_READER = int(os.environ.get("N_READER", 150))   # evaluation users that get reader calls
GRID_Q = [0.0, 0.3, 0.5, 0.65, 0.8, 0.9, 0.97, 1.0]  # sensitivity-threshold grid (train quantiles)
TOK_LAMBDA = 0.10                                  # token-cost weight of the ENOUGH-style gate
DET_KEYS = ["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "NRP", "DATE_TIME", "URL",
            "lex_health", "lex_finance", "lex_family", "lex_belief", "lex_place", "lex_firstperson", "len"]
FLAG_KEYS = ["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "NRP",
             "lex_health", "lex_finance", "lex_family", "lex_belief", "lex_place"]


def emb_index(task):
    """Row offset of each user's first candidate in the embedding matrix."""
    full = json.load(open(os.path.join(PROC, f"{task}.json")))["users"]
    off, o = {}, 0
    for u in full:
        off[u["uid"]] = o
        o += len(u["cands"])
    return off, np.load(os.path.join(PROC, f"{task}_emb.npy"))


def nec_features(u, ci):
    c = u["cands"][ci]
    sims = [x["sim"] for x in u["cands"][: u["n_pool"]]]
    bms = [x["bm25"] for x in u["cands"][: u["n_pool"]]]
    return [c["sim"], c["sim"] - float(np.mean(sims)), c["sim"] / (max(sims) + 1e-9),
            c["bm25"], c["bm25"] / (max(bms) + 1e-9), ci, len(c["text"].split()),
            np.log1p(u["n_profile"]), len(u["query"].split())]


def main(task, rtag="qwen25_7b"):
    users = load_proc(task)["users"]
    train = [u for u in users if u["split"] == "train"]
    evalu = [u for u in users if u["split"] == "eval"]
    audit = json.load(open(os.path.join(RES, f"{task}_audit_train.json")))
    det = json.load(open(os.path.join(RES, f"{task}_detect.json")))
    pcf = os.path.join(RES, f"{task}_cf_utility_{rtag}.json")
    cf = json.load(open(pcf)) if os.path.exists(pcf) else None
    off, EMB = emb_index(task)
    report = {}

    # ---------------- sensitivity proxy -------------------------------------------------
    def sens_x(u, ci):
        d = det[u["uid"]][str(ci)]
        return np.concatenate([EMB[off[u["uid"]] + ci], [np.log1p(d[k]) for k in DET_KEYS]])

    tr_ids = [(u, ci) for u in train for ci in audit[u["uid"]] if int(ci) < min(M_GATE, u["n_pool"])]
    Xs = np.array([sens_x(u, int(ci)) for u, ci in tr_ids])
    ys = np.array([int(len(audit[u["uid"]][ci]) > 0) for u, ci in tr_ids])
    sens = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced"))
    sens.fit(Xs, ys)
    report["sens"] = dict(n_train=int(len(ys)), base_rate_train=float(ys.mean()))
    grid = [-1.0] + [float(np.quantile(sens.predict_proba(Xs)[:, 1], q)) for q in GRID_Q[1:-1]] + [1.0]

    # ---------------- necessity scorer --------------------------------------------------
    if cf is None:
        # no counterfactual labels for this task (LaMP-2): the "necessity" score is similarity,
        # so the necessity-based rules coincide with similarity rules and are not reported
        class _Sim:
            def predict(self, X):
                return np.asarray(X)[:, 0]
        nec = _Sim()
        Xn = np.array([nec_features(u, ci) for u in train for ci in range(min(M_CF, u["n_pool"]))])
        report["nec"] = dict(cv_spearman={"none": float("nan")}, chosen="none", mean_gain_train=float("nan"),
                             frac_positive_gain=float("nan"), frac_negative_gain=float("nan"),
                             spearman_sim_gain=float("nan"))
    else:
        Xn, yn, gn = [], [], []
        for u in train:
            base = cf[u["uid"]][""]
            for ci in range(min(M_CF, u["n_pool"])):
                Xn.append(nec_features(u, ci))
                yn.append(cf[u["uid"]][str(ci)] - base)
                gn.append(u["uid"])
        Xn, yn, gn = np.array(Xn), np.array(yn), np.array(gn)
        cands = {"ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
                 "gbr": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                          subsample=0.8, random_state=SEED)}
        cv = {}
        for name, mk in cands.items():
            pred = np.zeros(len(yn))
            for tr, te in GroupKFold(5).split(Xn, yn, gn):
                pred[te] = mk().fit(Xn[tr], yn[tr]).predict(Xn[te])
            cv[name] = float(spearmanr(pred, yn).statistic)
        best = max(cv, key=cv.get)
        nec = cands[best]().fit(Xn, yn)
        report["nec"] = dict(cv_spearman=cv, chosen=best, mean_gain_train=float(yn.mean()),
                             frac_positive_gain=float((yn > 0).mean()), frac_negative_gain=float((yn < 0).mean()),
                             spearman_sim_gain=float(spearmanr(Xn[:, 0], yn).statistic))
    # threshold grids for the adaptive (threshold) variants, from training users only
    qs = np.linspace(0, 1, 41)
    gtr = nec.predict(Xn)
    str_ = np.array([c["sim"] for u in train for c in u["cands"][:min(M_GATE, u["n_pool"])]])
    thr = dict(nec=sorted(set([float("inf")] + [float(x) for x in np.quantile(gtr, qs)] + [float("-inf")]), reverse=True),
               sim=sorted(set([float("inf")] + [float(x) for x in np.quantile(str_, qs)] + [float("-inf")]), reverse=True))
    json.dump({k: [x if np.isfinite(x) else (1e9 if x > 0 else -1e9) for x in v] for k, v in thr.items()},
              open(os.path.join(RES, f"{task}_thr_grid.json"), "w"))
    if os.environ.get("GRID_ONLY"):
        return

    # ---------------- admission rules for evaluation users ------------------------------
    mean_tok = float(np.mean([len(record_text(task, c).split()) for u in train for c in u["cands"][:M_GATE]]))
    rules, contexts = {}, {}
    for j, u in enumerate(evalu):
        P = min(M_GATE, u["n_pool"])
        g = nec.predict(np.array([nec_features(u, ci) for ci in range(P)]))
        s = sens.predict_proba(np.array([sens_x(u, ci) for ci in range(P)]))[:, 1]
        toks = np.array([len(record_text(task, u["cands"][ci]).split()) for ci in range(P)])
        flag = [any(det[u["uid"]][str(ci)][k] > 0 for k in FLAG_KEYS) for ci in range(P)]
        o_sim = list(range(P))
        o_nec = [int(i) for i in np.argsort(-g, kind="stable")]
        v_tok = g - TOK_LAMBDA * toks / (K * mean_tok)       # net value, L_ref = K mean record lengths
        o_tok = [int(i) for i in np.argsort(-v_tok, kind="stable")]
        o_flt = [i for i in o_sim if not flag[i]]
        r = {}
        r["sim"] = [o_sim[:k] for k in range(K + 1)]
        r["rand"] = [u["rand"][:k] for k in range(K + 1)]
        r["nec"] = [o_nec[:k] for k in range(K + 1)]
        r["tok"] = [o_tok[:k] for k in range(K + 1)]
        r["flt"] = [o_flt[:k] for k in range(K + 1)]
        # certified gates: K most necessary (or most similar) records with predicted sensitivity <= lambda
        r["cert_nec"] = [[i for i in o_nec if s[i] <= lam][:K] for lam in grid]
        r["cert_sim"] = [[i for i in o_sim if s[i] <= lam][:K] for lam in grid]
        # native stopping point of the token-cost gate: admit while the net value is positive
        r["tok_native_k"] = int(min(K, np.sum(v_tok > 0)))
        r["full"] = [list(range(u["n_pool"]))]
        rules[u["uid"]] = dict(sets=r, g=[float(x) for x in g], s=[float(x) for x in s])
        if j < N_READER:
            seen, lst = set(), []
            for m in ["sim", "cert_nec", "nec", "cert_sim", "flt", "tok", "rand", "full"]:
                for ids in r[m]:
                    key = tuple(sorted(ids))
                    if key not in seen:
                        seen.add(key)
                        lst.append(list(key))
            contexts[u["uid"]] = lst
    report["grid"] = grid
    report["n_contexts"] = int(sum(len(v) for v in contexts.values()))
    report["n_eval"] = len(evalu)
    report["n_reader"] = len(contexts)
    json.dump(rules, open(os.path.join(RES, f"{task}_rules.json"), "w"))
    json.dump(contexts, open(os.path.join(RES, f"{task}_contexts.json"), "w"))
    json.dump(report, open(os.path.join(RES, f"{task}_gate_report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
