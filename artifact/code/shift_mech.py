"""Is the profile-length shift failure of the necessity-threshold rule driven by the profile-length
feature of the necessity score? Refit the score with and without that feature (training users
only, same model family as gate.py) and repeat the short<->long shift test on exposure with the
primary (union) labels. Exposure only, no reader calls.

  python shift_mech.py TASK
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import K, RES, SEED, load_proc
from crc import crc_index, monotonize
from gate import M_CF, M_GATE, nec_features

ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
PROFILE_COL = 7          # log1p(n_profile) in nec_features


def main(task):
    users = load_proc(task)["users"]
    train = [u for u in users if u["split"] == "train"]
    ev = [u for u in users if u["split"] == "eval"]
    cf = json.load(open(os.path.join(RES, f"{task}_cf_utility_qwen25_7b.json")))
    rep = json.load(open(os.path.join(RES, f"{task}_gate_report.json")))
    aq = json.load(open(os.path.join(RES, f"{task}_audit.json")))
    ag = json.load(open(os.path.join(RES, f"{task}_audit_gemma2_9b_evaltop.json")))
    ev = [u for u in ev if u["uid"] in aq and u["uid"] in ag]
    X, y = [], []
    for u in train:
        for ci in range(min(M_CF, u["n_pool"])):
            X.append(nec_features(u, ci)); y.append(cf[u["uid"]][str(ci)] - cf[u["uid"]][""])
    X, y = np.array(X), np.array(y)
    mk = {"ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
          "gbr": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                   subsample=0.8, random_state=SEED)}[rep["nec"]["chosen"]]
    flag = lambda u, ci: int(len(aq[u["uid"]][str(ci)]) > 0 or len(ag[u["uid"]][str(ci)]) > 0)
    npro = np.array([u["n_profile"] for u in ev])
    short = npro <= np.median(npro)
    out = {}
    for name, cols in (("with_profile_feature", list(range(X.shape[1]))),
                       ("without_profile_feature", [c for c in range(X.shape[1]) if c != PROFILE_COL])):
        model = mk().fit(X[:, cols], y)
        gtr = model.predict(X[:, cols])
        grid = sorted(set([1e9] + [float(v) for v in np.quantile(gtr, np.linspace(0, 1, 41))] + [-1e9]), reverse=True)
        Lr = []
        for u in ev:
            P = min(M_GATE, u["n_pool"])
            g = model.predict(np.array([nec_features(u, ci) for ci in range(P)])[:, cols])
            order = [int(i) for i in np.argsort(-g, kind="stable")]
            gs = sorted(g, reverse=True)
            Lr.append([sum(flag(u, i) for i in order[:min(K, sum(x >= t for x in gs))]) for t in grid])
        Lr = np.array(Lr, dtype=float)
        Lm = monotonize(Lr)
        rng = np.random.default_rng(SEED + 5)
        ex = []
        for cal_mask in (short, ~short):
            acc = {a: [] for a in ALPHAS}
            for _ in range(300):
                c = rng.choice(np.nonzero(cal_mask)[0], 100, replace=False)
                t = np.nonzero(~cal_mask)[0]
                for a in ALPHAS:
                    acc[a].append(Lr[t, crc_index(Lm[c], a, K)].mean())
            ex += [np.mean(v) - a for a, v in acc.items()]
        out[name] = dict(worst_excess=float(max(ex)), frac_cells_above=float(np.mean(np.array(ex) > 0)),
                         corr_g_profile=float(np.corrcoef(gtr, X[:, PROFILE_COL])[0, 1]))
    json.dump(out, open(os.path.join(RES, f"{task}_shift_mech.json"), "w"), indent=1)
    print(task, out)


if __name__ == "__main__":
    main(sys.argv[1])
