"""Held-out auditor test (round-3 plan). On the evaluation users that carry labels from all three
auditors (qwen2.5:7b, gemma2:9b, Llama-3.3-70B), calibrate each kept rule on the union of two
auditors and measure test exposure with the third, over random calibration/test halvings.

  python heldout.py TASK
"""
import json
import os
import sys

import numpy as np

from common import K, RES, SEED, load_proc
from crc import crc_index, monotonize

ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
RULES = ["sim", "sim_thr", "flt", "cert_sim"]
SRC = {"qwen": "{t}_audit.json", "gemma": "{t}_audit_gemma2_9b_evaltop.json",
       "llama70": "{t}_audit_llama33_70b_evaltop.json"}


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe))


def main(task):
    lab = {s: json.load(open(os.path.join(RES, f.format(t=task)))) for s, f in SRC.items()}
    rules = json.load(open(os.path.join(RES, f"{task}_rules.json")))
    grid = json.load(open(os.path.join(RES, f"{task}_thr_grid.json")))
    users = [u for u in load_proc(task)["users"] if u["split"] == "eval"
             and all(u["uid"] in lab[s] for s in SRC) and u["uid"] in rules]
    users = [u for u in users if all(str(ci) in lab["llama70"][u["uid"]] for ci in range(len(rules[u["uid"]]["s"])))]
    for u in users:
        r = rules[u["uid"]]
        sims = [u["cands"][ci]["sim"] for ci in range(len(r["g"]))]
        r["sets"]["sim_thr"] = [r["sets"]["sim"][min(K, sum(x >= t for x in sims))] for t in grid["sim"]]

    def f(src, uid, ci):
        if "+" in src:
            return max(f(s, uid, ci) for s in src.split("+"))
        return int(len(lab[src][uid][str(ci)]) > 0)

    recs = [(u["uid"], ci) for u in users for ci in range(len(rules[u["uid"]]["s"]))]
    y = {s: np.array([f(s, uid, ci) for uid, ci in recs]) for s in SRC}
    out = dict(task=task, n_users=len(users), n_records=len(recs),
               rate={s: float(v.mean()) for s, v in y.items()},
               kappa={f"{a}-{b}": kappa(y[a], y[b]) for a in SRC for b in SRC if a < b},
               recall_of_llama70={s: float(y[s][y["llama70"] == 1].mean()) for s in ("qwen", "gemma")},
               union_qg_recall_of_llama70=float(np.maximum(y["qwen"], y["gemma"])[y["llama70"] == 1].mean()),
               llama70_recall_of_union_qg=float(y["llama70"][np.maximum(y["qwen"], y["gemma"]) == 1].mean()))
    N = len(users)
    cal_names = {"llama70": "qwen+gemma", "gemma": "qwen+llama70", "qwen": "gemma+llama70"}
    pairs = [("qwen", "llama70"), ("qwen+gemma", "llama70"), ("qwen+llama70", "gemma"), ("gemma+llama70", "qwen")]
    L = {(src, m): np.array([[sum(f(src, u["uid"], i) for i in s) for s in rules[u["uid"]]["sets"][m]] for u in users], dtype=float)
         for src in sorted(set(list(SRC) + [c for c, _ in pairs])) for m in RULES}
    rng = np.random.default_rng(SEED + 9)
    acc = {}
    for _ in range(1000):
        perm = rng.permutation(N)
        cal, test = perm[:N // 2], perm[N // 2:]
        for cs, held in pairs:
            for m in RULES:
                Lc = monotonize(L[(cs, m)])
                for a in ALPHAS:
                    j = crc_index(Lc[cal], a, K)
                    acc.setdefault((cs, held, m, a), []).append(L[(held, m)][test, j].mean())
                    acc.setdefault((cs, cs, m, a), []).append(L[(cs, m)][test, j].mean())
    res = {}
    for cs, held in pairs:
        ex = [np.mean(acc[(cs, held, m, a)]) - a for m in RULES for a in ALPHAS]
        exs = [np.mean(acc[(cs, cs, m, a)]) - a for m in RULES for a in ALPHAS]
        res[f"{cs}->{held}"] = dict(frac_cells_above=float(np.mean(np.array(ex) > 0)), worst_excess=float(max(ex)),
                                    frac_cells_above_self=float(np.mean(np.array(exs) > 0)),
                                    cells={f"{m}|{a}": float(np.mean(acc[(cs, held, m, a)])) for m in RULES for a in ALPHAS})
    out["heldout"] = res
    json.dump(out, open(os.path.join(RES, f"{task}_heldout.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "heldout"}, indent=0))
    print({h: (round(r["frac_cells_above"], 3), round(r["worst_excess"], 3)) for h, r in res.items()})


if __name__ == "__main__":
    main(sys.argv[1])
