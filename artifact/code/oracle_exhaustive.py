"""Exhaustive small-slice oracle sensitivity for LaMP-3.

For a manageable evaluation slice, enumerate all 2^8 subsets of the top-eight candidate
pool, compare them with the full retrieved pool, and report only aggregate statistics.  The
search is hindsight/oracle only: auditor labels and the realized reader utility are used to
choose a witness.  A cap-three replay is included to expose the gap between exhaustive and
the original greedy/cap-limited implementation.

  python oracle_exhaustive.py [MODEL] [N_USERS]
"""
import itertools
import json
import os
import sys

import numpy as np

from common import K, NUM_PREDICT, RES, SEED, LLMCache, generate, load_proc, reader_prompt, utility
from run import M_GATE, pmap, tag


EPSILONS = [0.0, 0.01, 0.025, 0.05, 0.10]


def pct(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return [float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [None, None, None]


def main(model="qwen2.5:7b", n_users=20):
    users = [u for u in load_proc("lamp3")["users"] if u["split"] == "eval"][:n_users]
    qwen = json.load(open(os.path.join(RES, "lamp3_audit.json")))
    gemma = json.load(open(os.path.join(RES, "lamp3_audit_gemma2_9b_evaltop.json")))
    users = [u for u in users if u["uid"] in qwen and u["uid"] in gemma and
             all(str(i) in qwen[u["uid"]] and str(i) in gemma[u["uid"]]
                 for i in range(min(M_GATE, u["n_pool"]))) ]
    if not users:
        raise RuntimeError("no oracle users have both auditor labels")

    # All subsets of the top-eight pool.  The cap-three set is exactly the original size cap.
    subset_by_user = {}
    for u in users:
        p = min(M_GATE, u["n_pool"])
        all_subsets = [tuple(c) for r in range(p + 1) for c in itertools.combinations(range(p), r)]
        subset_by_user[u["uid"]] = all_subsets

    cache = LLMCache(f"reader_lamp3_{tag(model)}")
    items = []
    for u in users:
        items.append((u, "fullpool", tuple(range(u["n_pool"]))))
        items += [(u, "subset", ids) for ids in subset_by_user[u["uid"]]]

    def work(item):
        u, kind, ids = item
        out = generate(cache, model, reader_prompt("lamp3", u, ids), num_predict=NUM_PREDICT["lamp3"])
        return u["uid"], kind, ",".join(map(str, ids)), float(utility("lamp3", u, out))

    values = {}
    for j, (uid, kind, key, value) in enumerate(pmap(work, items)):
        values.setdefault(uid, {})[kind + ":" + key] = value
        if j % 500 == 0:
            print("oracle exhaustive reader", j, len(items), flush=True)

    def exposure(u, ids):
        uid = u["uid"]
        return sum(bool(set(qwen[uid][str(i)]) | set(gemma[uid][str(i)])) for i in ids)

    def select(u, eps, cap):
        uid = u["uid"]
        full_key = "fullpool:" + ",".join(map(str, range(u["n_pool"])))
        ref = values[uid][full_key]
        choices = [s for s in subset_by_user[uid] if cap is None or len(s) <= cap]
        feasible = [s for s in choices if values[uid]["subset:" + ",".join(map(str, s))] >= ref - eps]
        if not feasible:
            return None, ref, values[uid]["subset:"], False
        best = min(feasible, key=lambda s: (exposure(u, s), len(s),
                                             -values[uid]["subset:" + ",".join(map(str, s))]))
        key = "subset:" + ",".join(map(str, best))
        return best, ref, values[uid][key], True

    out = {
        "task": "lamp3", "model": model, "n_users": len(users), "candidate_pool": M_GATE,
        "search": "all 2^8 subsets of top-eight candidates; cap3 is the original cap-limited replay",
        "labels": "union of qwen2.5:7b and gemma2:9b for selected top-eight records",
        "epsilon": EPSILONS, "rows": {},
        "reference_definition": "full retrieved pool utility versus no-profile empty-set utility",
        "caveat": "oracle uses realized reader utility and auditor labels; it is a feasible-witness sensitivity analysis, not a deployable method",
    }
    for cap_name, cap in (("exhaustive_8", None), ("cap3", 3)):
        section = {}
        for eps in EPSILONS:
            rows = []
            for u in users:
                s, ref, util, feasible = select(u, eps, cap)
                uid = u["uid"]
                none = values[uid]["subset:"]
                top5_key = "subset:" + ",".join(map(str, range(min(K, M_GATE))))
                top5 = values[uid][top5_key]
                row = {
                    "feasible": bool(feasible), "exposure": float(exposure(u, s)) if feasible else None,
                    "size": len(s) if feasible else None, "utility": float(util) if feasible else None,
                    "empty_sufficient": bool(none >= ref - eps),
                    "full_pool_utility": float(ref), "no_profile_utility": float(none),
                    "full_minus_none": float(ref - none), "top5_utility": float(top5),
                    "top5_minus_none": float(top5 - none),
                }
                rows.append(row)
            feasible_rows = [r for r in rows if r["feasible"]]
            sec = {
                "sufficient_rate": float(np.mean([r["feasible"] for r in rows])),
                "empty_sufficient_rate": float(np.mean([r["empty_sufficient"] for r in rows])),
                "full_pool_utility": pct([r["full_pool_utility"] for r in rows]),
                "no_profile_utility": pct([r["no_profile_utility"] for r in rows]),
                "full_minus_none": pct([r["full_minus_none"] for r in rows]),
                "top5_utility": pct([r["top5_utility"] for r in rows]),
                "top5_minus_none": pct([r["top5_minus_none"] for r in rows]),
                "feasible_only_exposure": pct([r["exposure"] for r in feasible_rows]),
                "feasible_only_size": pct([r["size"] for r in feasible_rows]),
                "feasible_only_utility": pct([r["utility"] for r in feasible_rows]),
                "feasible_users_needing_disclosure": float(np.mean([r["exposure"] > 0 for r in feasible_rows])) if feasible_rows else None,
                "infeasible_rate": float(np.mean([not r["feasible"] for r in rows])),
            }
            section[str(eps)] = sec
        out["rows"][cap_name] = section

    path = os.path.join(RES, "lamp3_oracle_exhaustive_sensitivity.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({"out": path, "n_users": len(users), "rows": out["rows"]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b",
         int(sys.argv[2]) if len(sys.argv) > 2 else 20)
