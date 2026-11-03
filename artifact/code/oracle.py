"""Greedy counterfactual approximation of the minimal sufficient record set.

For a user with query q, candidate pool C (top-M_GATE retrieved records) and reference utility
U_ref = U(q, full retrieved pool), the minimal sufficient set is
    S* = argmin_{S subset of C, |S| <= K} E(S)   s.t.   U(q, S) >= U_ref - eps,
where E counts audited disclosing records. The search uses the gold output (hindsight), so it
is an oracle and a lower bound on exposure for sufficiency, never a deployable method.

Greedy search: start from the empty set. While the set is insufficient and smaller than
MAX_STEPS, evaluate every one-record extension. If some extension is sufficient, take the one
with least exposure (ties: highest utility). Otherwise extend with the highest-utility record,
preferring non-disclosing records on ties, provided it improves utility; stop if none improves.

  python oracle.py TASK [N_USERS]
"""
import json
import os
import sys

from common import NUM_PREDICT, READER, RES, K, LLMCache, generate, load_proc, reader_prompt, utility
from run import M_GATE, pmap, tag

EPS = 0.05
MAX_STEPS = 3


def main(task, n_users=80, model=READER):
    users = [u for u in load_proc(task)["users"] if u["split"] == "eval"][:n_users]
    audit = json.load(open(os.path.join(RES, f"{task}_audit.json")))
    cache = LLMCache(f"reader_{task}_{tag(model)}")
    jc = LLMCache(f"judge_{task}") if task == "lampqa" else None

    def U(u, ids):
        out = generate(cache, model, reader_prompt(task, u, ids), num_predict=NUM_PREDICT[task])
        return utility(task, u, out, jc)

    def search(u):
        P = min(M_GATE, u["n_pool"])
        a = [int(len(audit[u["uid"]][str(ci)]) > 0) for ci in range(P)]
        u_ref = U(u, tuple(range(u["n_pool"])))
        S, uS, calls, trace = (), U(u, ()), 2, []
        while uS < u_ref - EPS and len(S) < MAX_STEPS:
            ext = [(ci, U(u, tuple(sorted(S + (ci,))))) for ci in range(P) if ci not in S]
            calls += len(ext)
            suff = [(a[ci], -v, ci) for ci, v in ext if v >= u_ref - EPS]
            if suff:
                _, nv, ci = min(suff)
                S, uS = S + (ci,), -nv
                break
            v, na, ci = max((v, -a[ci], ci) for ci, v in ext)
            if v <= uS:
                break
            S, uS = S + (ci,), v
            trace.append(ci)
        return dict(uid=u["uid"], S=sorted(S), u=uS, u_ref=u_ref, u_none=U(u, ()),
                    exposure=sum(a[i] for i in S), sufficient=bool(uS >= u_ref - EPS), calls=calls)

    res = []
    for n, r in enumerate(pmap(search, users)):
        res.append(r)
        if n % 10 == 0:
            print("oracle", task, n, r, flush=True)
    json.dump(dict(eps=EPS, max_steps=MAX_STEPS, users=res), open(os.path.join(RES, f"{task}_oracle.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 80)
