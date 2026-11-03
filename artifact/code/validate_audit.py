"""Agreement between the primary auditor (qwen2.5:7b) and a second auditor (gemma2:9b)
on a fixed random sample of audited records per task. Writes results/audit_validation.json."""
import json
import os

import numpy as np

from common import RES, SENSITIVE_CATS

TK = {"lampqa": "qa", "lamp3": "three", "lamp7": "seven"}


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def main():
    out, mac = {}, {}
    for t, k in TK.items():
        p2 = os.path.join(RES, f"{t}_audit_gemma2_9b_sample.json")
        if not os.path.exists(p2):
            continue
        a1 = json.load(open(os.path.join(RES, f"{t}_audit.json")))
        a2 = json.load(open(p2))
        x, y, xs, ys = [], [], [], []
        for uid, d in a2.items():
            for ci, c2 in d.items():
                if uid in a1 and ci in a1[uid]:
                    c1 = a1[uid][ci]
                    x.append(int(len(c1) > 0)); y.append(int(len(c2) > 0))
                    xs.append(int(any(c in SENSITIVE_CATS for c in c1))); ys.append(int(any(c in SENSITIVE_CATS for c in c2)))
        x, y = np.array(x), np.array(y)
        r = dict(n=int(len(x)), agree=float(np.mean(x == y)), kappa=kappa(x, y), rate_primary=float(x.mean()),
                 rate_second=float(y.mean()), second_given_primary=float(y[x == 1].mean()) if x.sum() else None,
                 primary_given_second=float(x[y == 1].mean()) if y.sum() else None,
                 kappa_sens=kappa(xs, ys))
        out[t] = r
        mac[k + "AudN"] = r["n"]; mac[k + "AudAgree"] = f"{100*r['agree']:.0f}"; mac[k + "AudKappa"] = f"{r['kappa']:.2f}"
        mac[k + "AudRateTwo"] = f"{100*r['rate_second']:.0f}"; mac[k + "AudRateOne"] = f"{100*r['rate_primary']:.0f}"
        print(t, r)
    out["macros"] = mac
    json.dump(out, open(os.path.join(RES, "audit_validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
