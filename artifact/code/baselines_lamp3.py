"""LaMP-3 field-level baselines (round-3 plan): the top-5 records by similarity with
  ratings  : text removed, rating kept
  redact_det : text of records flagged by the on-device detector replaced by "[removed]"
  redact_aud : text of records flagged by either auditor replaced by "[removed]" (not deployable)
Reader given as argument (Ollama model name); same prompt template as the main reader.

  python baselines_lamp3.py MODEL N_USERS
"""
import json
import os
import sys

import numpy as np

from common import K, NUM_PREDICT, RES, SEED, LLMCache, generate, load_proc, utility
from run import pmap, tag

FLAG_KEYS = ["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "NRP",
             "lex_health", "lex_finance", "lex_family", "lex_belief", "lex_place"]


def prompt(user, recs):
    head = ("Past reviews written by this user with the ratings they gave:\n"
            + "\n".join(f"- {r}" for r in recs) + "\n\n")
    from common import trunc
    return (head + "What rating from 1 to 5 would this user give to the following "
            "review they wrote? Answer with a single digit and nothing else.\n"
            f"Review: {trunc(user['query'], 150)}\nRating:")


def main(model, n):
    from common import record_text
    users = [u for u in load_proc("lamp3")["users"] if u["split"] == "eval"][:n]
    aq = json.load(open(os.path.join(RES, "lamp3_audit.json")))
    ag = json.load(open(os.path.join(RES, "lamp3_audit_gemma2_9b_evaltop.json")))
    det = json.load(open(os.path.join(RES, "lamp3_detect.json")))
    util = json.load(open(os.path.join(RES, f"lamp3_reader_utility_{tag(model)}.json")))
    cache = LLMCache(f"reader_lamp3_{tag(model)}")
    items, meta = [], []
    for u in users:
        ids = list(range(K))
        fa = {ci: int(len(aq[u["uid"]][str(ci)]) > 0 or len(ag[u["uid"]][str(ci)]) > 0) for ci in ids}
        fd = {ci: int(any(det[u["uid"]][str(ci)][k] > 0 for k in FLAG_KEYS)) for ci in ids}
        rate = lambda ci: f"(rating {u['cands'][ci]['score']})"
        full = lambda ci: record_text("lamp3", u["cands"][ci])
        variants = {
            "ratings": ([rate(ci) for ci in ids], 0),
            "redact_det": ([rate(ci) + " [removed]" if fd[ci] else full(ci) for ci in ids],
                           sum(fa[ci] for ci in ids if not fd[ci])),
            "redact_aud": ([rate(ci) + " [removed]" if fa[ci] else full(ci) for ci in ids], 0),
        }
        for v, (recs, expo) in variants.items():
            items.append(prompt(u, recs)); meta.append((u, v, expo))
    outs = list(pmap(lambda p: generate(cache, model, p, num_predict=NUM_PREDICT["lamp3"]), items))
    res = {}
    for (u, v, expo), o in zip(meta, outs):
        res.setdefault(v, []).append((utility("lamp3", u, o), expo))
    none = np.array([util[u["uid"]][""] for u in users])
    top5 = np.array([util[u["uid"]][",".join(map(str, range(K)))] for u in users])
    rng = np.random.default_rng(SEED)
    out = dict(model=model, n=len(users), none=float(none.mean()), top5=float(top5.mean()))
    B = [rng.integers(0, len(users), len(users)) for _ in range(2000)]
    gain = np.array([top5[b].mean() - none[b].mean() for b in B])
    for v, rows in res.items():
        uv = np.array([r[0] for r in rows]); ev = np.array([r[1] for r in rows])
        kept_bs = np.array([(uv[b].mean() - none[b].mean()) / g if g > 1e-9 else np.nan for b, g in zip(B, gain)])
        d_bs = np.array([uv[b].mean() - top5[b].mean() for b in B])
        plug = float((uv.mean() - none.mean()) / (top5.mean() - none.mean()))
        out[v] = dict(util=float(uv.mean()), exposure=float(ev.mean()),
                      util_ci=[float(uv.mean()), float(np.percentile([uv[b].mean() for b in B], 2.5)),
                               float(np.percentile([uv[b].mean() for b in B], 97.5))],
                      kept=[plug, float(np.nanpercentile(kept_bs, 2.5)), float(np.nanpercentile(kept_bs, 97.5))],
                      kept_90=[plug, float(np.nanpercentile(kept_bs, 5)), float(np.nanpercentile(kept_bs, 95))],
                      vs_top5_90=[float(uv.mean() - top5.mean()), float(np.percentile(d_bs, 5)), float(np.percentile(d_bs, 95))])
        # same declared rules as analyze.py: non-inferior if the lower 90% bound of the share kept exceeds 75%,
        # significant loss if the 90% interval of the difference to top-5 lies below zero
        out[v]["noninferior"] = bool(out[v]["kept_90"][1] > 0.75)
        out[v]["sig_loss"] = bool(out[v]["vs_top5_90"][2] < 0)
    out["gain"] = [float(top5.mean() - none.mean()), float(np.percentile(gain, 2.5)), float(np.percentile(gain, 97.5))]
    json.dump(out, open(os.path.join(RES, f"lamp3_baselines_{tag(model)}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
