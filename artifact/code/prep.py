"""Build per-task user samples and candidate pools.

For every sampled user: the query, the gold output, the top-M profile records by
dense similarity to the query (the candidate pool the gate re-ranks) and K records
drawn uniformly from the whole profile (for the random baseline).
Users are deduplicated so that one user contributes exactly one query.
"""
import hashlib
import json
import os
import sys

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from common import K, M, PROC, RAW, SEED, record_text

N_TRAIN = {"lamp3": 200, "lamp7": 200, "lampqa": 200, "lamp2": 120}
N_EVAL = {"lamp3": 400, "lamp7": 400, "lampqa": 400, "lamp2": 300}


def load(task):
    if task == "lamp3":
        q = json.load(open(os.path.join(RAW, "LaMP_3", "dev_questions.json")))
        g = {x["id"]: x["output"] for x in json.load(open(os.path.join(RAW, "LaMP_3", "dev_outputs.json")))["golds"]}
        for x in q:
            yield dict(uid=x["id"], query=x["input"].split("review: ", 1)[1], gold=g[x["id"]],
                       profile=[dict(text=p["text"], score=p["score"]) for p in x["profile"]])
    elif task == "lamp7":
        q = json.load(open(os.path.join(RAW, "LaMP_7", "dev_questions.json")))
        g = {x["id"]: x["output"] for x in json.load(open(os.path.join(RAW, "LaMP_7", "dev_outputs.json")))["golds"]}
        for x in q:
            yield dict(uid=x["id"], query=x["input"].split("before or after it: ", 1)[1], gold=g[x["id"]],
                       profile=[dict(text=p["text"]) for p in x["profile"]])
    elif task == "lamp2":
        q = json.load(open(os.path.join(RAW, "LaMP_2", "dev_questions_new.json")))
        g = {x["id"]: x["output"] for x in json.load(open(os.path.join(RAW, "LaMP_2", "dev_outputs_new.json")))["golds"]}
        for x in q:
            inp = x["input"]
            desc = inp.split("description: ", 1)[1] if "description: " in inp else inp.split(" ", 1)[1]
            yield dict(uid=x["id"], query=desc, gold=g[x["id"]],
                       profile=[dict(text=p["description"], tag=p["tag"]) for p in x["profile"]])
    else:
        q = json.load(open(os.path.join(RAW, "LaMP_QA", "lifestyle_validation.json")))
        for x in q:
            yield dict(uid=x["id"], query=x["question"], gold=x["narrative"], aspects=x["rubric_aspects"],
                       category=x["category"], profile=[dict(text=p["text"]) for p in x["profile"]])


def tok(s):
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split()]


def main(task):
    rng = np.random.default_rng(SEED)
    users, seen = [], set()
    for u in load(task):
        h = hashlib.md5("||".join(sorted(p["text"] for p in u["profile"][:5])).encode()).hexdigest()
        if h in seen or len(u["profile"]) < K + 1:
            continue
        seen.add(h)
        users.append(u)
    n_all = len(users)
    order = rng.permutation(n_all)
    n_tr, n_ev = N_TRAIN[task], N_EVAL[task]
    chosen = [users[i] for i in order[: n_tr + n_ev]]
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    out, embs = [], []
    for j, u in enumerate(chosen):
        prof = u.pop("profile")
        texts = [record_text(task, p) for p in prof]
        E = enc.encode(texts + [u["query"]], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        sim = E[:-1] @ E[-1]
        bm = BM25Okapi([tok(t) for t in texts]).get_scores(tok(u["query"]))
        top = list(np.argsort(-sim)[:M])
        rnd = [int(i) for i in rng.permutation(len(prof))[:K]]
        idx = top + [i for i in rnd if i not in top]
        pos = {pi: ci for ci, pi in enumerate(idx)}
        cands = []
        for ci, pi in enumerate(idx):
            c = dict(prof[pi])
            c.update(sim=float(sim[pi]), bm25=float(bm[pi]), in_pool=ci < len(top), pidx=int(pi))
            cands.append(c)
            embs.append(E[pi])
        u.update(split="train" if j < n_tr else "eval", n_profile=len(prof), cands=cands,
                 n_pool=len(top), rand=[pos[i] for i in rnd])
        out.append(u)
        if j % 100 == 0:
            print(task, j, len(prof), flush=True)
    json.dump(dict(task=task, n_users_available=n_all, users=out), open(os.path.join(PROC, f"{task}.json"), "w"))
    np.save(os.path.join(PROC, f"{task}_emb.npy"), np.asarray(embs, dtype=np.float32))
    print(task, "users", len(out), "of", n_all, "records", len(embs))


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
