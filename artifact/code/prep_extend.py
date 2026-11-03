"""Appends further evaluation users to an existing task file without touching the users already
prepared. It replays the random stream of prep.py (same seed, same order of draws), so the
appended users are exactly those that prep.py would have produced with a larger N_EVAL.

  python prep_extend.py TASK N_EXTRA
"""
import json
import os
import sys

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from common import K, M, PROC, SEED, record_text
from prep import load, tok
import hashlib


def main(task, n_extra):
    path = os.path.join(PROC, f"{task}.json")
    d = json.load(open(path))
    old = d["users"]
    emb = np.load(os.path.join(PROC, f"{task}_emb.npy"))
    rng = np.random.default_rng(SEED)
    users, seen = [], set()
    for u in load(task):
        h = hashlib.md5("||".join(sorted(p["text"] for p in u["profile"][:5])).encode()).hexdigest()
        if h in seen or len(u["profile"]) < K + 1:
            continue
        seen.add(h)
        users.append(u)
    assert len(users) == d["n_users_available"]
    order = rng.permutation(len(users))
    for u in old:                      # replay the per-user draws of prep.py
        rng.permutation(u["n_profile"])
    start = len(old)
    chosen = [users[i] for i in order[start: start + n_extra]]
    assert [users[i]["uid"] for i in order[:start]] == [u["uid"] for u in old]
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    new_emb = []
    for u in chosen:
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
            new_emb.append(E[pi])
        u.update(split="eval", n_profile=len(prof), cands=cands, n_pool=len(top), rand=[pos[i] for i in rnd])
        old.append(u)
    d["users"] = old
    json.dump(d, open(path, "w"))
    np.save(os.path.join(PROC, f"{task}_emb.npy"), np.concatenate([emb, np.asarray(new_emb, dtype=np.float32)]))
    print(task, "users now", len(old), "eval", sum(u["split"] == "eval" for u in old))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
