"""Writes a blank hand-check sheet (120 records, stratified by task and auditor label) for a
human annotator to validate the auditor. The sheet holds benchmark text: keep it private."""
import csv
import json
import os

import numpy as np

from common import RES, load_proc, record_text

rng = np.random.default_rng(11)
os.makedirs(os.path.join(RES, "handcheck"), exist_ok=True)
rows = []
for t in ("lampqa", "lamp3", "lamp7"):
    a = json.load(open(os.path.join(RES, f"{t}_audit.json")))
    users = {u["uid"]: u for u in load_proc(t)["users"] if u["split"] == "eval"}
    pos = [(uid, ci) for uid, d in a.items() if uid in users for ci, c in d.items() if c]
    neg = [(uid, ci) for uid, d in a.items() if uid in users for ci, c in d.items() if not c]
    for lab, pool in ((1, pos), (0, neg)):
        for i in rng.permutation(len(pool))[:20]:
            uid, ci = pool[i]
            rows.append(dict(task=t, uid=uid, rec=ci, text=record_text(t, users[uid]["cands"][int(ci)]),
                             auditor_categories=";".join(a[uid][ci]), human_discloses="", human_categories=""))
order = rng.permutation(len(rows))
with open(os.path.join(RES, "handcheck", "handcheck_sheet.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for i in order:
        r = dict(rows[i]); r["auditor_categories"] = ""   # blind: the key is kept separately
        w.writerow(r)
with open(os.path.join(RES, "handcheck", "handcheck_key.json"), "w") as f:
    json.dump([{k: rows[i][k] for k in ("task", "uid", "rec", "auditor_categories")} for i in order], f, indent=0)
print(len(rows), "rows")
