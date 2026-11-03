"""Materialize training-only auditor labels from the resumable full audit outputs.

The main audit stage stores one aggregate mapping per task.  Gate fitting and the secondary
exposure script intentionally consume a separate training-only file so their user split is
explicit and cannot accidentally include evaluation labels.
"""
import json
import os
import sys

from common import RES, load_proc


def main(tasks):
    for task in tasks:
        full = json.load(open(os.path.join(RES, f"{task}_audit.json")))
        train = {u["uid"] for u in load_proc(task)["users"] if u["split"] == "train"}
        out = {uid: full[uid] for uid in train if uid in full}
        expected = len(train)
        if len(out) != expected:
            raise RuntimeError(f"{task}: expected {expected} train users, found {len(out)}")
        json.dump(out, open(os.path.join(RES, f"{task}_audit_train.json"), "w"))
        print(task, "train users", len(out), "records", sum(len(v) for v in out.values()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["lamp3", "lamp7", "lampqa"])
