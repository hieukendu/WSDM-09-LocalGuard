"""Exploratory, aggregate-only weighted/bin/finer-grid analyses; no model calls.

Uses fixed training-only profile-length cutpoints and refits the existing sensitivity
proxy solely to recover training score quantiles. Asserts equivalence to saved scores.
All exported results are cohort aggregates, never record indices or sensitive labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import K, SENSITIVE_CATS, SEED
from crc import crc_index, monotonize
from gate import DET_KEYS, M_GATE


def read(path):
    return json.loads(path.read_text())


def cleared(scores, grid):
    return [[i for i, s in enumerate(scores) if s <= lam][:K] for lam in grid]


def weighted_record(cats):
    # Normative sensitivity, not an empirically validated severity scale.
    return 2 if set(cats) & SENSITIVE_CATS else int(bool(cats))


def run(task, root, splits):
    res, proc = root / "results", root / "data/proc"
    data = read(proc / f"{task}.json")["users"]
    train = [u for u in data if u["split"] == "train"][:120]
    rules = read(res / f"{task}_rules.json")
    qwen = read(res / f"{task}_audit.json")
    gemma = read(res / f"{task}_audit_gemma2_9b_evaltop.json")
    users = [u for u in data if u["split"] == "eval" and u["uid"] in rules and
             all(str(i) in qwen.get(u["uid"], {}) and str(i) in gemma.get(u["uid"], {})
                 for i in range(min(M_GATE, u["n_pool"]))) ]
    union = [[set(qwen[u["uid"]][str(i)]) | set(gemma[u["uid"]][str(i)])
              for i in range(len(rules[u["uid"]]["s"]))] for u in users]
    # Reconstruct training features exactly as gate.py; no evaluation labels in grid.
    det = read(res / f"{task}_detect.json")
    atr = read(res / f"{task}_audit_train.json")
    emb = np.load(proc / f"{task}_emb.npy", mmap_mode="r")
    off, offset = {}, 0
    for u in data:
        off[u["uid"]] = offset
        offset += len(u["cands"])

    def feature(u, ci):
        return np.concatenate([emb[off[u["uid"]] + ci],
                               [np.log1p(det[u["uid"]][str(ci)][k]) for k in DET_KEYS]])

    pairs = [(u, int(ci)) for u in train for ci in atr[u["uid"]]
             if int(ci) < min(M_GATE, u["n_pool"])]
    x = np.array([feature(u, ci) for u, ci in pairs])
    y = [int(bool(atr[u["uid"]][str(ci)])) for u, ci in pairs]
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=2000, class_weight="balanced"))
    model.fit(x, y)
    oldgrid = read(res / f"{task}_gate_report.json")["grid"]
    newgrid = sorted(set(oldgrid + list(map(float, np.quantile(model.predict_proba(x)[:, 1], np.linspace(0, 1, 41))))))
    errors = []
    for u in users:
        pred = model.predict_proba(np.array([feature(u, i) for i in range(len(rules[u["uid"]]["s"]))]))[:, 1]
        errors.append(float(np.max(np.abs(pred - rules[u["uid"]]["s"]))))
    if max(errors) > 1e-7:
        raise ValueError("Refitted proxy differs from original; do not report a grid-only sensitivity")
    sets = {"coarse": [rules[u["uid"]]["sets"]["cert_sim"] for u in users],
            "fine": [cleared(rules[u["uid"]]["s"], newgrid) for u in users]}
    loss, size = {}, {}
    for grid, allsets in sets.items():
        size[grid] = np.array([[len(s) for s in ss] for ss in allsets])
        for name, fn in [("count", lambda c: int(bool(c))), ("severity2", weighted_record)]:
            loss[grid, name] = np.array([[sum(fn(union[i][j]) for j in s) for s in ss]
                                        for i, ss in enumerate(allsets)], dtype=float)
    envelope = {key: monotonize(value) for key, value in loss.items()}
    cut = float(np.median([u["n_profile"] for u in train]))
    short = np.array([u["n_profile"] <= cut for u in users])
    out = {"task": task, "n_users": len(users), "n_training": len(train), "splits": splits,
           "seed": SEED + 91, "scope": "exploratory; union-auditor-relative; no human validation",
           "weights": "2 for any special-category flag, 1 for other flagged records, 0 otherwise; maximum per record, not sum; normative sensitivity only",
           "severity2_bound": 2*K, "count_bound": K,
           "profile_length_cut_training_median": cut,
           "bin_counts": {"short": int(short.sum()), "long": int((~short).sum())},
           "grid_sizes": {"coarse": len(oldgrid), "fine": len(newgrid)},
           "maximum_proxy_reconstruction_error": max(errors), "cells": {}, "reader_coverage": {}}
    rng = np.random.default_rng(SEED + 91)
    acc, sizes, ncal = {}, {}, {}
    for _ in range(splits):
        perm = rng.permutation(len(users))
        cal, test = perm[:len(users)//2], perm[len(users)//2:]
        for grid in sets:
            for name, B in [("count", K), ("severity2", 2*K)]:
                for alpha in (.2, .5):
                    jp = crc_index(envelope[grid, name][cal], alpha, B)
                    for binname, mask in [("all", np.ones(len(users), dtype=bool)), ("short", short), ("long", ~short)]:
                        cb, tb = cal[mask[cal]], test[mask[test]]
                        if not len(cb) or not len(tb):
                            continue
                        jb = crc_index(envelope[grid, name][cb], alpha, B)
                        for kind, j in [("pooled", jp), ("within_bin", jb)]:
                            key = f"{grid}|{name}|{alpha}|{binname}|{kind}"
                            acc.setdefault(key, []).append(float(loss[grid, name][tb, j].mean()))
                            sizes.setdefault(key, []).append(float(size[grid][tb, j].mean()))
                            ncal.setdefault(key, []).append(len(cal) if kind == "pooled" else len(cb))
    for key, values in acc.items():
        alpha = float(key.split("|")[2])
        B = K if key.split("|")[1] == "count" else 2*K
        out["cells"][key] = {"mean_test_loss": float(np.mean(values)), "mean_size": float(np.mean(sizes[key])),
                            "fraction_test_means_above_alpha": float(np.mean(np.array(values) > alpha)),
                            "n_cal_min": min(ncal[key]), "n_cal_max": max(ncal[key]),
                            "minimum_n_for_nonempty_crc_zero_empirical_loss": int(np.ceil(B/alpha)-1),
                            "draws_with_no_nonempty_crc_possible": int(sum(B/(n+1) > alpha for n in ncal[key])),
                            "valid_draws": len(values)}
    for path in sorted(res.glob(f"{task}_reader_utility_*.json")):
        util = read(path)
        required = missing = covered_users = 0
        for i, u in enumerate(users):
            if u["uid"] not in util:
                continue
            covered_users += 1
            contexts = {",".join(map(str, sorted(s))) for s in sets["fine"][i]}
            required += len(contexts)
            missing += sum(k not in util[u["uid"]] for k in contexts)
        out["reader_coverage"][path.stem.removeprefix(task + "_reader_utility_")] = {
            "users": covered_users, "distinct_contexts_required": required, "missing_calls": missing,
            "utility_claim": "not estimated; incomplete fine-grid contexts" if missing else "complete contexts available; utility not recomputed here"}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--splits", type=int, default=300)
    p.add_argument("tasks", nargs="+", choices=["lampqa", "lamp3", "lamp7", "lamp2"])
    args = p.parse_args()
    if args.splits < 1:
        p.error("--splits must be positive")
    for task in args.tasks:
        result = run(task, args.artifact, args.splits)
        (args.artifact / f"results/{task}_secondary_exposure.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps({k: result[k] for k in ["task", "n_users", "bin_counts", "grid_sizes", "reader_coverage"]}), flush=True)


if __name__ == "__main__":
    main()
