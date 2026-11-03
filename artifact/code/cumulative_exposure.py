"""Synthetic multi-arrival ranking stress on real, already-audited user profiles.

This is NOT observed multi-query evidence. Each arrival independently permutes the
same top-8 pool while the query-specific sensitivity scores remain fixed. Report both
unique flagged records and unique categories, with user as the calibration unit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import K, SEED
from crc import crc_index, monotonize


def unique_disclosures(sets, labels):
    record_ids = set().union(*(set(s) for s in sets))
    flagged = {i for i in record_ids if labels[i]}
    categories = set().union(*(set(labels[i]) for i in flagged))
    return len(flagged), len(categories)


def run(root, task, splits=300):
    read = lambda name: json.loads((root / "results" / f"{task}_{name}.json").read_text())
    rules, qwen, gemma, grid = read("rules"), read("audit"), read("audit_gemma2_9b_evaltop"), read("gate_report")["grid"]
    ids = [uid for uid, r in rules.items() if all(str(i) in qwen.get(uid, {}) and str(i) in gemma.get(uid, {})
                                                for i in range(len(r["s"])))]
    horizon = [1, 2, 5, 10]
    records, cats = {t: [] for t in horizon}, {t: [] for t in horizon}
    rng = np.random.default_rng(SEED + 92)
    for uid in ids:
        scores = rules[uid]["s"]
        labels = [set(qwen[uid][str(i)]) | set(gemma[uid][str(i)]) for i in range(len(scores))]
        orders = [rng.permutation(len(scores)) for _ in range(max(horizon))]
        for t in horizon:
            pairs = [unique_disclosures([[int(i) for i in order if scores[i] <= threshold][:K]
                                         for order in orders[:t]], labels) for threshold in grid]
            records[t].append([r for r, c in pairs])
            cats[t].append([c for r, c in pairs])
    records = {t: np.array(v, dtype=float) for t, v in records.items()}
    cats = {t: np.array(v, dtype=float) for t, v in cats.items()}
    env = {t: monotonize(v) for t, v in records.items()}
    acc = {}
    for _ in range(splits):
        perm = rng.permutation(len(ids))
        cal, test = perm[:len(ids)//2], perm[len(ids)//2:]
        for alpha in (.2, .5):
            j_single = crc_index(env[1][cal], alpha, K)
            for t in horizon:
                # Cumulative pool union has bound 8 (or the single-query cap at T=1).
                B = min(8, K*t)
                j_cumulative = crc_index(env[t][cal], alpha, B)
                for mode, j in [("single_prompt_calibration", j_single), ("cumulative_calibration", j_cumulative)]:
                    key = f"{alpha}|{t}|{mode}"
                    acc.setdefault(key, []).append((records[t][test, j].mean(), cats[t][test, j].mean()))
    raw_files = {"lamp3": "LaMP_3/dev_questions.json", "lamp7": "LaMP_7/dev_questions.json",
                 "lampqa": "LaMP_QA/lifestyle_validation.json", "lamp2": "LaMP_2/dev_questions_new.json"}
    raw = json.loads((root / "data/raw" / raw_files[task]).read_text())
    groups = {}
    for row in raw:
        fingerprint = hashlib.sha256(json.dumps(sorted(json.dumps(x, sort_keys=True)
                                  for x in row["profile"])).encode()).hexdigest()
        groups.setdefault(fingerprint, set()).add(row.get("input", row.get("question")))
    availability = {"raw_entries": len(raw), "exact_profile_groups": len(groups),
                    "exact_profile_groups_with_multiple_distinct_queries": sum(len(v) > 1 for v in groups.values()),
                    "qualification": "exact full-profile equality only; does not establish that different profiles belong to different humans"}
    return {"task": task, "n_users": len(ids), "splits": splits, "seed": SEED + 92,
            "observed_multi_query_availability": availability,
            "scope": "synthetic arrivals/rankings on real profiles; fixed already-audited top-8 pool, no new query text, no reader utility; not observed multi-query validation",
            "loss": "unique union-flagged record IDs within user; categories deduplicated separately",
            "cells": {key: {"mean_unique_flagged_records": float(np.mean(values, axis=0)[0]),
                             "mean_unique_categories": float(np.mean(values, axis=0)[1])}
                      for key, values in acc.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", nargs="+", choices=["lampqa", "lamp3", "lamp7", "lamp2"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for task in args.tasks:
        result = run(root, task)
        (root / f"results/{task}_cumulative_exposure.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps({"task": task, "n_users": result["n_users"], "status": "synthetic_stress_only"}), flush=True)
