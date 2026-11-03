"""Partial identification of human-labelled exposure, without zero-filling missing labels.

Intervals are deterministic missing-label bounds, NOT confidence intervals. They identify
what the available annotations can establish on a fixed evaluation cohort. Complete
human labels collapse the bounds; sparse labels may leave wide or uninformative intervals.
"""
import json
from pathlib import Path

import numpy as np

from crc import crc_index, monotonize


def loss_bounds(sets_by_user, labels_by_user):
    """labels_by_user: sequence of {candidate index: human binary label} mappings.

    The input sets are a fixed grid, including the empty set at column zero. Repeated
    candidate indices in a set are rejected instead of counted as independent evidence.
    """
    if not sets_by_user or len(sets_by_user) != len(labels_by_user):
        raise ValueError("A nonempty matched cohort is required")
    width = len(sets_by_user[0])
    if width < 1:
        raise ValueError("An empty-admission grid point is required")
    low, high = [], []
    for sets, labels in zip(sets_by_user, labels_by_user):
        if len(sets) != width or sets[0]:
            raise ValueError("Grid widths must agree and column zero must be empty")
        if any(v not in (0, 1) for v in labels.values()):
            raise ValueError("Human binary labels must be 0 or 1; omit unknowns")
        lo, hi = [], []
        for admitted in sets:
            if len(admitted) != len(set(admitted)):
                raise ValueError("Duplicate candidate in an admission set")
            known_positive = sum(labels[i] for i in admitted if i in labels)
            unknown = sum(i not in labels for i in admitted)
            lo.append(known_positive)
            hi.append(known_positive + unknown)
        low.append(lo)
        high.append(hi)
    return np.array(low, dtype=float), np.array(high, dtype=float)


def index_bounds(low, high, alpha, bound):
    """All full-label CRC indices lie in [pessimistic_index, optimistic_index]."""
    low, high = np.asarray(low), np.asarray(high)
    if low.ndim != 2 or low.shape != high.shape or not low.shape[0] or not low.shape[1]:
        raise ValueError("Matched nonempty loss matrices required")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("Missing labels must be bounded explicitly, not NaN")
    if bound <= 0 or alpha < 0 or np.any(low < 0) or np.any(high < low) or np.any(high > bound):
        raise ValueError("Invalid loss bounds")
    if np.any(low[:, 0]) or np.any(high[:, 0]):
        raise ValueError("Column zero must be the truly empty profile set")
    return (crc_index(monotonize(high), alpha, bound),
            crc_index(monotonize(low), alpha, bound))


def fixed_policy_summary(sets_by_user, human_labels, selected_indices):
    """Describe losses under a fixed existing policy; never tune it on these labels."""
    lo, hi = loss_bounds(sets_by_user, human_labels)
    chosen = np.asarray(selected_indices)
    if chosen.shape != (len(sets_by_user),) or not np.issubdtype(chosen.dtype, np.integer):
        raise ValueError("One integer fixed-policy index per user is required")
    if np.any(chosen < 0) or np.any(chosen >= lo.shape[1]):
        raise ValueError("Fixed-policy grid index out of range")
    rows = np.arange(len(chosen))
    known_records = sum(sum(i in labels for i in sets[j])
                        for sets, labels, j in zip(sets_by_user, human_labels, chosen))
    return {"n_users": len(chosen), "admitted_records": sum(len(s[j]) for s, j in zip(sets_by_user, chosen)),
            "human_labelled_admitted_records": known_records,
            "known_positive_admitted_records": int(lo[rows, chosen].sum()),
            "mean_loss_lower": float(lo[rows, chosen].mean()),
            "mean_loss_upper": float(hi[rows, chosen].mean()),
            "exact": bool(np.array_equal(lo[rows, chosen], hi[rows, chosen])),
            "interval_type": "deterministic missing-label bounds; not a confidence interval or population estimate"}


def cohort_inputs(results, key, gold=None):
    """Join only in private memory. Returned identifiers must never be exported."""
    results = Path(results)
    bytask = {}
    for ident, record in key.items():
        bytask.setdefault(record["task"], {}).setdefault(record["uid"], {})[int(record["rec"])] = (
            int(bool(gold[ident])) if gold is not None else None)
    for task, sampled in bytask.items():
        read = lambda suffix: json.loads((results / f"{task}_{suffix}.json").read_text())
        rules, qwen, gemma = read("rules"), read("audit"), read("audit_gemma2_9b_evaltop")
        ids = [uid for uid, row in rules.items() if all(str(i) in qwen.get(uid, {}) and str(i) in gemma.get(uid, {})
                                                      for i in range(len(row["s"])))]
        yield task, ids, rules, qwen, gemma, sampled


def packet_coverage(results, key):
    """Structural coverage of the packet, independent of whether humans filled it."""
    out = {}
    for task, ids, rules, qwen, gemma, sampled in cohort_inputs(results, key):
        out[task] = {
            "evaluation_users": len(ids),
            "users_represented_in_packet": sum(uid in sampled for uid in ids),
            "candidate_slots": sum(len(rules[uid]["s"]) for uid in ids),
            "candidate_slots_in_packet": sum(sum(i in sampled.get(uid, {}) for i in range(len(rules[uid]["s"]))) for uid in ids),
            "fully_covered_candidate_pools_in_packet": sum(all(i in sampled.get(uid, {}) for i in range(len(rules[uid]["s"]))) for uid in ids),
            "note": "packet membership only, not completed human labels"}
    return out


def evaluate_adjudicated(results, key, gold):
    """Two-fold comparison to union-calibrated cleared similarity on the fixed cohort.

    Folds use seed 20261098 before examining human label values. Union-calibrated policy
    indices are frozen before measuring sparse human labels; their missing-label bounds
    are reported on the opposite fold. Human CRC index intervals use calibration labels
    only. The output has no user identifiers, record indices, labels or prompts.
    """
    if set(gold) != set(key):
        raise ValueError("Complete adjudication of the prepared packet is required")
    out = {"scope": "finite-cohort partial-identification diagnostic, not a human population estimate",
           "seed": 20261098, "procedure": "two disjoint user folds; cleared similarity; original grid and scores fixed",
           "tasks": {}}
    for task, ids, rules, qwen, gemma, sampled in cohort_inputs(results, key, gold):
        sets = [rules[uid]["sets"]["cert_sim"] for uid in ids]
        human = [sampled.get(uid, {}) for uid in ids]
        lo, hi = loss_bounds(sets, human)
        union_labels = [{i: int(bool(set(qwen[uid][str(i)]) | set(gemma[uid][str(i)])))
                         for i in range(len(rules[uid]["s"]))} for uid in ids]
        union, _ = loss_bounds(sets, union_labels)
        permutation = np.random.default_rng(20261098).permutation(len(ids))
        folds = np.array_split(permutation, 2)
        if any(not len(f) for f in folds):
            raise ValueError("At least two evaluation users are required")
        task_out = {"n_users": len(ids), "cells": {}}
        for alpha in [.2, .5]:
            chosen = np.zeros(len(ids), dtype=int)
            fold_rows = []
            for f in [0, 1]:
                cal, test = folds[1-f], folds[f]
                current = crc_index(monotonize(union[cal]), alpha, 5)
                left, right = index_bounds(lo[cal], hi[cal], alpha, 5)
                chosen[test] = current
                fold_rows.append({"fold": f, "calibration_users": len(cal), "test_users": len(test),
                                  "union_calibrated_grid_index": current,
                                  "human_crc_grid_index_lower": left, "human_crc_grid_index_upper": right,
                                  "human_operating_point_identified": left == right,
                                  "change_from_union_index_lower": left-current,
                                  "change_from_union_index_upper": right-current})
            task_out["cells"][str(alpha)] = {"fixed_union_policy_human_loss": fixed_policy_summary(sets, human, chosen),
                                            "calibration_index_bounds": fold_rows}
        out["tasks"][task] = task_out
    return out
