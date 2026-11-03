"""Matched calibrated redaction baseline for the LaMP-3 rating task.

The family keeps the same top-five retrieved records and the same rating field, but masks
review text when the on-device detector's flag count crosses a training-fixed threshold.
The thresholds are calibrated with CRC on the same held-out reader cohort as the admission
analysis.  This is a representation baseline, not a claim that detector scores are a
privacy ground truth.

  python calibrated_redaction.py [MODEL] [N_USERS] [N_SPLITS]
"""
import json
import os
import sys

import numpy as np

from common import K, NUM_PREDICT, RES, SEED, LLMCache, generate, load_proc, record_text, trunc, utility
from crc import crc_index, monotonize
from run import pmap, tag


FLAG_KEYS = ["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "NRP",
             "lex_health", "lex_finance", "lex_family", "lex_belief", "lex_place"]


def detector_score(det, uid, ci):
    return int(sum(int(det[uid][str(ci)].get(k, 0) > 0) for k in FLAG_KEYS))


def rating_prompt(user, det_scores, threshold):
    """Build the exact rating prompt while preserving ratings for masked records."""
    rows = []
    for ci in range(K):
        rating = f"(rating {user['cands'][ci]['score']})"
        if det_scores[ci] >= threshold:
            rows.append(rating + " [removed]")
        else:
            rows.append(record_text("lamp3", user["cands"][ci]))
    head = "Past reviews written by this user with the ratings they gave:\n" + \
        "\n".join(f"- {r}" for r in rows) + "\n\n"
    return (head + "What rating from 1 to 5 would this user give to the following "
            "review they wrote? Answer with a single digit and nothing else.\n"
            f"Review: {trunc(user['query'], 150)}\nRating:")


def pct(x):
    x = np.asarray(x, dtype=float)
    return [float(np.mean(x)), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main(model="qwen2.5:7b", n_users=150, n_splits=500):
    users = [u for u in load_proc("lamp3")["users"] if u["split"] == "eval"][:n_users]
    qwen = json.load(open(os.path.join(RES, "lamp3_audit.json")))
    gemma = json.load(open(os.path.join(RES, "lamp3_audit_gemma2_9b_evaltop.json")))
    det = json.load(open(os.path.join(RES, "lamp3_detect.json")))
    users = [u for u in users if u["uid"] in qwen and u["uid"] in gemma and
             all(str(ci) in qwen[u["uid"]] and str(ci) in gemma[u["uid"]] for ci in range(K))]
    if not users:
        raise RuntimeError("no evaluation users have both auditor labels")

    scores = {u["uid"]: [detector_score(det, u["uid"], ci) for ci in range(K)] for u in users}
    max_score = max(max(v) for v in scores.values())
    # threshold 0 is ratings-only, intermediate thresholds are progressively less redacted,
    # and +inf is the ordinary full-text top-five prompt.
    thresholds = list(range(max_score + 2)) + [10 ** 9]
    thresholds = sorted(set(thresholds))

    cache = LLMCache(f"redaction_lamp3_{tag(model)}")
    items = [(u, t) for u in users for t in thresholds]

    def work(item):
        u, threshold = item
        out = generate(cache, model, rating_prompt(u, scores[u["uid"]], threshold),
                       num_predict=NUM_PREDICT["lamp3"])
        return u["uid"], threshold, utility("lamp3", u, out)

    values = {}
    for j, (uid, threshold, value) in enumerate(pmap(work, items)):
        values.setdefault(uid, {})[str(threshold)] = float(value)
        if j % 250 == 0:
            print("redaction reader", j, len(items), flush=True)

    def flagged(uid, ci):
        # Union of the primary and second auditor, used consistently for this comparison.
        return int(bool(set(qwen[uid][str(ci)]) | set(gemma[uid][str(ci)])))

    losses = np.zeros((len(users), len(thresholds)), dtype=float)
    utilities = np.full_like(losses, np.nan)
    text_kept = np.zeros_like(losses)
    for i, u in enumerate(users):
        uid = u["uid"]
        for j, threshold in enumerate(thresholds):
            losses[i, j] = sum(flagged(uid, ci) for ci in range(K)
                                if scores[uid][ci] < threshold)
            text_kept[i, j] = sum(scores[uid][ci] < threshold for ci in range(K))
            utilities[i, j] = values[uid][str(threshold)]

    rng = np.random.default_rng(SEED + 210)
    split_rows = {str(a): [] for a in (0.2, 0.5)}
    for _ in range(n_splits):
        perm = rng.permutation(len(users))
        cal, test = perm[:len(users) // 2], perm[len(users) // 2:]
        env = monotonize(losses[cal])
        for alpha in (0.2, 0.5):
            j = crc_index(env, alpha, K)
            split_rows[str(alpha)].append({
                "index": int(j), "threshold": int(thresholds[j]),
                "calibration_loss": float(losses[cal, j].mean()),
                "test_loss": float(losses[test, j].mean()),
                "test_utility": float(utilities[test, j].mean()),
                "test_text_records_kept": float(text_kept[test, j].mean()),
                "test_text_records_masked": float(K - text_kept[test, j].mean()),
            })

    def summarize(rows):
        out = {}
        for key in ("test_loss", "test_utility", "test_text_records_kept", "test_text_records_masked",
                    "calibration_loss"):
            out[key] = pct([r[key] for r in rows])
        out["selected_index"] = pct([r["index"] for r in rows])
        out["selected_threshold"] = pct([r["threshold"] for r in rows])
        out["fraction_empty_text"] = float(np.mean([r["test_text_records_kept"] == 0 for r in rows]))
        return out

    fixed = {}
    for j, threshold in enumerate(thresholds):
        fixed[str(threshold)] = {
            "threshold": int(threshold), "text_masked": "all" if threshold == 0 else "detector_score >= threshold",
            "loss": float(losses[:, j].mean()), "utility": float(utilities[:, j].mean()),
            "text_records_kept": float(text_kept[:, j].mean()),
            "text_records_masked": float(K - text_kept[:, j].mean()),
        }

    out = {
        "task": "lamp3", "model": model, "n_users": len(users), "n_training": 120,
        "candidate_pool": "top-5 similarity records", "n_splits": n_splits, "alphas": [0.2, 0.5],
        "auditor_labels": "union of qwen2.5:7b and gemma2:9b; count is text self-disclosure only",
        "family": "ratings are always retained; review text is masked when detector flag count >= threshold",
        "thresholds": [int(t) for t in thresholds],
        "fixed": fixed,
        "calibrated": {a: summarize(rows) for a, rows in split_rows.items()},
        "paired_utility_definition": "same user, same query, same top-five records; only text representation changes",
        "caveat": "detector-based redaction is a matched representation baseline, not an independently validated privacy label",
    }
    path = os.path.join(RES, f"lamp3_calibrated_redaction_{tag(model)}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({"out": path, "n_users": len(users), "thresholds": thresholds,
                      "calibrated": out["calibrated"]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b",
         int(sys.argv[2]) if len(sys.argv) > 2 else 150,
         int(sys.argv[3]) if len(sys.argv) > 3 else 500)
