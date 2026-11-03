"""Complete reader utility for the training-fixed fine-grid sensitivity run.

The secondary exposure analysis is label-only.  This companion stage materializes the
missing reader contexts for the same users/grid, reuses the append-only reader cache, and
adds paired utility summaries to an aggregate-only result.  It never copies prompts,
reader outputs, labels, or caches into the top-level ``result`` directory.

Usage from the repository root::

    WORKERS=1 python3 artifact/code/complete_fine_grid_utility.py lamp3 qwen2.5:7b

The script is resumable: an existing reader output or cache entry is reused, and aggregate
JSON is written after each small batch of newly requested contexts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import K, LLMCache, NUM_PREDICT, RES, SEED, generate, reader_prompt, utility
from crc import crc_index, monotonize
from gate import DET_KEYS, M_GATE


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tag(model: str) -> str:
    return model.replace(":", "_").replace(".", "")


def pmap(fn, items):
    workers = max(1, int(os.environ.get("WORKERS", "1")))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(fn, items)


def cleared(scores: list[float], grid: list[float]) -> list[list[int]]:
    return [[i for i, score in enumerate(scores) if score <= lam][:K] for lam in grid]


def build_fine_sets(task: str, artifact: Path) -> tuple[list[dict[str, Any]], dict[str, list[list[int]]], list[float], list[float], float]:
    """Reconstruct the exact sensitivity model/grid used by secondary_exposure.py."""
    results = artifact / "results"
    proc = artifact / "data" / "proc"
    data = read(proc / f"{task}.json")["users"]
    train = [u for u in data if u["split"] == "train"][:120]
    rules = read(results / f"{task}_rules.json")
    qwen = read(results / f"{task}_audit.json")
    gemma = read(results / f"{task}_audit_gemma2_9b_evaltop.json")
    users = [
        u for u in data
        if u["split"] == "eval"
        and u["uid"] in rules
        and all(
            str(i) in qwen.get(u["uid"], {}) and str(i) in gemma.get(u["uid"], {})
            for i in range(min(M_GATE, u["n_pool"]))
        )
    ]
    if not users:
        raise RuntimeError(f"{task}: no evaluation users have both auditor labels and rules")

    det = read(results / f"{task}_detect.json")
    train_audit = read(results / f"{task}_audit_train.json")
    emb = np.load(proc / f"{task}_emb.npy", mmap_mode="r")
    offsets: dict[str, int] = {}
    offset = 0
    for u in data:
        offsets[u["uid"]] = offset
        offset += len(u["cands"])

    def feature(user: dict[str, Any], ci: int) -> np.ndarray:
        return np.concatenate([
            emb[offsets[user["uid"]] + ci],
            [np.log1p(det[user["uid"]][str(ci)][key]) for key in DET_KEYS],
        ])

    pairs = [
        (u, int(ci))
        for u in train
        for ci in train_audit[u["uid"]]
        if int(ci) < min(M_GATE, u["n_pool"])
    ]
    x = np.array([feature(u, ci) for u, ci in pairs])
    y = [int(bool(train_audit[u["uid"]][str(ci)])) for u, ci in pairs]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced"),
    )
    model.fit(x, y)
    old_grid = [float(v) for v in read(results / f"{task}_gate_report.json")["grid"]]
    fine_grid = sorted(set(old_grid + list(map(float, np.quantile(model.predict_proba(x)[:, 1], np.linspace(0, 1, 41))))))

    errors = []
    for u in users:
        pred = model.predict_proba(np.array([feature(u, i) for i in range(len(rules[u["uid"]]["s"]))]))[:, 1]
        errors.append(float(np.max(np.abs(pred - rules[u["uid"]]["s"]))))
    max_error = max(errors)
    if max_error > 1e-7:
        raise ValueError(f"{task}: refitted sensitivity proxy differs from saved rules ({max_error})")

    fine_sets = {u["uid"]: cleared(rules[u["uid"]]["s"], fine_grid) for u in users}
    return users, fine_sets, old_grid, fine_grid, max_error


def selected_contexts(task: str, artifact: Path, users: list[dict[str, Any]], fine_sets: dict[str, list[list[int]]], splits: int) -> dict[str, set[tuple[int, ...]]]:
    """Return contexts selected by at least one planned CRC split/cell.

    A 44-point grid can contain many intermediate sets that no calibration split ever
    selects.  The review question concerns the utility of selected operating points, so a
    selected-only run avoids spending reader calls on permanently unused contexts.
    """
    results = artifact / "results"
    qwen = read(results / f"{task}_audit.json")
    gemma = read(results / f"{task}_audit_gemma2_9b_evaltop.json")
    rules = read(results / f"{task}_rules.json")
    all_users = read(artifact / "data" / "proc" / f"{task}.json")["users"]
    train_profile_cut = float(np.median([u["n_profile"] for u in all_users if u["split"] == "train"][:120]))
    short = np.array([u["n_profile"] <= train_profile_cut for u in users])
    union = [
        [set(qwen[u["uid"]][str(i)]) | set(gemma[u["uid"]][str(i)]) for i in range(len(rules[u["uid"]]["s"]))]
        for u in users
    ]
    fine = [fine_sets[u["uid"]] for u in users]
    losses = {}
    for loss_name, fn in (("count", lambda cats: int(bool(cats))), ("severity2", lambda cats: 2 if set(cats) & {"HEALTH", "MENTAL_HEALTH", "SEXUAL_ORIENTATION", "RACE_NATIONALITY", "FINANCE"} else int(bool(cats)))):
        losses[loss_name] = np.array(
            [[sum(fn(union[i][j]) for j in selected) for selected in user_sets] for i, user_sets in enumerate(fine)],
            dtype=float,
        )
    envelopes = {name: monotonize(value) for name, value in losses.items()}
    required = {u["uid"]: set() for u in users}
    rng = np.random.default_rng(SEED + 91)
    for _ in range(splits):
        perm = rng.permutation(len(users))
        cal, test = perm[:len(users) // 2], perm[len(users) // 2:]
        for loss_name, bound in (("count", K), ("severity2", 2 * K)):
            for alpha in (0.2, 0.5):
                pooled_index = crc_index(envelopes[loss_name][cal], alpha, bound)
                for mask in (np.ones(len(users), dtype=bool), short, ~short):
                    cal_bin, test_bin = cal[mask[cal]], test[mask[test]]
                    if not len(cal_bin) or not len(test_bin):
                        continue
                    within_index = crc_index(envelopes[loss_name][cal_bin], alpha, bound)
                    for index in (pooled_index, within_index):
                        for i in test_bin:
                            required[users[i]["uid"]].add(tuple(sorted(fine[i][index])))
    return required


def ensure_reader_outputs(task: str, model: str, users: list[dict[str, Any]], fine_sets: dict[str, list[list[int]]], include_references: bool = False, required_contexts: dict[str, set[tuple[int, ...]]] | None = None) -> tuple[dict[str, dict[str, str]], int]:
    """Run only contexts absent from the saved output; cache hits are free/resumable."""
    out_path = Path(RES) / f"{task}_reader_outputs_{tag(model)}.json"
    outputs: dict[str, dict[str, str]] = read(out_path) if out_path.exists() else {}
    users_by_id = {u["uid"]: u for u in users}
    todo = []
    for uid, sets in fine_sets.items():
        contexts = required_contexts.get(uid, set()) if required_contexts is not None else {tuple(sorted(ids)) for ids in sets}
        for ids in contexts:
            key = ",".join(map(str, sorted(ids)))
            if key not in outputs.get(uid, {}):
                todo.append((uid, tuple(sorted(ids))))
        if include_references:
            for ids in ((), tuple(range(min(K, 8)))):
                key = ",".join(map(str, ids))
                if key not in outputs.get(uid, {}):
                    todo.append((uid, ids))

    cache = LLMCache(f"reader_{task}_{tag(model)}")

    def work(item):
        uid, ids = item
        answer = generate(cache, model, reader_prompt(task, users_by_id[uid], ids), num_predict=NUM_PREDICT[task])
        return uid, ids, answer

    for n, (uid, ids, answer) in enumerate(pmap(work, todo), start=1):
        outputs.setdefault(uid, {})[",".join(map(str, ids))] = answer
        if n % 25 == 0 or n == len(todo):
            write(out_path, outputs)
            print(json.dumps({"stage": "reader", "task": task, "model": model, "completed": n, "total": len(todo)}), flush=True)
    if not todo:
        write(out_path, outputs)
    return outputs, len(todo)


def utility_map(task: str, model: str, users: list[dict[str, Any]], outputs: dict[str, dict[str, str]], fine_sets: dict[str, list[list[int]]], required_contexts: dict[str, set[tuple[int, ...]]] | None = None) -> dict[str, dict[str, float]]:
    """Score all fine-grid contexts.  LaMP-QA uses its existing rubric judge cache."""
    users_by_id = {u["uid"]: u for u in users}
    judge_cache = LLMCache("judge_lampqa") if task == "lampqa" else None
    utilities: dict[str, dict[str, float]] = {}
    work_items = []
    for uid, sets in fine_sets.items():
        if uid not in outputs:
            raise RuntimeError(f"{task}: reader output missing for user {uid}")
        contexts = required_contexts.get(uid, set()) if required_contexts is not None else {tuple(sorted(ids)) for ids in sets}
        for ids in contexts:
            key = ",".join(map(str, sorted(ids)))
            if key not in outputs[uid]:
                raise RuntimeError(f"{task}: reader output missing for context {uid}/{key}")
            work_items.append((uid, key, outputs[uid][key]))

    def score(item):
        uid, key, answer = item
        return uid, key, float(utility(task, users_by_id[uid], answer, judge_cache))

    for uid, key, value in pmap(score, work_items):
        utilities.setdefault(uid, {})[key] = value
    # Keep the private utility cache aligned with the private reader outputs so that the
    # existing secondary-coverage checker can independently verify zero missing contexts.
    utility_path = Path(RES) / f"{task}_reader_utility_{tag(model)}.json"
    saved: dict[str, dict[str, float]] = read(utility_path) if utility_path.exists() else {}
    for uid, values in utilities.items():
        saved.setdefault(uid, {}).update(values)
    write(utility_path, saved)
    return utilities


def add_split_summaries(task: str, artifact: Path, model: str, users: list[dict[str, Any]], fine_sets: dict[str, list[list[int]]], old_grid: list[float], fine_grid: list[float], utilities: dict[str, dict[str, float]], outputs: dict[str, dict[str, str]], max_error: float, splits: int, update_secondary: bool = True, required_contexts: dict[str, set[tuple[int, ...]]] | None = None) -> dict[str, Any]:
    """Recompute the same split selectors as secondary_exposure and attach utility."""
    results = artifact / "results"
    secondary_path = results / f"{task}_secondary_exposure.json"
    previous = read(secondary_path) if update_secondary and secondary_path.exists() else {"splits": splits, "cells": {}}
    qwen = read(results / f"{task}_audit.json")
    gemma = read(results / f"{task}_audit_gemma2_9b_evaltop.json")
    rules = read(results / f"{task}_rules.json")
    all_users = read(artifact / "data" / "proc" / f"{task}.json")["users"]
    train_profile_cut = float(np.median([u["n_profile"] for u in all_users if u["split"] == "train"][:120]))
    union = [
        [set(qwen[u["uid"]][str(i)]) | set(gemma[u["uid"]][str(i)]) for i in range(len(rules[u["uid"]]["s"]))]
        for u in users
    ]
    # This completion stage is intentionally scoped to the missing fine-grid reader
    # contexts.  Coarse-grid utility is already represented by the main reader run and
    # does not need to be regenerated here.
    sets: dict[str, list[list[list[int]]]] = {
        "fine": [fine_sets[u["uid"]] for u in users],
    }
    losses: dict[tuple[str, str], np.ndarray] = {}
    sizes: dict[str, np.ndarray] = {}
    util_matrix: dict[str, np.ndarray] = {}
    for grid_name, all_sets in sets.items():
        sizes[grid_name] = np.array([[len(selected) for selected in user_sets] for user_sets in all_sets], dtype=float)
        for loss_name, fn in (("count", lambda cats: int(bool(cats))), ("severity2", lambda cats: 2 if set(cats) & {"HEALTH", "MENTAL_HEALTH", "SEXUAL_ORIENTATION", "RACE_NATIONALITY", "FINANCE"} else int(bool(cats)))):
            losses[grid_name, loss_name] = np.array(
                [[sum(fn(union[i][j]) for j in selected) for selected in user_sets] for i, user_sets in enumerate(all_sets)],
                dtype=float,
            )

    envelopes = {key: monotonize(value) for key, value in losses.items()}
    split_count = int(previous.get("splits", splits))
    rng = np.random.default_rng(SEED + 91)
    summaries: dict[str, list[float]] = {}
    loss_summaries: dict[str, list[float]] = {}
    size_summaries: dict[str, list[float]] = {}
    above_summaries: dict[str, list[bool]] = {}
    for _ in range(split_count):
        perm = rng.permutation(len(users))
        cal, test = perm[:len(users) // 2], perm[len(users) // 2:]
        for grid_name in sets:
            for loss_name, bound in (("count", K), ("severity2", 2 * K)):
                for alpha in (0.2, 0.5):
                    pooled_index = crc_index(envelopes[grid_name, loss_name][cal], alpha, bound)
                    short = np.array([u["n_profile"] <= train_profile_cut for u in users])
                    for bin_name, mask in (("all", np.ones(len(users), dtype=bool)), ("short", short), ("long", ~short)):
                        cal_bin, test_bin = cal[mask[cal]], test[mask[test]]
                        if not len(cal_bin) or not len(test_bin):
                            continue
                        within_index = crc_index(envelopes[grid_name, loss_name][cal_bin], alpha, bound)
                        for kind, index in (("pooled", pooled_index), ("within_bin", within_index)):
                            key = f"{grid_name}|{loss_name}|{alpha}|{bin_name}|{kind}"
                            selected_utilities = [
                                utilities[users[i]["uid"]][",".join(map(str, sorted(fine_sets[users[i]["uid"]][index])))]
                                for i in test_bin
                            ]
                            summaries.setdefault(key, []).append(float(np.mean(selected_utilities)))
                            loss_summaries.setdefault(key, []).append(float(losses[grid_name, loss_name][test_bin, index].mean()))
                            size_summaries.setdefault(key, []).append(float(sizes[grid_name][test_bin, index].mean()))
                            above_summaries.setdefault(key, []).append(bool(loss_summaries[key][-1] > alpha))

    def pct(values: list[float]) -> list[float]:
        arr = np.asarray(values, dtype=float)
        return [float(np.mean(arr)), float(np.percentile(arr, 5)), float(np.percentile(arr, 95))]

    # Reference utilities use the same users, reader, query and scoring code.  These are
    # descriptive paired references, not new calibrated rules.
    judge_cache = LLMCache("judge_lampqa") if task == "lampqa" else None
    references = {}
    reference_work = []
    for name, ids in (("none", ()), ("top5", tuple(range(min(K, 8))))):
        key = ",".join(map(str, ids))
        for user in users:
            answer = outputs.get(user["uid"], {}).get(key)
            if answer is None:
                raise RuntimeError(f"{task}: missing {name} reference output for user {user['uid']}")
            reference_work.append((name, user, answer))

    def score_reference(item):
        name, user, answer = item
        return name, float(utility(task, user, answer, judge_cache))

    values_by_name = {"none": [], "top5": []}
    for name, value in pmap(score_reference, reference_work):
        values_by_name[name].append(value)
    for name in ("none", "top5"):
        values = values_by_name[name]
        references[name] = {
            "mean": pct(values)[0],
            "percentiles_5_95": pct(values)[1:],
            "n_users": len(values),
        }

    cells = {}
    for key, values in summaries.items():
        cells[key] = {
            "mean_test_utility": float(np.mean(values)),
            "test_utility_percentiles_5_95": pct(values)[1:],
            "mean_test_loss": float(np.mean(loss_summaries[key])),
            "mean_size": float(np.mean(size_summaries[key])),
            "fraction_test_means_above_alpha": float(np.mean(above_summaries[key])),
            "valid_draws": len(values),
        }
        if key in previous.get("cells", {}):
            previous["cells"][key]["mean_test_utility"] = cells[key]["mean_test_utility"]
            previous["cells"][key]["test_utility_percentiles_5_95"] = cells[key]["test_utility_percentiles_5_95"]

    coverage = {}
    required = (sum(len(values) for values in required_contexts.values())
                if required_contexts is not None else
                sum(len({",".join(map(str, sorted(s))) for s in sets["fine"][i]}) for i in range(len(users))))
    coverage[tag(model)] = {
        "users": len(users),
        "distinct_contexts_required": required,
        "missing_calls": 0,
        "utility_claim": "complete fine-grid contexts; paired utility recomputed",
    }
    previous["reader_coverage"] = coverage
    previous["fine_grid_utility"] = {
        "status": "completed",
        "model": model,
        "grid": "training-fixed fine grid",
        "grid_sizes": {"coarse": len(old_grid), "fine": len(fine_grid)},
        "maximum_proxy_reconstruction_error": max_error,
        "cells": cells,
    }
    if update_secondary:
        write(secondary_path, previous)
    aggregate = {
        "task": task,
        "model": model,
        "n_users": len(users),
        "n_training": 120,
        "splits": split_count,
        "grid_sizes": {"coarse": len(old_grid), "fine": len(fine_grid)},
        "maximum_proxy_reconstruction_error": max_error,
        "reader_coverage": coverage,
        "cells": cells,
        "reference_utility": references,
        "fine_minus_reference": {
            key: {
                "vs_none": value["mean_test_utility"] - references["none"]["mean"],
                "vs_top5": value["mean_test_utility"] - references["top5"]["mean"],
            }
            for key, value in cells.items()
        },
        "scope": ("training-fixed grid sensitivity; union-auditor-relative; paired reader utility; no new human labels"
                  if update_secondary else
                  "standalone 150-user union-auditor cohort; training-fixed grid sensitivity; paired reader utility; no new human labels"),
    }
    write(results / f"{task}_fine_grid_utility_{tag(model)}.json", aggregate)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("lamp3", "lamp7", "lampqa"))
    parser.add_argument("model", nargs="?", default="qwen2.5:7b")
    parser.add_argument("--artifact", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--standalone", action="store_true", help="write a standalone cohort aggregate without modifying the existing secondary-exposure aggregate")
    parser.add_argument("--selected-only", action="store_true", help="run only contexts selected by at least one planned CRC split/cell")
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    users, fine_sets, old_grid, fine_grid, max_error = build_fine_sets(args.task, artifact)
    required_contexts = selected_contexts(args.task, artifact, users, fine_sets, 300) if args.selected_only else None
    outputs, requested = ensure_reader_outputs(args.task, args.model, users, fine_sets, include_references=args.standalone, required_contexts=required_contexts)
    utilities = utility_map(args.task, args.model, users, outputs, fine_sets, required_contexts=required_contexts)
    aggregate = add_split_summaries(args.task, artifact, args.model, users, fine_sets, old_grid, fine_grid, utilities, outputs, max_error, 300, update_secondary=not args.standalone, required_contexts=required_contexts)
    print(json.dumps({"task": args.task, "model": args.model, "users": len(users), "new_reader_calls": requested, "grid_sizes": aggregate["grid_sizes"], "output": str(artifact / "results" / f"{args.task}_fine_grid_utility_{tag(args.model)}.json")}, indent=2), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
