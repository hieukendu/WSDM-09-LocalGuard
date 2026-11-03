"""Collect aggregate results from the additional review experiments.

This collector deliberately copies only aggregate JSON/CSV/Markdown files into the root
``result`` folder.  Raw benchmark rows, prompts, row-level labels, and LLM caches stay in
the ignored artifact workspace.  It also audits the parser contract over the actual cache
response units, including batched auditor responses.

Run from the repository root after each completed experiment, or at the end:
    python3 artifact/code/collect_review_results.py
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any


CATS = {
    "AGE_GENDER", "RACE_NATIONALITY", "LOCATION", "APPEARANCE", "PARTNER", "FINANCE",
    "OCCUPATION", "FAMILY", "HEALTH", "MENTAL_HEALTH", "EDUCATION", "PET",
    "SEXUAL_ORIENTATION", "NAME_CONTACT",
}
TASKS = ("lamp3", "lamp7", "lampqa")
BATCH_SIZES = (5, 8, 16)
CROSS_BATCH_SIZES = (16, 24, 32)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_response(raw: str) -> tuple[bool, int]:
    """Return (valid, known_record_count) for one cached auditor response."""
    try:
        obj = json.loads(raw)
    except Exception:
        return False, 0
    if isinstance(obj, dict) and isinstance(obj.get("categories"), list):
        # Unknown category tokens are filtered by common._cats; they are a label-schema
        # event, not a malformed JSON response.  Keep parser-failure accounting separate.
        return all(isinstance(x, str) for x in obj["categories"]), 1
    labels = obj.get("labels") if isinstance(obj, dict) else None
    if not isinstance(labels, list) or not labels:
        return False, 0
    known = 0
    valid = True
    for label in labels:
        if not isinstance(label, dict) or not isinstance(label.get("categories"), list):
            valid = False
            continue
        known += 1
        valid = valid and all(isinstance(x, str) for x in label["categories"])
    return valid, known


def response_label_count(raw: str) -> int | None:
    """Return the number of labels represented by a structurally valid response."""
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("categories"), list):
        return 1
    labels = obj.get("labels") if isinstance(obj, dict) else None
    return len(labels) if isinstance(labels, list) else None


def audit_key_map(repo: Path, task: str, model: str, evaltop: bool,
                  eval_limit: int = 300) -> dict[str, list[tuple[str, int]]]:
    """Reconstruct cache keys so malformed batches can be conservatively attached to records.

    The cache intentionally stores hashes rather than prompts.  The public preprocessing and
    prompt code are available locally, so the key map can be rebuilt without exporting text.
    We include the batch sizes used during this run and the two interrupted warm-up runs.
    """
    code = repo / "artifact" / "code"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from common import (AUDIT_BATCH_PROMPT, AUDIT_MULTI_BATCH_PROMPT, AUDIT_PROMPT, LLMCache,
                        N_TRAIN_USE, PROC, record_text)
    from run import N_READER, audit_ids

    raw_users = json.loads((Path(PROC) / f"{task}.json").read_text())["users"]
    train = [u for u in raw_users if u["split"] == "train"][:N_TRAIN_USE]
    evalu = [u for u in raw_users if u["split"] == "eval"][:eval_limit]
    users = train + evalu
    todo: list[tuple[Any, int]] = []
    n_ev = 0
    for u in users:
        if evaltop:
            if u["split"] == "eval":
                todo += [(u, ci) for ci in audit_ids(u, with_rand=False)]
            continue
        rd = u["split"] == "eval" and n_ev < N_READER
        n_ev += int(u["split"] == "eval")
        todo += [(u, ci) for ci in audit_ids(u, with_rand=rd)]

    out: dict[str, list[tuple[str, int]]] = {}
    def put(prompt: str, opts: dict[str, Any], rows: list[tuple[Any, int]]) -> None:
        ck = dict(opts, fmt="json")
        key = LLMCache.key(model, prompt, ck)
        out.setdefault(key, []).extend((u["uid"], int(ci)) for u, ci in rows)

    one_opts = {"temperature": 0.0, "num_predict": 40, "seed": 0}
    for u, ci in todo:
        put(AUDIT_PROMPT.format(text=record_text(task, u["cands"][ci])), one_opts, [(u, ci)])
    for batch_size in BATCH_SIZES:
        current_uid, current = None, []
        batches: list[list[tuple[Any, int]]] = []
        for row in todo:
            u, ci = row
            if current_uid != u["uid"] or len(current) >= batch_size:
                if current:
                    batches.append(current)
                current_uid, current = u["uid"], []
            current.append(row)
        if current:
            batches.append(current)
        for batch in batches:
            texts = "\n".join(f'{i + 1}. \"\"\"{record_text(task, u["cands"][ci])}\"\"\"'
                               for i, (u, ci) in enumerate(batch))
            opts = {"temperature": 0.0, "num_predict": min(30 * len(batch) + 20, 16 * len(batch) + 48), "seed": 0}
            put(AUDIT_BATCH_PROMPT.format(texts=texts), opts, batch)
    # Gemma's faster schedule may group independent records across users.  Map those keys too
    # so conservative parser accounting remains record-aware for that run.
    for batch_size in CROSS_BATCH_SIZES:
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            texts = "\n".join(f'{i + 1}. \"\"\"{record_text(task, u["cands"][ci])}\"\"\"'
                               for i, (u, ci) in enumerate(batch))
            opts = {"temperature": 0.0, "num_predict": min(30 * len(batch) + 20, 16 * len(batch) + 48), "seed": 0}
            put(AUDIT_MULTI_BATCH_PROMPT.format(texts=texts), opts, batch)
    return out


def cache_accounting(repo: Path, cache_dir: Path) -> tuple[dict[str, Any], dict[str, set[tuple[str, int]]]]:
    rows: dict[str, Any] = {}
    malformed_records: dict[str, set[tuple[str, int]]] = {}
    for task in TASKS:
        for tag, label in (("qwen25_7b", "qwen2.5:7b"), ("gemma2_9b", "gemma2:9b"),
                           ("llama33_70b", "llama3.3:70b")):
            path = cache_dir / f"audit_{task}_{tag}.jsonl"
            key = f"{task}|{label}"
            rec = {
                "task": task, "auditor": label, "cache": str(path), "exists": path.exists(),
                "response_units": 0, "valid_response_units": 0, "malformed_response_units": 0,
                "known_record_units": 0, "valid_record_units": 0,
                "malformed_record_units_unknown": 0,
                "cardinality_mismatch_response_units": 0,
                "overflow_label_units": 0,
                "short_label_response_units": 0,
            }
            if path.exists():
                evaltop = tag in ("gemma2_9b", "llama33_70b")
                key_map: dict[str, list[tuple[str, int]]] = {}
                # Different stages intentionally use 150/100 held-out users while the primary
                # audit uses 300.  The union keeps parser accounting record-aware for either.
                for eval_limit in (100, 150, 300):
                    for cache_key, rows_for_key in audit_key_map(repo, task, label, evaltop, eval_limit).items():
                        key_map.setdefault(cache_key, []).extend(rows_for_key)
                stream = f"{task}|{label}"
                malformed_records[stream] = set()
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        entry = json.loads(line)
                        val = entry.get("v", "")
                        cache_key = entry.get("k")
                    except Exception:
                        val, cache_key = "", None
                    valid, known = parse_response(val)
                    rec["response_units"] += 1
                    rec["known_record_units"] += known
                    mapped = list(set(key_map.get(cache_key or "", [])))
                    observed = response_label_count(val)
                    expected = len(mapped)
                    if valid and expected and observed is not None and observed != expected:
                        rec["cardinality_mismatch_response_units"] += 1
                        if observed > expected:
                            rec["overflow_label_units"] += observed - expected
                        else:
                            rec["short_label_response_units"] += 1
                    if valid:
                        rec["valid_response_units"] += 1
                        rec["valid_record_units"] += known
                    else:
                        rec["malformed_response_units"] += 1
                        # A malformed batch has no prompt stored in the released cache, so its
                        # exact record multiplicity is intentionally not guessed.
                        malformed_records[stream].update(mapped)
                        if not mapped:
                            rec["malformed_record_units_unknown"] += 1
                rec["malformed_response_rate"] = (rec["malformed_response_units"] /
                                                   rec["response_units"] if rec["response_units"] else None)
                rec["malformed_record_units_mapped"] = len(malformed_records[stream])
            else:
                rec["malformed_response_rate"] = None
            rows[key] = rec
    return rows, malformed_records


def output_label_summary(results: Path, malformed_records: dict[str, set[tuple[str, int]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in TASKS:
        for tag, label in (("qwen25_7b", "qwen2.5:7b"), ("gemma2_9b_evaltop", "gemma2:9b"),
                           ("llama33_70b_evaltop", "llama3.3:70b")):
            path = results / f"{task}_audit_{tag}.json" if tag != "qwen25_7b" else results / f"{task}_audit.json"
            if not path.exists():
                continue
            d = read_json(path)
            records = [cats for user in d.values() for cats in user.values()]
            stream = f"{task}|{label}"
            bad = malformed_records.get(stream, set())
            actual_rows = [(uid, int(ci), cats) for uid, user in d.items()
                           for ci, cats in user.items()]
            actual_flagged = sum(bool(cats) for _, _, cats in actual_rows)
            conservative_flagged = sum(bool(cats) or (uid, ci) in bad for uid, ci, cats in actual_rows)
            out[f"{task}|{label}"] = {
                "output": str(path), "users": len(d), "record_labels": len(records),
                "flagged_records": int(sum(bool(c) for c in records)),
                "flagged_rate": float(sum(bool(c) for c in records) / len(records)) if records else None,
                "conservative_malformed_records_mapped": len(bad),
                "conservative_flagged_records": int(conservative_flagged),
                "conservative_flagged_rate": float(conservative_flagged / len(records)) if records else None,
                "conservative_rate_delta": float((conservative_flagged - actual_flagged) / len(records)) if records else None,
                "categories": {c: int(sum(c in cats for cats in records)) for c in sorted(CATS)},
            }
    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    artifact = repo / "artifact"
    results = artifact / "results"
    cache_dir = results / "llm_cache"
    out = repo / "result"
    out.mkdir(parents=True, exist_ok=True)

    cache_rows, malformed_records = cache_accounting(repo, cache_dir)
    parser = {
        "run_date": date.today().isoformat(),
        "contract": "common.py::audit_record and common.py::audit_batch; malformed JSON/category schema is a parse failure",
        "response_units_are_not_records": True,
        "cache_accounting": cache_rows,
        "label_output_summary": output_label_summary(results, malformed_records),
        "conservative_handling": {
            "policy": "malformed replies map to no categories in the existing pipeline; a conservative replay would mark every record attached to a malformed reply as flagged",
            "status": "exact_record_replay_available_only_when_malformed_response_units_are_zero",
            "if_nonzero": "the cache does not retain the original prompt/group size for every historical line, so no record-level worst-case number is fabricated",
        },
    }
    write_json(out / "parser_accounting.json", parser)
    write_json(out / "parser_accounting_status.json", parser)

    # Safe aggregate-only copies.  Do not copy any *_audit, *_rules, *_detect, utility, or cache file.
    allow = {
        "lamp3_calibrated_redaction_qwen25_7b.json",
        "lamp3_oracle_exhaustive_sensitivity.json",
        "lamp3_baselines_qwen25_7b.json",
        "lamp3_secondary_exposure.json",
        "lamp3_analysis_review.json",
        "sim_crc.json",
    }
    copied = []
    for name in sorted(allow):
        src = results / name
        if src.exists():
            dst = out / name
            shutil.copy2(src, dst)
            copied.append(name)

    fresh = {
        "parser_accounting": "completed" if any(v["exists"] for v in parser["cache_accounting"].values()) else "not_started",
        "matched_calibrated_redaction": "completed" if (results / "lamp3_calibrated_redaction_qwen25_7b.json").exists() else "not_started",
        "oracle_epsilon_and_exhaustive_slice": "completed" if (results / "lamp3_oracle_exhaustive_sensitivity.json").exists() else "not_started",
        "lamp3_grid_envelope": "completed" if (results / "lamp3_secondary_exposure.json").exists() else "not_started",
        "fixed_gate_comparator": "available_from_preexisting_aggregate",
        "human_audit": "skipped_by_user_already_completed",
        "model_70b": "postponed_by_latest_user_instruction; model downloaded but no 70B inference run",
    }
    write_json(out / "experiment_status.json", {"run_date": date.today().isoformat(), "status": fresh,
                                                  "copied_aggregate_outputs": copied,
                                                  "raw_and_cache_policy": "raw rows, prompts, labels, and caches remain under artifact/results and are not copied into result"})
    summary = [
        "# Review-requested experiment results",
        "",
        f"Updated: {date.today().isoformat()}",
        "",
        "Human audit was skipped because it was already completed. The 70B model was downloaded but inference was postponed by the latest instruction.",
        "",
        "Completed additional runs are listed in `experiment_status.json`; parser response counts are in `parser_accounting.json`.",
        "",
        "The result folder contains aggregate outputs only. Raw benchmark rows, prompts, row-level auditor labels, and LLM caches remain in the artifact workspace.",
        "",
    ]
    (out / "review_experiment_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps({"result": str(out), "copied": copied, "status": fresh}, indent=2))


if __name__ == "__main__":
    main()
