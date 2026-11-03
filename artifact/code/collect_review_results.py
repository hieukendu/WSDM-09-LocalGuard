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
from datetime import date
from pathlib import Path
from typing import Any


CATS = {
    "AGE_GENDER", "RACE_NATIONALITY", "LOCATION", "APPEARANCE", "PARTNER", "FINANCE",
    "OCCUPATION", "FAMILY", "HEALTH", "MENTAL_HEALTH", "EDUCATION", "PET",
    "SEXUAL_ORIENTATION", "NAME_CONTACT",
}
TASKS = ("lamp3", "lamp7", "lampqa")


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


def cache_accounting(cache_dir: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
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
            }
            if path.exists():
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        val = json.loads(line).get("v", "")
                    except Exception:
                        val = ""
                    valid, known = parse_response(val)
                    rec["response_units"] += 1
                    rec["known_record_units"] += known
                    if valid:
                        rec["valid_response_units"] += 1
                        rec["valid_record_units"] += known
                    else:
                        rec["malformed_response_units"] += 1
                        # A malformed batch has no prompt stored in the released cache, so its
                        # exact record multiplicity is intentionally not guessed.
                        rec["malformed_record_units_unknown"] += max(1, known)
                rec["malformed_response_rate"] = (rec["malformed_response_units"] /
                                                   rec["response_units"] if rec["response_units"] else None)
            else:
                rec["malformed_response_rate"] = None
            rows[key] = rec
    return rows


def output_label_summary(results: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in TASKS:
        for tag, label in (("qwen25_7b", "qwen2.5:7b"), ("gemma2_9b_evaltop", "gemma2:9b"),
                           ("llama33_70b_evaltop", "llama3.3:70b")):
            path = results / f"{task}_audit_{tag}.json" if tag != "qwen25_7b" else results / f"{task}_audit.json"
            if not path.exists():
                continue
            d = read_json(path)
            records = [cats for user in d.values() for cats in user.values()]
            out[f"{task}|{label}"] = {
                "output": str(path), "users": len(d), "record_labels": len(records),
                "flagged_records": int(sum(bool(c) for c in records)),
                "flagged_rate": float(sum(bool(c) for c in records) / len(records)) if records else None,
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

    parser = {
        "run_date": date.today().isoformat(),
        "contract": "common.py::audit_record and common.py::audit_batch; malformed JSON/category schema is a parse failure",
        "response_units_are_not_records": True,
        "cache_accounting": cache_accounting(cache_dir),
        "label_output_summary": output_label_summary(results),
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
