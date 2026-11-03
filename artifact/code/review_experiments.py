"""Run the review-requested analyses that are reproducible from the released artifact.

The public artifact intentionally omits row-level benchmark data, auditor outputs,
reader utilities, admission sets, and model caches.  This driver therefore does two
things explicitly:

* recomputes the review-relevant aggregate comparisons that are present in the
  released JSON (fine/coarse grid sensitivity and fixed-gate comparisons); and
* records, without fabricating values, which requested experiments need omitted
  private inputs or a model endpoint.

Usage from the repository root::

    python3 artifact/code/review_experiments.py --out result

All files written by this script are aggregate/diagnostic outputs only.  It never
copies raw text, row-level labels, prompts, caches, or model outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any


TASKS = ("lampqa", "lamp3", "lamp7", "lamp2")
MAIN_TASKS = ("lampqa", "lamp3", "lamp7")
LOSSES = ("count", "severity2")
ALPHAS = (0.2, 0.5)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")


def mean_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def copy_aggregate_snapshot(src: Path, out: Path) -> list[str]:
    dest = out / "preexisting_aggregate"
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in sorted(src.glob("*.json")):
        # Only aggregate JSON is present in the checked-in artifact.  Keep the
        # allow-list here so this remains safe if private files are materialized
        # locally later.
        if any(token in path.name for token in ("_audit.json", "_detect.json", "_rules.json", "_reader_utility")):
            continue
        target = dest / path.name
        shutil.copy2(path, target)
        copied.append(str(target.relative_to(out)))
    return copied


def grid_sensitivity(results: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for task in TASKS:
        path = results / f"{task}_secondary_exposure.json"
        if not path.exists():
            continue
        data = read_json(path)
        coverage = data.get("reader_coverage", {})
        for alpha in ALPHAS:
            for loss in LOSSES:
                coarse_key = f"coarse|{loss}|{alpha}|all|pooled"
                fine_key = f"fine|{loss}|{alpha}|all|pooled"
                if coarse_key not in data["cells"] or fine_key not in data["cells"]:
                    continue
                coarse = data["cells"][coarse_key]
                fine = data["cells"][fine_key]
                rows.append(
                    {
                        "task": task,
                        "alpha": alpha,
                        "loss": loss,
                        "coarse_grid_size": data["grid_sizes"]["coarse"],
                        "fine_grid_size": data["grid_sizes"]["fine"],
                        "coarse_mean_test_loss": coarse["mean_test_loss"],
                        "fine_mean_test_loss": fine["mean_test_loss"],
                        "delta_loss_fine_minus_coarse": fine["mean_test_loss"] - coarse["mean_test_loss"],
                        "coarse_mean_size": coarse["mean_size"],
                        "fine_mean_size": fine["mean_size"],
                        "delta_size_fine_minus_coarse": fine["mean_size"] - coarse["mean_size"],
                        "coarse_fraction_test_means_above_alpha": coarse["fraction_test_means_above_alpha"],
                        "fine_fraction_test_means_above_alpha": fine["fraction_test_means_above_alpha"],
                        "reader_coverage": coverage,
                        "maximum_proxy_reconstruction_error": data.get("maximum_proxy_reconstruction_error"),
                    }
                )
        task_rows = [r for r in rows if r["task"] == task]
        if task_rows:
            max_row = max(task_rows, key=lambda r: r["delta_size_fine_minus_coarse"])
            summary[task] = {
                "grid_sizes": data["grid_sizes"],
                "profile_length_cut_training_median": data.get("profile_length_cut_training_median"),
                "bin_counts": data.get("bin_counts"),
                "maximum_proxy_reconstruction_error": data.get("maximum_proxy_reconstruction_error"),
                "largest_admission_increase": {
                    "alpha": max_row["alpha"],
                    "loss": max_row["loss"],
                    "delta_size": max_row["delta_size_fine_minus_coarse"],
                },
                "reader_coverage": coverage,
            }
    return rows, summary


def fixed_gate_comparator(results: Path) -> dict[str, Any]:
    """Summarize the existing fixed sensitivity-threshold rows.

    These are not newly calibrated redaction runs.  They are the artifact's
    training-fixed cleared-similarity thresholds, included because the review
    allows a faithful fixed-gate comparator as the secondary comparison.
    """

    out: dict[str, Any] = {
        "status": "completed_from_existing_aggregate",
        "interpretation": "Training-fixed cleared-similarity thresholds; not a new utility-aware selector and not a calibrated redaction family.",
        "tasks": {},
    }
    for task in MAIN_TASKS:
        path = results / f"{task}_analysis.json"
        if not path.exists():
            continue
        data = read_json(path)
        fixed = data.get("fixed", {})
        rows = []
        for name, value in fixed.items():
            if not name.startswith("cert_fix") or not isinstance(value, dict):
                continue
            rows.append(
                {
                    "rule": name,
                    "mean_admitted_records": value.get("size"),
                    "utility_mean": (value.get("utility") or [None])[0],
                    "exposure_qwen": value.get("exposure_qwen"),
                    "exposure_gemma": value.get("exposure_gemma"),
                    "exposure_union": value.get("exposure_union"),
                    "any_union": value.get("any_union"),
                    "sensitive_union": value.get("sens_union"),
                }
            )
        rows.sort(key=lambda row: int(row["rule"].replace("cert_fix", "")))
        by_target: dict[str, Any] = {}
        for alpha in ALPHAS:
            valid = [r for r in rows if r["exposure_union"] is not None]
            if valid:
                closest = min(valid, key=lambda r: abs(r["exposure_union"] - alpha))
                by_target[str(alpha)] = {
                    "closest_rule_by_union_exposure": closest["rule"],
                    "union_exposure": closest["exposure_union"],
                    "utility_mean": closest["utility_mean"],
                    "mean_admitted_records": closest["mean_admitted_records"],
                }
        out["tasks"][task] = {"rows": rows, "closest_to_target": by_target}
    return out


def baseline_summary(results: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "existing_uncalibrated_baselines_only",
        "calibrated_redaction_status": "not_run",
        "reason": "A matched calibrated redaction family needs row-level users, detector/auditor labels, admission sets and reader utilities, which are excluded from the public artifact.",
        "models": {},
    }
    for tag in ("qwen25_7b", "qwen25_14b"):
        path = results / f"lamp3_baselines_{tag}.json"
        if not path.exists():
            continue
        data = read_json(path)
        out["models"][tag] = {
            "n": data.get("n"),
            "none": data.get("none"),
            "top5": data.get("top5"),
            "existing_baselines": {
                key: data.get(key)
                for key in ("ratings", "redact_det", "redact_aud")
                if key in data
            },
        }
    return out


def oracle_summary(results: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "single_epsilon_aggregate_only",
        "requested_addition": "epsilon sensitivity, feasible-only exposure, and full-pool versus no-profile reference",
        "sensitivity_curve": "not_run",
        "reason": "The checked-in artifact retains only aggregate epsilon=0.05 summaries; row-level oracle witnesses and utilities are excluded.",
        "tasks": {},
    }
    for task in MAIN_TASKS:
        path = results / f"{task}_analysis.json"
        if not path.exists():
            continue
        data = read_json(path)
        oracle = data.get("oracle")
        if oracle:
            out["tasks"][task] = {
                "n": oracle.get("n"),
                "epsilon": oracle.get("eps"),
                "max_records": oracle.get("max_records"),
                "oracle": oracle.get("oracle"),
                "methods": oracle.get("methods"),
                "warning": "Method exposure/gap aggregates are not conditioned on sufficient users in the released summary.",
            }
    return out


def parser_accounting(results: Path, repo: Path) -> dict[str, Any]:
    # Reproduce the parser contract on a tiny, deterministic fixture.  This is a
    # parser behavior check, not a claim about production malformed-response rates.
    valid = '{"categories": ["HEALTH", "NOT_A_CATEGORY"]}'
    malformed = '{"categories": ["HEALTH"'
    missing = '{"other": ["HEALTH"]}'

    def parse(raw: str) -> list[str]:
        try:
            value = json.loads(raw).get("categories")
            if not isinstance(value, list):
                return []
            allowed = {
                "AGE_GENDER", "RACE_NATIONALITY", "LOCATION", "APPEARANCE", "PARTNER", "FINANCE",
                "OCCUPATION", "FAMILY", "HEALTH", "MENTAL_HEALTH", "EDUCATION", "PET",
                "SEXUAL_ORIENTATION", "NAME_CONTACT",
            }
            return sorted({str(x) for x in value if x in allowed})
        except Exception:
            return []

    private_files = sorted(
        str(p.relative_to(repo))
        for pattern in ("artifact/results/*_audit.json", "artifact/results/llm_cache/*")
        for p in repo.glob(pattern)
        if p.is_file()
    )
    return {
        "status": "parser_contract_checked_full_run_counts_unavailable",
        "synthetic_contract_check": {
            "valid_categories": parse(valid),
            "malformed_reply_maps_to": parse(malformed),
            "missing_categories_maps_to": parse(missing),
        },
        "full_run_counts": None,
        "raw_response_or_cache_files_found": private_files,
        "reason": "Per-call raw auditor responses/caches are intentionally excluded; malformed replies therefore cannot be counted by task/auditor or replayed under conservative parsing.",
        "source_parser": "artifact/code/common.py::audit_record",
        "conservative_reanalysis": "not_run",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(out: Path, grid_rows: list[dict[str, Any]], grid_summary: dict[str, Any], statuses: dict[str, Any]) -> None:
    lines = [
        "# Review-requested experiment run",
        "",
        f"Run date: {date.today().isoformat()}",
        "",
        "This folder contains aggregate-only outputs. Human audit was not rerun, and the 70B experiment was not run, as requested.",
        "",
        "## Completed from the checked-in artifact",
        "",
        "- Coarse (8-point) versus fine training-fixed grid sensitivity was extracted for four tasks, two losses, and targets 0.2/0.5.",
        "- Existing training-fixed cleared-similarity thresholds were summarized as the allowed fixed-gate comparator.",
        "- Existing uncalibrated LaMP-3 field-level baselines and the single-epsilon oracle summaries were collected without relabeling or rerunning models.",
        "- The parser contract was checked on deterministic fixtures; this is not a production parse-failure rate.",
        "",
        "## Key grid observations",
        "",
    ]
    for task, summary in grid_summary.items():
        inc = summary["largest_admission_increase"]
        lines.append(
            f"- `{task}`: fine grid has {summary['grid_sizes']['fine']} points versus {summary['grid_sizes']['coarse']} coarse points; largest mean admission increase is {inc['delta_size']:.3f} records at alpha={inc['alpha']} ({inc['loss']})."
        )
    lines += [
        "",
        "## Not executable in this workspace",
        "",
        "- Matched calibrated redaction: requires omitted row-level data/labels/utilities and model calls.",
        "- Full parser-failure counts and conservative retry/relabel sensitivity: requires omitted raw responses or caches.",
        "- Oracle epsilon curve, feasible-only exposure, and exhaustive slice: requires omitted row-level oracle witnesses/utilities.",
        "- Exact envelope-versus-grid decomposition: aggregate files contain coarse/fine grid results but not row-level raw/envelope losses.",
        "",
        "See `experiment_status.json` for input-level reasons and `grid_sensitivity.csv` / JSON files for machine-readable values.",
        "",
    ]
    (out / "review_experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    results = repo / "artifact" / "results"
    out = (args.out or (repo / "result")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    copied = copy_aggregate_snapshot(results, out)
    grid_rows, grid_summary = grid_sensitivity(results)
    write_json(out / "grid_sensitivity.json", {"rows": grid_rows, "summary": grid_summary})
    write_csv(out / "grid_sensitivity.csv", grid_rows)

    fixed = fixed_gate_comparator(results)
    write_json(out / "fixed_gate_comparator.json", fixed)
    baseline = baseline_summary(results)
    write_json(out / "redaction_baseline_status.json", baseline)
    oracle = oracle_summary(results)
    write_json(out / "oracle_sensitivity_status.json", oracle)
    parser_result = parser_accounting(results, repo)
    write_json(out / "parser_accounting_status.json", parser_result)

    statuses = {
        "human_audit": {
            "status": "skipped_by_user_already_completed",
            "source": "PUBLIC-STATUS.md and artifact/results/human_audit_aggregate.json",
        },
        "model_70b": {"status": "skipped_by_user"},
        "grid_sensitivity": {
            "status": "completed_from_existing_aggregate",
            "output": "grid_sensitivity.json",
            "n_rows": len(grid_rows),
        },
        "fixed_gate_comparator": {"status": "completed_from_existing_aggregate"},
        "matched_calibrated_redaction": {
            "status": "blocked_missing_private_inputs_and_model_endpoint",
            "output": "redaction_baseline_status.json",
        },
        "parser_failure_accounting": {
            "status": "blocked_missing_raw_responses_or_cache",
            "output": "parser_accounting_status.json",
        },
        "oracle_sensitivity": {
            "status": "blocked_missing_row_level_oracle_outputs_and_utilities",
            "output": "oracle_sensitivity_status.json",
        },
        "envelope_decomposition": {
            "status": "partial_grid_comparison_only_missing_row_level_losses",
            "output": "grid_sensitivity.json",
        },
        "copied_preexisting_aggregate_files": copied,
    }
    write_json(out / "experiment_status.json", statuses)
    write_summary(out, grid_rows, grid_summary, statuses)
    print(json.dumps({"out": str(out), "grid_rows": len(grid_rows), "copied": len(copied)}, indent=2))


if __name__ == "__main__":
    main()
