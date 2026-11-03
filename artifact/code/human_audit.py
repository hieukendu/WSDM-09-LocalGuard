
import argparse
import csv
import json
import random
import secrets
from pathlib import Path

from common import AUDIT_HEAD, CATS
from human_exposure import evaluate_adjudicated, packet_coverage


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_labels(rows, expected):
    out = {}
    for row in rows:
        ident = row["audit_id"]
        if ident not in expected or ident in out:
            raise ValueError("Unknown or duplicate audit ID")
        raw = row.get("human_discloses", "").strip()
        if raw not in ("0", "1"):
            raise ValueError("Every item needs a human_discloses value of 0 or 1")
        cats = set(filter(None, (x.strip() for x in row.get("human_categories", "").split(";"))))
        if cats - set(CATS) or bool(cats) != bool(int(raw)):
            raise ValueError("Invalid categories or disagreement between binary and category labels")
        out[ident] = cats
    if set(out) != set(expected):
        raise ValueError("Incomplete annotation packet")
    return out


def agreement(a, b):
    n = len(a)
    if not n:
        return {"n": 0, "agreement": None, "kappa": None}
    p = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return {"n": n, "agreement": p, "kappa": (p - pe) / (1 - pe) if pe < 1 else None}


def precision_recall(pred, truth):
    tp = sum(p and t for p, t in zip(pred, truth))
    fp = sum(p and not t for p, t in zip(pred, truth))
    fn = sum(not p and t for p, t in zip(pred, truth))
    return {"n": len(truth), "tp": tp, "fp": fp, "fn": fn,
            "tn": len(truth) - tp - fp - fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def prepare(args):
    root = Path(args.results)
    rows = read_csv(root / "handcheck/handcheck_sheet.csv")
    if any(r["human_discloses"].strip() or r["human_categories"].strip() for r in rows):
        raise ValueError("Source sheet already contains annotations; do not overwrite their provenance")
    out = Path(args.output)
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    audits = {}
    key, packets = {}, []
    for r in rows:
        task = r["task"]
        if task not in audits:
            audits[task] = [json.loads((root / f"{task}{suffix}").read_text()) for suffix in
                            ("_audit.json", "_audit_gemma2_9b_evaltop.json")]
        ident = secrets.token_hex(16)
        qwen, gemma = [a.get(r["uid"], {}).get(r["rec"]) for a in audits[task]]
        key[ident] = {"task": task, "uid": r["uid"], "rec": r["rec"],
                      "qwen": qwen, "gemma": gemma}
        packets.append({"audit_id": ident, "text": r["text"],
                        "human_discloses": "", "human_categories": ""})
    for name in ("annotator_a", "annotator_b"):
        shuffled = list(packets)
        random.SystemRandom().shuffle(shuffled)
        write_csv(out / f"{name}.csv", shuffled, list(packets[0]))
    (out / "PRIVATE_KEY.json").write_text(json.dumps(key, indent=2))
    (out / "attestation.json").write_text(json.dumps({
        "annotator_a": {"pseudonym": "", "human": False, "independent": False},
        "annotator_b": {"pseudonym": "", "human": False, "independent": False},
        "adjudicator": {"pseudonym": "", "human": False}}, indent=2))
    (out / "INSTRUCTIONS.md").write_text(
        "# Private human annotation packet\n\nDo not publish this directory. Each person receives only their CSV and these instructions. "
        "Do not consult the other annotator, model labels, or an LLM. Judge the displayed text only. "
        "Use 0 or 1 in human_discloses; for 1 list all applicable category names separated by semicolons; "
        "for 0 leave categories empty. The same text may have several categories. Do not infer unstated facts. "
        "The coordinator retains PRIVATE_KEY.json and records truthful attestations after both independent passes. "
        "A human adjudicates category-set disagreements only after those passes are locked.\n\n" + AUDIT_HEAD)
    print(json.dumps({"n": len(rows), "gemma_available": sum(v["gemma"] is not None for v in key.values()),
                      "status": "awaiting_two_independent_humans"}))


def score(args):
    root = Path(args.packet)
    key = json.loads((root / "PRIVATE_KEY.json").read_text())
    att = json.loads((root / "attestation.json").read_text())
    people = [att[k] for k in ("annotator_a", "annotator_b")]
    if not all(p.get("human") is True and p.get("independent") is True and p.get("pseudonym") for p in people):
        raise ValueError("Two truthful independent-human attestations are required")
    if people[0]["pseudonym"] == people[1]["pseudonym"]:
        raise ValueError("Two distinct human annotators are required")
    a, b = [parse_labels(read_csv(root / f"{name}.csv"), key) for name in ("annotator_a", "annotator_b")]
    disagreed = {i for i in key if a[i] != b[i]}
    report = {"scope": "unweighted descriptive metrics on a Qwen-stratified record sample; not population estimates",
              "n": len(key), "category_set_disagreements": len(disagreed), "agreement": {}}
    for cat in ["ANY"] + CATS:
        binary = lambda labels, i: bool(labels[i]) if cat == "ANY" else cat in labels[i]
        report["agreement"][cat] = agreement([binary(a, i) for i in key], [binary(b, i) for i in key])
    adjudication = root / "adjudication.csv"
    if disagreed and not adjudication.exists():
        texts = {r["audit_id"]: r["text"] for r in read_csv(root / "annotator_a.csv")}
        rows = [{"audit_id": i, "text": texts[i], "human_discloses": "", "human_categories": ""}
                for i in sorted(disagreed)]
        write_csv(adjudication, rows, list(rows[0]))
        report["status"] = "awaiting_human_adjudication"
    else:
        gold = {i: a[i] for i in key if i not in disagreed}
        if disagreed:
            if att["adjudicator"].get("human") is not True or not att["adjudicator"].get("pseudonym"):
                raise ValueError("A truthful human adjudicator attestation is required")
            gold.update(parse_labels(read_csv(adjudication), disagreed))
        common = [i for i, r in key.items() if r["qwen"] is not None and r["gemma"] is not None]
        report.update(status="human_record_audit_complete_recalibration_not_identified", metrics={})
        for scope, ids in [("all_qwen_covered", [i for i in key if key[i]["qwen"] is not None]),
                           ("common_coverage", common)]:
            report["metrics"][scope] = {}
            for model in (["qwen"] if scope == "all_qwen_covered" else ["qwen", "gemma", "union"]):
                pred = {i: set(key[i]["qwen"]) | set(key[i]["gemma"]) if model == "union"
                        else set(key[i][model]) for i in ids}
                report["metrics"][scope][model] = {}
                for cat in ["ANY"] + CATS:
                    test = lambda cats: bool(cats) if cat == "ANY" else cat in cats
                    report["metrics"][scope][model][cat] = precision_recall(
                        [test(pred[i]) for i in ids], [test(gold[i]) for i in ids])
        report["exposure_recalibration"] = "exact recalibration not identified unless missing-label intervals collapse; missing candidates cannot be zero-filled"
        if args.results:
            report["partial_identification"] = evaluate_adjudicated(args.results, key, gold)
    (root / "aggregate_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"status": report["status"], "n": len(key), "disagreements": len(disagreed)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--results", required=True)
    prep.add_argument("--output", required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--packet", required=True)
    scoring.add_argument("--results", help="Optional private results directory for human exposure/operating-point bounds")
    coverage = sub.add_parser("coverage")
    coverage.add_argument("--packet", required=True)
    coverage.add_argument("--results", required=True)
    args = parser.parse_args()
    if args.command == "coverage":
        key = json.loads((Path(args.packet) / "PRIVATE_KEY.json").read_text())
        print(json.dumps(packet_coverage(args.results, key), indent=2))
    else:
        (prepare if args.command == "prepare" else score)(args)


if __name__ == "__main__":
    main()
