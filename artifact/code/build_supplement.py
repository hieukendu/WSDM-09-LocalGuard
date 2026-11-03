"""Fail-closed allowlisted anonymous export. Never package raw or row-level data.

Every destination must be new. No repository upload or credential handling occurs here.
"""
import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

CODE = """analyze analyze_r1 analyze_r2 baselines_lamp3 common crc gate heldout
human_audit human_exposure lamp2_pilot make_assets make_secondary_tables oracle paperfig pilot2
prep prep_extend qa_pilot replication run secondary_exposure cumulative_exposure
shift_mech sim_crc validate_audit handcheck_sheet test_human_audit
test_secondary_exposure test_human_exposure test_supplement build_supplement""".split()
SHELL = """chain10_70b chain11_lamp2 chain11a_lamp2_laptop chain3 chain4_cpp chain4_qwen
chain5_lamp3_600 chain6 chain7 chain8 chain9_14b download_data extend_users pipeline_rest
reanalyze_1008 repl_chain2 repl_chain2b""".split()
RESULTS = [f"{task}_{kind}.json" for task in ["lampqa", "lamp3", "lamp7"]
           for kind in ["analysis", "shift_mech"]] + [
    "lamp3_analysis_qwen25_14b.json", "lamp3_analysis_llama31_8b.json", "lamp3_analysis_300users.json",
    "lamp3_baselines_qwen25_7b.json", "lamp3_baselines_qwen25_14b.json",
    "replication.json", "qa_pilot.json", "sim_crc.json"] + [
    f"{task}_{kind}.json" for task in ["lampqa", "lamp3", "lamp7", "lamp2"]
    for kind in ["secondary_exposure", "cumulative_exposure"]]
ROW_KEYS = {"uid", "user_id", "record_id", "pidx", "rec", "text", "prompt", "query",
            "examples", "categories", "audit_id", "indices", "annotator_a", "annotator_b"}


def check_aggregate(value):
    if isinstance(value, dict):
        if ROW_KEYS.intersection(value):
            raise ValueError("Row-level or text field in an aggregate export")
        for v in value.values():
            check_aggregate(v)
    elif isinstance(value, list):
        for v in value:
            check_aggregate(v)


def scan_text(text, terms):
    patterns = [r"hf_[A-Za-z0-9]{20,}", "/" + r"Users/[^/\s]+", "/" + r"home/[^/\s]+",
                r"103[.]73[.]232[.][0-9]+", r"-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"]
    hits = sum(bool(re.search(pattern, text)) for pattern in patterns)
    return hits + sum(term.casefold() in text.casefold() for term in terms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-manuscript", action="store_true",
                        help="Only after separate content/privacy review; unsupported input dependencies fail closed")
    parser.add_argument("--private-term", action="append", default=[])
    args = parser.parse_args()
    project, dest = args.project.resolve(), args.output.resolve()
    if dest.exists() or dest.with_suffix(".zip").exists():
        raise ValueError("Use a new output path; existing releases are never overwritten")
    selected = {"README.md": "artifact/SUPPLEMENT-README.md"}
    for name in ["DATA.md", "PROOFS.md", "SUPPLEMENT.md", "SECONDARY-TABLES.md"]:
        selected[f"artifact/{name}"] = f"artifact/{name}"
    for name in CODE:
        selected[f"artifact/code/{name}.py"] = f"artifact/code/{name}.py"
    for name in SHELL:
        selected[f"artifact/code/{name}.sh"] = f"artifact/code/{name}.sh"
    for name in RESULTS:
        selected[f"artifact/results/{name}"] = f"artifact/results/{name}"
    if args.include_manuscript:
        for name in ["main.tex", "references.bib", "main.pdf"]:
            selected[f"manuscript/{name}"] = f"manuscript/{name}"
        # Do not silently include newly introduced real-user walkthrough assets.
        for path in (project / "manuscript/sections").glob("*.tex"):
            if re.search(r"\\input\{figures/|\\Fig(?:Uid|Query|Cands)", path.read_text()):
                raise ValueError("Manuscript contains an unreviewed per-user figure dependency; privacy review required")
    folders = [("generated", {".tex", ".pdf"})]
    if args.include_manuscript:
        folders.append(("sections", {".tex"}))
    for folder, suffixes in folders:
        for path in sorted((project / "manuscript" / folder).iterdir()):
            if path.suffix in suffixes:
                rel = str(path.relative_to(project))
                selected[rel] = rel
    payload = {}
    for target, source in selected.items():
        path = project / source
        if path.is_symlink() or not path.is_file() or project not in path.resolve().parents:
            raise ValueError("Missing, symlinked or escaping allowlisted file")
        content = path.read_bytes()
        if path.suffix == ".json":
            check_aggregate(json.loads(content))
        if path.suffix == ".pdf":
            extracted = subprocess.check_output(["pdftotext", str(path), "-"], text=True)
            metadata = subprocess.check_output(["pdfinfo", str(path)], text=True)
            text = extracted + metadata
        else:
            text = content.decode("utf-8")
        if scan_text(text, args.private_term):
            raise ValueError(f"Anonymity/secret scan failed: {target}")
        payload[target] = content
    # Pilot examples are model outputs, even if short; export only numeric summaries.
    pilot = json.loads((project / "artifact/results/lamp2_pilot.json").read_text())
    clean_pilot = {k: pilot[k] for k in ["n", "acc_none", "acc_top5", "gain", "go"]}
    check_aggregate(clean_pilot)
    payload["artifact/results/lamp2_pilot.json"] = (json.dumps(clean_pilot, indent=2, allow_nan=False) + "\n").encode()
    manifest = {"scope": "anonymous aggregate/code supplement; excludes all raw/processed and row-level derived data",
                "manuscript_included": args.include_manuscript,
                "files": [{"path": path, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                          for path, content in sorted(payload.items())],
                "scan": {"allowlist_passed": True, "aggregate_schema_passed": True,
                         "text_and_pdf_anonymity_passed": True},
                "excluded": ["raw and processed benchmark text", "per-record labels and indices", "per-user utilities and sets",
                             "cached prompts and outputs", "human packets and keys", "logs", "credentials", "Git history"]}
    payload["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    dest.mkdir(parents=True)
    for name, content in payload.items():
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    archive = dest.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zf.writestr(info, content)
    # Verify the exact archive contents and hashes after writing.
    with zipfile.ZipFile(archive) as zf:
        if set(zf.namelist()) != set(payload) or any(zf.read(k) != v for k, v in payload.items()):
            raise ValueError("Archive verification failed")
    print(json.dumps({"files": len(payload), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                      "bytes": archive.stat().st_size, "scan": "passed"}))


if __name__ == "__main__":
    main()
